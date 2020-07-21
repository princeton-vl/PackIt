import time
from collections import deque
import sys
import multiprocessing

import numpy as np
import tensorflow as tf
import scipy as sci
import gc

from stable_baselines import logger
from stable_baselines.common import explained_variance, BaseRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger


class PPO2(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        should be smaller or equal than number of environments run in parallel.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param make_env: (function) To create a new env while updating
    """

    def __init__(self, policy, env, policy_config, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, make_env=None, zero_mean_advs=True, packing_id_start=0,
                 restore_exp=False, restore_path=""):

        super(PPO2, self).__init__(policy=policy, env=env, verbose=verbose, policy_base=ActorCriticPolicy,
                                   requires_vec_env=True)

        if isinstance(learning_rate, float):
            learning_rate = constfn(learning_rate)
        else:
            assert callable(learning_rate)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        #####
        self.policy_config = policy_config
        self.make_env = make_env
        self.zero_mean_advs = zero_mean_advs
        self.packing_id_start = packing_id_start
        self.restore_exp = restore_exp
        self.restore_path = restore_path
        #####

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):

            # prevent issues with circular imports
            from packing.packing_policy import PackingPolicy

            assert issubclass(self.policy, ActorCriticPolicy) or issubclass(self.policy, PackingPolicy), \
            "Error: the input policy for the PPO2 model must be " \
            "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                # act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                #                         n_batch_step, reuse=False)
                act_model = self.policy(sess=self.sess, reuse=False, **self.policy_config)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    # train_model = self.policy(self.sess, self.observation_space, self.action_space,
                    #                           self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                    #                           reuse=True)
                    train_model = self.policy(sess=self.sess, reuse=True, **self.policy_config)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_fn
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model.value_fn - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph)))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leiber', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                    grads = tf.gradients(loss, self.params, colocate_gradients_with_ops=True)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                # applying the batchnorm update ops with training
                self._train = tf.group(train_model.batchnorm_updates_op, trainer.apply_gradients(grads))

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.histogram('discounted_rewards', self.rewards_ph)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.histogram('learning_rate', self.learning_rate_ph)
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.histogram('advantage', self.advs_ph)
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.histogram('clip_range', self.clip_range_ph)
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))
                    tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                    tf.summary.histogram('final_value_pred', vpred)
                    tf.summary.histogram('final_value_pred_clipped',  vpredclipped)
#                     if len(self.observation_space.shape) == 3:
#                         tf.summary.image('observation', train_model.obs_ph)
#                     else:
#                         tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

                self.Saver = tf.train.Saver()
                if self.restore_exp:
                    self.restore_weights(self.restore_path)


    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        """
        # prevent issues with circular imports
        from packing.packing_policy import PackingPolicy, get_feed_dict_rl

        advs = returns - values
        if self.zero_mean_advs:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        else:
            advs = (advs) / (advs.std() + 1e-8)

        if issubclass(self.policy, PackingPolicy):
            temp = get_feed_dict_rl(self.train_model, obs, is_training=True)
            td_map = {self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
            td_map.update(temp)
        else:
            td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def evaluate(
        self,
        pack_file_name,
        evaluate_first_n=80,
        beam_search=False,
        beam_size=1,
        back_track_search=False,
        budget=4,
        rot_before_mov_env=None):

        from packing.packing_evalute import evaluate
        from packing.packing_heuristic import HeuristicModel, sha_lar, \
            mov_best, rot_best, rot_best_pos

        if rot_before_mov_env is None:
            rot_before_mov_env = self.policy_config['rot_before_mov']

        if rot_before_mov_env:
            model_oracle = HeuristicModel(
                sha_lar,
                rot_best,
                mov_best)
        else:
            model_oracle = HeuristicModel(
                sha_lar,
                mov_best,
                rot_best_pos)

        if beam_search or back_track_search:
            n_envs = 1
            batch_size = 1

        else:
            n_envs = 8
            batch_size = 10

        reward = evaluate(
            pack_file_name=pack_file_name,
            model=self.act_model,
            n_envs=n_envs,
            env_name='unity/envs/packit',
            rot_before_mov=rot_before_mov_env,
            batch_size=batch_size,
            save_sup_data=False,
            evaluate_first_n=evaluate_first_n,
            worker_id_start=(500 + self.packing_id_start),
            env_config={
                'sha': model_oracle.action_best if self.policy_config['comp_pol_config']['sha_pol'] is None else None,
                'mov': model_oracle.action_best,
                'rot': model_oracle.action_best if self.policy_config['comp_pol_config']['rot_pol'] is None else None
            },
            beam_search=beam_search,
            beam_size=beam_size,
            back_track_search=back_track_search,
            budget=budget)

        avg_reward = sum(reward) / len(reward)
        succ = [(x == 1) for x in reward]
        per_succ = sum(succ) / len(reward)
        eff = [x+1 if x<0 else x for x in reward]
        avg_eff = sum(eff) / len(eff)

        return avg_reward, avg_eff, per_succ, reward

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="PPO2"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            from packing.packing_policy import PackingPolicy
            if issubclass(self.policy, PackingPolicy):
                is_packing_env = True
            else:
                is_packing_env = False

            if not is_packing_env:
                runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                                is_packing_env=is_packing_env)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch
            best_avg_eff_va = 0
            best_avg_eff_tr = 0
            for update in range(nupdates + 1):

                ###########################################################################
                # very temporary fix
                if update % 20 == 0:
                    pack_file_name_va = [ "pack_va/" + str(i) + "_va" for i in range(0, 50)]
                    pack_file_name_tr = [ "pack_tr/" + str(i) + "_tr" for i in range(0, 50)]

                    avg_reward_va, avg_eff_va, per_succ_va, _ = self.evaluate(pack_file_name_va)
                    avg_reward_tr, avg_eff_tr, per_succ_tr, _ = self.evaluate(pack_file_name_tr)

                    log_path = "{}_{}".format(writer.get_logdir(), "log")
                    with open(log_path, "a+") as log_file:
                        log_file.write("Updata Number: {}\n".format(update))
                        log_file.write("Validation Average Reward: {}\n".format(avg_reward_va))
                        log_file.write("Validaition Average Efficiency: {}\n".format(avg_eff_va))
                        log_file.write("Validation Percentage of Success: {}\n\n".format(per_succ_va))
                        log_file.write("Training Average Reward: {}\n".format(avg_reward_tr))
                        log_file.write("Training Average Efficiency: {}\n".format(avg_eff_tr))
                        log_file.write("Training Percentage of Success: {}\n\n".format(per_succ_tr))

                    if avg_eff_va > best_avg_eff_va:
                        print("Saving best model on validation ....")
                        best_avg_eff_va = avg_eff_va
                        self.save_weights("{}/model_va".format(writer.get_logdir()))

                    if avg_eff_tr > best_avg_eff_tr:
                        print("Saving best model on training ....")
                        best_avg_eff_tr = avg_eff_tr
                        self.save_weights("{}/model_tr".format(writer.get_logdir()))

                    self.save_weights("{}/model_latest".format(writer.get_logdir()))

                ############################################################################

                assert self.n_batch % self.nminibatches == 0
                n_batch_train = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update / (nupdates + 1))
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)

                # for packing, we start a new env each update step
                # done so that the code could run smoothly
                # as otherwise there is memory/resource leakage
                if is_packing_env and (self.make_env is not None):
                    while True:
                        try:
                            self.env.close()
                            self.env = self.make_env()
                            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                                    is_packing_env=is_packing_env)
                            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                            self.env.close()
                        except:
                            print("Unable to complete the run.")
                            gc.collect()
                            continue
                        break
                else:
                    # true_reward is the reward without discount
                    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()

                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, n_batch_train):
                            timestep = ((update * self.noptepochs * self.n_batch + epoch_num * self.n_batch + start) //
                                        n_batch_train)
                            end = start + n_batch_train
                            mbinds = inds[start:end]

                            _obs = obs[mbinds]
                            slices = (arr[mbinds] for arr in (returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, _obs, *slices, writer=writer,
                                                                 update=timestep))
                else:  # recurrent version
                    assert self.n_envs % self.nminibatches == 0
                    envinds = np.arange(self.n_envs)
                    flatinds = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envsperbatch = n_batch_train // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(envinds)
                        for start in range(0, self.n_envs, envsperbatch):
                            timestep = ((update * self.noptepochs * self.n_envs + epoch_num * self.n_envs + start) //
                                        envsperbatch)
                            end = start + envsperbatch
                            mb_env_inds = envinds[start:end]
                            mb_flat_inds = flatinds[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, update=timestep,
                                                                 writer=writer, states=mb_states))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, update * (self.n_batch + 1))

                if callback is not None:
                    callback(locals(), globals())

                if self.verbose >= 1 and ((update + 1) % log_interval//100 == 0 or update == 0):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", (update + 1) * self.n_steps)
                    logger.logkv("nupdates", (update + 1))
                    logger.logkv("total_timesteps", (update + 1) * self.n_batch)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()
            return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)

        actions, _, states, _ = self.step(observation, state, mask, deterministic=deterministic)
        return actions, states

    def action_probability(self, observation, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)

        return self.proba_step(observation, state, mask)

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)


    def save_weights(self, path):
        self.Saver.save(self.sess, path)


    def restore_weights(self, path):
        self.Saver.restore(self.sess, path)


    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, is_packing_env):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps, is_packing_env=is_packing_env)
        self.lam = lam
        self.gamma = gamma
        self.is_packing_env = is_packing_env

    def run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        if self.is_packing_env:
            from packing.packing_env import PackingEnv
            mb_obs = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
        else:
            mb_obs = []
        mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], []
        mb_states = self.states
        ep_infos = []

        for _ in range(self.n_steps):

            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)

            if self.is_packing_env:
                mb_obs = sci.sparse.vstack((mb_obs, self.obs.copy()))
            else:
                mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            if self.is_packing_env:
                self.obs, rewards, self.dones, infos = self.env.step(actions)
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeep_info = info.get('episode')
                if maybeep_info:
                    ep_infos.append(maybeep_info)
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        if not self.is_packing_env:
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        if self.is_packing_env:
            cur_indices = np.arange(self.n_steps * self.env.num_envs)
            cur_indices = np.reshape(cur_indices, (self.n_steps, self.env.num_envs))
            cur_indices = cur_indices.swapaxes(0, 1).reshape(self.n_steps
                                                             * self.env.num_envs)
            mb_obs = mb_obs[cur_indices, :]
        else:
            mb_obs = swap_and_flatten(mb_obs)

        mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs = \
            map(swap_and_flatten, (mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return zero. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)

"""
reults before making changes for verification
n_envs = 2
n_steps = 10

before swap and flatten:
shape(mb_obs) = (10, 2, 32015650)
mb_obs =
[[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]

after swap and flatten:
shape(mb_obs) = (20, 32015650)
mb_obs =
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [3. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
"""
