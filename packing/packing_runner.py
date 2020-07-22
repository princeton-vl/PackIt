import json

import numpy as np
import scipy as sci
import pickle as pkl
import scipy.sparse as sparse

from packing.packing_env import PackingEnv
from packing.packing_env import mul_pro_packing_env


def unity_action_to_one_hot(unity_action, step, rot_before_mov):
    pos_act = np.zeros((1, PackingEnv.NUM_MOV))

    """
    Similar to get_gt_action in packing_env

    unity_gt_action = _unity_gt_action[2:]
    if self.step_typ == 1:
        return (int)(unity_gt_action[0])

    elif self.step_typ == 2:
        if self.rot_before_mov:
            return (int)(unity_gt_action[0])
        else:
            return (int)(unity_gt_action[2] \
                     + (unity_gt_action[1] * self.MOV_RES ) \
                     + (unity_gt_action[0] * (self.MOV_RES ** 2)))

    else:
        if self.rot_before_mov:
            return (int)(unity_gt_action[2] \
                     + (unity_gt_action[1] * self.MOV_RES ) \
                     + (unity_gt_action[0] * (self.MOV_RES ** 2)))
        else:
            return (int)(unity_gt_action[0]
    """

    assert step in [2, 3, 4]
    if step == 2:
        for _action in unity_action:
            pos_act[0, _action['x']] = 1

    elif step == 3:
        if rot_before_mov:
            for _action in unity_action:
                pos_act[0, _action['x']] = 1
        else:
            for _action in unity_action:
                pos_act[0,
                        (_action['z']
                         + (_action['y'] * PackingEnv.MOV_RES)
                         + (_action['x'] * (PackingEnv.MOV_RES ** 2)))] = 1

    else:
        if rot_before_mov:
            for _action in unity_action:
                pos_act[0,
                        (_action['z']
                         + (_action['y'] * PackingEnv.MOV_RES)
                         + (_action['x'] * (PackingEnv.MOV_RES ** 2)))] = 1
        else:
            for _action in unity_action:
                pos_act[0, _action['x']] = 1

    return pos_act


def post_process_pos_act(obs, act, pos_act_path, config, rot_before_mov=True):
    """ To post prcess the obs. It loads all the possible actions and
    post-processes it in suitable one hot form. Actions are matched to the
    corresponding observations. In case of issue, empty values are returned.
    """
    with open(pos_act_path) as file:
        all_data = json.load(file)

    if rot_before_mov:
        pos_act_data = all_data['gtRotBeforeMov2']
    else:
        pos_act_data = all_data['gtNotRotBeforeMov2']

    # find the policies for which we will extract pos_actions
    # use it to estimate the valid samples
    # calculation based on the assumption that each policy has same
    # number of samples
    sum_not_gt_poilicies = 0
    for key, value in config.items():
        if config[key] is None:
            sum_not_gt_poilicies += 1

    if not obs.shape[0] == (len(pos_act_data) * ( sum_not_gt_poilicies / 3)):
        obs = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
        act = np.zeros((0))
        pos_act = np.zeros((0, PackingEnv.NUM_MOV))
        print("Issue 1 with {}".format(pos_act_path))

    else:
        pos_act = np.zeros((obs.shape[0], PackingEnv.NUM_MOV))
        for i in range(obs.shape[0]):
            # +1 because step in unity and python differ by 1
            step = obs[i, 0] + 1
            step_num = obs[i, 1]

            found_pos_act = False
            # searching the correct action
            for j in range(len(pos_act_data)):
                if ((pos_act_data[j]['step'] == step)
                    and (pos_act_data[j]['stepNum'] == step_num)):
                    pos_act_data_i = pos_act_data[j]['action']
                    found_pos_act = True
                    break

            if found_pos_act:
                pos_act[i] = unity_action_to_one_hot(pos_act_data_i, step,
                                                     rot_before_mov)
                assert pos_act[i, act[i]] == 1
            else:
                obs = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
                act = np.zeros((0))
                pos_act = np.zeros((0, PackingEnv.NUM_MOV))
                print("Issue 2 with {}".format(pos_act_path))
                break

    return obs, act, pos_act


def remove_based_on_config(obs, act, config, rot_before_mov):
    """ Remove samples based on the config.

        Used in order to simulate the behaviour of an environment.
    """

    if rot_before_mov:
        step_typ_int_str = {
                1: 'sha',
                2: 'rot',
                3: 'mov'
            }
    else:
        step_typ_int_str = {
                1: 'sha',
                2: 'mov',
                3: 'rot'
            }

    obs_new = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
    act_new = np.zeros((0), dtype=np.int32)

    for i in range(obs.shape[0]):
        _obs = obs[i]
        _act = act[i]

        step_typ = _obs[0, 0]
        if config[step_typ_int_str[step_typ]] is None:
            obs_new = sci.sparse.vstack((obs_new, _obs))
            act_new = np.hstack((act_new, np.array(_act)))

    return obs_new, act_new


class RunnerStepSup(object):
    def __init__(
        self,
        env,
        n_steps):

        """ A runner that works for running each environment to specific steps
        """

        self.env = env
        self.n_envs = env.num_envs
        self.n_steps = n_steps
        self.first_time = True

    def run(self):
        """
        If the run is called first time, it will reset the evs, otherwise it
        will just run for the number of steps
        Assumption: obs from the env is sparse

        Return:
            obs (sci.sparse.csr_matrix[n_envs * n_steps, obs_size]):
            actions (np.array[n_envs * n_steps]):
        """

        if self.first_time:
            _obs = self.env.reset()
            _act = self.env.get_gt_action()

            obs = _obs
            act = _act

        else:
            obs = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
            act = np.zeros((0))
            # to get the first correct action
            _act = self.env.get_gt_action()

        # for the first time we already have one observation
        for i in range(self.n_steps - int(self.first_time)):
            _obs, _, _, _ = self.env.step(_act.tolist())
            _act = self.env.get_gt_action()

            obs = sci.sparse.vstack((obs, _obs))
            act = np.hstack((act, _act))

        self.first_time = False

        return obs, act


class LoadStepSup(object):
    """ It loads the prestored data rather than running an env. The data is
    loaded and kept in a buffer. Larger buffer size would mean better
    shuffling of data (if shuffle is True). All possible action for an
    observation are returned in one-hot form. For now, the possible actions
    are loaded, directly from the pre_compute_unity folder.
    """

    def __init__(
        self,
        env_name,
        pack_file_name,
        n_steps,
        buffer_size=None,
        rot_before_mov=True,
        shuffle=False,
        config={
            'sha': None,
            'mov': None,
            'rot': None,
        }):
        """
        Args:
            env_name (string), pack_file_name(list), n_steps(int), buffer(int),
            rot_before_mov(bool), shuffle(bool), config(dict)
        """

        assert rot_before_mov is True, "current version only implements \
            rot_before_move"

        assert ((config['sha'] is None) or (config['mov'] is None)
                or (config['rot'] is None))

        self.env_name = env_name
        self.n_steps = n_steps
        self.shuffle = shuffle
        self.rot_before_mov = rot_before_mov
        self.config = config
        if buffer_size is None:
            self.buffer_size = n_steps
        else:
            self.buffer_size = max(buffer_size, n_steps)

        from packing.packing_evalute import get_file_id_lst
        self.file_id_lst = get_file_id_lst(
            env_name=env_name,
            pack_file_name=pack_file_name)

        self.obs_buf = sci.sparse.csr_matrix((0, PackingEnv.obs_size))
        self.act_buf = np.zeros((0))
        self.pos_act_buf = np.zeros((0, PackingEnv.NUM_MOV))

        # for shuffling the packs
        self.file_id_index = 0
        self.num_packs = len(self.file_id_lst)
        self.file_id_indices = np.arange(self.num_packs)

    def run(self):
        """
        Return:
            obs (sci.sparse.csr_matrix[n_envs * n_steps, obs_size]):
            actions (np.array[n_envs * n_steps]):
        """

        assert self.obs_buf.shape[0] == self.act_buf.shape[0]
        assert self.obs_buf.shape[0] == self.pos_act_buf.shape[0]

        # load more data
        while self.obs_buf.shape[0] < self.buffer_size:
            if ((self.file_id_index % self.num_packs) == 0
                and self.shuffle):
                np.random.shuffle(self.file_id_indices)

            _file_id_index = self.file_id_indices[
                    self.file_id_index % self.num_packs]
            self.file_id_index += 1

            file_info = self.file_id_lst[_file_id_index][0].split('/')
            pack_id = self.file_id_lst[_file_id_index][1]
            assert len(file_info) <=2
            if(len(file_info)) == 2:
                dir_name = file_info[0]
                file_name = file_info[1]
            else:
                dir_name = ""
                file_name = file_info[0]

            sup_data_path = \
                "{}_Data/StreamingAssets/{}_sup_data/{}".format(
                    self.env_name,
                    dir_name,
                    "{}_{}".format(
                        file_name,
                        pack_id
                    ))
            with open(sup_data_path, "rb") as file:
                obs, act = pkl.load(file)

            obs, act = remove_based_on_config(obs, act, self.config,
                                              self.rot_before_mov)

            # for adding the new all possible actions data
            pos_act_path = \
            "{}_Data/StreamingAssets/{}_precompute_unity/{}".format(
                self.env_name,
                dir_name,
                "{}_{}".format(
                    file_name,
                    pack_id
                ))

            obs, act, pos_act = post_process_pos_act(obs, act,
                                                     pos_act_path,
                                                     self.config,
                                                     self.rot_before_mov)

            self.obs_buf = sci.sparse.vstack((self.obs_buf, obs))
            self.act_buf = np.hstack((self.act_buf, np.array(act)))
            self.pos_act_buf = np.vstack((self.pos_act_buf, pos_act))

        if self.shuffle:
            index = np.arange(self.obs_buf.shape[0])
            np.random.shuffle(index)
            self.obs_buf = self.obs_buf[index]
            self.act_buf = self.act_buf[index]
            self.pos_act_buf = self.pos_act_buf[index]

        obs = self.obs_buf[0: self.n_steps]
        self.obs_buf = self.obs_buf[self.n_steps:]

        act = self.act_buf[0: self.n_steps]
        self.act_buf = self.act_buf[self.n_steps:]

        pos_act = self.pos_act_buf[0: self.n_steps]
        self.pos_act_buf = self.pos_act_buf[self.n_steps:]

        return obs, act, pos_act


class RunnerEpisode(object):
    def __init__(
        self,
        env,
        model,
        n_episodes):

        """ A runner that works for testing.

            Args:
                model (the policy for predictining the best action):
                    there should be a function model.action_best(obs)
                n_episodes: n_episodes to run per env
        """

        self.env = env
        self.n_envs = env.num_envs
        self.model = model
        self.n_episodes = n_episodes

    def run(self, return_supervised_data):
        """
            It will first reset the env, and then run each env for n_episodes
            using the action returned by the model.

            WARNING: For SubProcVecEnv, you must close the env and start a
            new env before running again

            Args:
                return_supervised_data (bool): whether to test the model or
                    return supervised data, when false it would return only
                    the reward. obs and action would be null then.

            Return:
                obs(list[n_envs, n_episodes]): each element of the list
                    is scipy.sparse.csr_matrix of dimentions
                    (num_step * obs_size)
                act(list[n_envs, n_episodes]): each element is a list
                    of gt actions
                reward (np.array[n_envs, n_episodes]):

        """

        if return_supervised_data:
            obs = [[None] * self.n_episodes for x in range(self.n_envs)]
            act = [[None] * self.n_episodes for x in range(self.n_envs)]

        reward = np.zeros((self.n_envs, self.n_episodes))
        # current episode number for each env
        # also the number of episodes completed in each env
        episode_num = np.zeros(self.n_envs, dtype=np.int32)
        _obs_old = self.env.reset()

        while True:
            if all(episode_num >= self.n_episodes):
                break
            # get the action
            if return_supervised_data:
                _act = self.env.get_gt_action()
            else:
                _act = self.model.action_best(_obs_old)
            # execute the action
            _obs_new, _reward, _done, _ = self.env.step(_act.tolist())

            # update
            for j in range(self.n_envs):
                # Important to add the reward before done is called
                if episode_num[j] < self.n_episodes:
                    reward[j, episode_num[j]] += _reward[j]

                    if return_supervised_data:
                        if obs[j][episode_num[j]] is None:
                            obs[j][episode_num[j]] = _obs_old[j]
                            act[j][episode_num[j]] = [_act[j]]
                        else:
                            obs[j][episode_num[j]] = sci.sparse.vstack((
                                                    obs[j][episode_num[j]],
                                                    _obs_old[j]
                                                ))
                            act[j][episode_num[j]].append(_act[j])

                if _done[j]:
                    if episode_num[j] < self.n_episodes:
                        print("Env: {}, Episode: {}, Reward: {}".format(
                                j, episode_num[j], reward[j, episode_num[j]]))

                    episode_num[j] += 1

            _obs_old = _obs_new

        if return_supervised_data:
            print(obs[0][0].shape)
            return (obs, reward, act)
        else:
            return ([], reward, [])


def run_env(env, act_seq):
    """
    Assumption:
        actions don't lead any environment being done
        only supports env with one process
    Args:
        env: a single processor env
        act_seq: nparray(act_len)
    Return:
        reward:
    """

    act_len = len(act_seq)
    reward = 0
    for i in range(act_len):
        act = act_seq[i]
        obs, _reward, done, _ = env.step([int(act)])
        reward += _reward
        assert not done

    return reward


class RunnerBeamSearch(object):
    """ For running a beam search for a particular model """

    def __init__(self, model, beam_size, file_id, env_name, rot_before_mov,
                 worker_id_start, config):
        """
        Args:
            model: should have a function model.action_best_n(obs, beam_size)
                which takes in obs of size (n_envs, obs_size) and returns
                actions of size (n_envs, beam_size) and a score of size (n_envs,
                beam_size). The score is the log probability for each action.
                If there are less actions possible than score, the action
                should be -1.
            beam_size: hyperparameter for beam search
            file_id: the packing problem for which we have to do beam search
            env_name, rot_before_mov, worker_id_start, config: parameters
            for starting a new environment
        """

        self.model = model
        # contains all the arguments for the mul_pro_packing_env to create a new
        # env
        self.env_param = {
            'num_pro': 1,
            'env_name': env_name,
            'file_id_lst_lst': [[file_id]],
            'rot_before_mov': rot_before_mov,
            'shuffle': False,
            'get_gt': False,
            'worker_id_start': worker_id_start,
            'config': config
        }
        self.beam_size = beam_size

    def run(self):
        """
        Returns:
            the best reward upon doing the beam search
        """

        # we start with one
        envs = [mul_pro_packing_env(**self.env_param)]
        obs = [envs[0].reset()]
        obs = sparse.vstack(obs)

        # initialize
        act_seq = np.zeros([1, 0], dtype=np.int32)
        act_seq_reward = np.zeros([1])
        act_seq_score = np.zeros([1])

        # to keep track of best branch
        best_reward = 0

        while True:
            """
            assumming we have varibable number of environments in envs
            the envs[i] is run till act_seq[i] and has observation obs[i].
            the reward and score for envs[i] is act_seq_reward[i] and
            act_seq_score[i] repectively. none of the envs are done
            Input:
                envs list(num_envs):
                obs csr_matrix(num_envs, obs_size):
                act_seq nparray(num_envs, act_len):
                act_seq_reward nparray(num_envs):
                act_seq_score nparray(num_envs):
            """

            num_envs, act_len = np.shape(act_seq)
            act_next, act_next_score = self.model.action_best_n(
                obs, self.beam_size)

            # select the best set of next actions
            # transpose done so that when we select new branches, then while
            # taking max we select sub-branch coming from different branches
            act_next = np.transpose(act_next)
            act_next_score = np.transpose(act_next_score)
            act_next = np.reshape(act_next, [num_envs * self.beam_size])
            act_next_score = np.reshape(act_next_score,
                                        [num_envs * self.beam_size])
            _act_seq = np.tile(act_seq, [self.beam_size, 1])
            _act_seq_score = np.tile(act_seq_score, [self.beam_size])
            _act_seq_reward = np.tile(act_seq_reward, [self.beam_size])
            # tell which act_seq corresponds to which env
            act_seq_env_map = np.tile(np.arange(num_envs), [self.beam_size])

            # remove non-possible actions
            # happens when less the beam_size actions are possible
            poss_act_next = (act_next != -1)
            act_next = act_next[poss_act_next]
            act_next_score = act_next_score[poss_act_next]
            _act_seq = _act_seq[poss_act_next]
            _act_seq_score = _act_seq_score[poss_act_next]
            _act_seq_reward = _act_seq_reward[poss_act_next]
            act_seq_env_map = act_seq_env_map[poss_act_next]

            _act_seq_next_score = (_act_seq_score + act_next_score)
            num_new_envs = min(len(act_next), self.beam_size)
            sel_act_seq_next = np.argsort(
                _act_seq_next_score, axis=0)[::-1][0:num_new_envs]

            new_envs = []
            new_act_seq = np.zeros([num_new_envs, act_len + 1], dtype=np.int32)
            new_act_seq_score = np.zeros([num_new_envs])
            new_act_seq_reward = np.zeros([num_new_envs])
            avail_envs = np.ones(num_envs)
            for i in range(num_new_envs):
                id_act_seq = sel_act_seq_next[i]
                id_env = act_seq_env_map[id_act_seq]

                # if env available use it else rerun to create a new one
                if avail_envs[id_env] == 1:
                    avail_envs[id_env] = 0
                    _env = envs[id_env]
                else:
                    self.env_param['worker_id_start'] += 1
                    _env = mul_pro_packing_env(**self.env_param)
                    _env.reset()
                    run_env(_env, _act_seq[id_act_seq])

                new_envs.append(_env)
                new_act_seq[i] = np.concatenate(
                    [_act_seq[id_act_seq],
                     np.array([act_next[id_act_seq]])])
                new_act_seq_score[i] = _act_seq_next_score[id_act_seq]
                new_act_seq_reward[i] = _act_seq_reward[id_act_seq]

            # close and delete the envs not used
            del_env_indices = []
            for i in range(num_envs):
                if avail_envs[i] == 1:
                    envs[i].close()
                    del_env_indices.append(i)
            for i in sorted(del_env_indices, reverse=True):
                del envs[i]

            new_obs = []
            i = 0
            while True:
                if not i < len(new_envs):
                    break

                _env = new_envs[i]
                _obs, _reward, _done, _ = _env.step([new_act_seq[i, -1]])
                if _done:
                    # update best reward
                    if new_act_seq_reward[i] + _reward > best_reward:
                        best_reward = new_act_seq_reward[i] + _reward
                    # all clean up operations
                    _env.close()
                    del new_envs[i]
                    new_act_seq = np.delete(new_act_seq, i, axis=0)
                    new_act_seq_score = np.delete(new_act_seq_score, i, axis=0)
                    new_act_seq_reward = np.delete(
                        new_act_seq_reward, i, axis=0)

                else:
                    new_obs.append(_obs)
                    new_act_seq_reward[i] += _reward
                    i += 1

            # break and continue condition
            if len(new_envs) == 0:
                break
            else:
                envs = new_envs
                act_seq = new_act_seq
                act_seq_score = new_act_seq_score
                act_seq_reward = new_act_seq_reward
                obs = sparse.vstack(new_obs)

        return best_reward[0]


class RunnerBackTrackSearch(object):
    """ For running a beam search for a particular model """

    def __init__(self, model, budget, file_id, env_name, rot_before_mov,
                 worker_id_start, config):
        """
        Args:
            model: should have a function model.action_all_sorted which takes
            in obs (num_obs, obs_size) and gives back a list of len(num_obs).
            Each element in the list is the number of possible actions for that
            obs.
            budget: hyperparameter for back tracking
            file_id: the packing problem for which we have to do beam search
            env_name, rot_before_mov, worker_id_start, config: parameters
            for starting a new environment
        """

        self.model = model
        # contains all the arguments for the mul_pro_packing_env to create a new
        # env
        self.env_param = {
            'num_pro': 1,
            'env_name': env_name,
            'file_id_lst_lst': [[file_id]],
            'rot_before_mov': rot_before_mov,
            'shuffle': False,
            'get_gt': False,
            'worker_id_start': worker_id_start,
            'config': config
        }
        self.budget = budget

    def run(self):
        """
        Returns:
            the best reward upon doing the beam search
        """
        # we start with one env
        env = mul_pro_packing_env(**self.env_param)
        obs = env.reset()
        reward = 0
        act_pos_seq = [self.model.action_all_sorted(obs)[0]]
        # to keep track of best branch
        best_reward = 0
        no_more_paths = False
        for i in range(self.budget + 1):
            """
            Input:
                env: one environment
                act_pos_seq list(): list of nparray of possible actions. The
                    best of the actions (except the last element in the list)
                    have lead to the current environment. The first action for
                    the current environment is the first action of the last
                    element in act_pos_seq
                reward: reward collected till the point env is run
            """

            # forward movement till done
            while True:
                _act = act_pos_seq[-1][0]
                obs, _reward, _done, _ = env.step([int(_act)])
                reward += _reward[0]
                if _done[0]:
                    if reward > best_reward:
                        best_reward = reward
                    break
                else:
                    act_pos_seq.append(self.model.action_all_sorted(obs)[0])

            if i != self.budget:
                # backward movement till we find a valid position to start again
                while True:
                    if len(act_pos_seq) > 0:
                        last_pos_act = act_pos_seq[-1]
                        act_pos_seq = act_pos_seq[:-1]
                        assert len(last_pos_act) > 0
                        if len(last_pos_act) == 1:
                            continue
                        else:
                            act_pos_seq.append(last_pos_act[1:])
                            break
                    else:
                        no_more_paths = True
                        break

                if no_more_paths:
                    break
                print("after running")
                print(act_pos_seq)
                # getting the new environment
                act_seq = []
                # -1 to not consider the last pos_seq
                if len(act_pos_seq) > 1:
                    for pos_seq in act_pos_seq[:-1]:
                        act_seq.append(pos_seq[0])

                env.close()
                del env
                env = mul_pro_packing_env(**self.env_param)
                env.reset()
                print(act_seq)
                reward = run_env(env,
                                 np.array(act_seq))

            else:
                env.close()
                del env

        return best_reward
