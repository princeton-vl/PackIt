import numpy as np
import scipy as sci
import gym
import gc
from gym import spaces
from unity import UnityEnvironment
from stable_baselines.common.vec_env import SubprocVecEnv
import json


class PackingEnv(gym.Env):
    """Unity Environment for packing that follows gym interface"""
    # constants for the class
    metadata = {'render.modes': ['human']}
    VOX_RES = 100
    MOV_RES = 25
    MAX_NUM_SHA = 40
    NUM_VOX = VOX_RES ** 3
    NUM_MOV = MOV_RES ** 3
    NUM_ROT = 24
    obs_size = (1                  # step_typ
        + 1                        # step_num
        + NUM_VOX                  # box_rep
        + NUM_VOX                  # cho_sha_rep
        + (NUM_VOX * MAX_NUM_SHA)  # shape_rep
        + NUM_MOV                  # pos_loc
        + NUM_ROT)                 # pos_rot

    def __init__(
            self,
            env_name,
            file_id_lst=[["test", 1]],
            rot_before_mov=True,
            worker_id=0,
            shuffle=False,
            get_gt=False,
            output_sparse=True,
            seed_start=0,
            config={
                'sha': None,
                'mov': None,
                'rot': None
            },
            save_actions=False,
            save_action_path="visualize/{}_{}"):
        """ Initilization of gym compatible unity environment

        Args:
            env_name (string): relative path to the env, eg-"unity/envs/agent"
                when calling from the notebooks folder
            file_id_lst (list[[file_name, pack_id]]): list of file name and
                pack id that the environment goes through one by one
            rot_before_mov (bool):
            worker_id (int): the serial number of the worker, useful when
                multiple environments are launced
            get_gt (bool): whether to get the groundtruth packing actions
            seed_start (int): the starting seed for numpy, final seed of env
                is seed_start + worker_id. seed_start should be same for
                multiple workers
            config (dict): can be used to create custom environments.
                config[sha] specifies the policy for the sha step and so on.
                if policy is None then the env.step would stop there.
                if policy is a function then that fucntion will be called with
                the obs to get the action.
            save_actions (bool): whether the environment save the actions taken
                by the agent.
            save_action_path (str): the action is saved at
                save_action_path.format(pack_file, pack_id)
        """

        super(PackingEnv, self).__init__()
        self.unity_env = None

        self.env_name = env_name
        self.rot_before_mov = rot_before_mov
        self.worker_id = worker_id
        self.file_id_lst = file_id_lst
        self.shuffle = shuffle
        self.get_gt = get_gt
        self.output_sparse = output_sparse
        self.config = config
        self.save_actions = save_actions
        self.save_action_path = save_action_path

        assert ((config['sha'] is None) or (config['mov'] is None)
                    or (config['rot'] is None)), "Atleast one of the policy\
            should be none"

        np.random.seed(seed_start + worker_id)

        # setting up action and observation space
        self.action_space = spaces.Discrete(self.NUM_MOV)

        self.observation_space = spaces.Box(
                    low=0, high=1,
                    shape=(self.obs_size,))

        # variables to store observation
        # sha for shapes
        # pos for possible
        # mov for move
        # rot for rotation
        # rep for representation
        # chos for chosen
        self.cho_sha = None
        self.step_typ = None
        self.step_num = None
        self.box_rep = np.zeros((self.NUM_VOX))
        self.cho_sha_rep = sci.sparse.csr_matrix((1, self.NUM_VOX))
        self.sha_rep = sci.sparse.csr_matrix(
                        (self.MAX_NUM_SHA, self.NUM_VOX))
        self.pos_mov = np.zeros((self.NUM_MOV))
        if self.rot_before_mov:
            self.pos_rot = np.ones((self.NUM_ROT))
        else:
            self.pos_rot = np.zeros((self.NUM_ROT))

        self.file_id_index = 0
        self.num_packs = len(self.file_id_lst)
        self.file_id_indices = np.arange(self.num_packs)
        if self.save_actions:
            self.act_record = {}
            self.act_record['data'] = []

        if self.rot_before_mov:
            self.step_typ_int_str = {
                1: 'sha',
                2: 'rot',
                3: 'mov'
            }
        else:
            self.step_typ_int_str = {
                1: 'sha',
                2: 'mov',
                3: 'rot'
            }

    def _get_env_obs(self):
        """Returns the observation based on the state of the env"""

        if self.output_sparse:
            return sci.sparse.csr_matrix(
                sci.sparse.hstack(
                (
                    sci.sparse.csr_matrix(self.step_typ),
                    sci.sparse.csr_matrix(self.step_num),
                    sci.sparse.csr_matrix(self.box_rep),
                    self.cho_sha_rep,
                    self.sha_rep.reshape((1, -1)),
                    sci.sparse.csr_matrix(self.pos_mov),
                    sci.sparse.csr_matrix(self.pos_rot)
                ))
            )

        else:
            return np.concatenate(
                (
                    (self.step_typ,),
                    (self.step_num,),
                    self.box_rep,
                    self.cho_sha_rep.toarray().squeeze(),
                    np.reshape(
                        self.sha_rep.toarray(),
                        (self.MAX_NUM_SHA * self.NUM_VOX)),
                    self.pos_mov,
                    self.pos_rot
                )
            )

    def _step(self, action):
        """Returns the next step of the environment

        Args:
            action (int or array[1]): identifies either the chosen shape,
                chosen location or chosen rotation depending on
                the step_typ
        """

        assert (isinstance(action, np.int32) or isinstance(action, int)
                    or isinstance(action, np.ndarray))
        if isinstance(action, np.ndarray):
            assert action.shape == (1,) and action.dtype == np.int32
            action = action[0]

        # assert that env running correctly
        assert self.step_typ in [1, 2, 3]

        # decode the action based on the step type
        # simultaneously change the shape representation
        unity_action = np.zeros((1, 3))
        if self.step_typ == 1:
            unity_action[0, 0] = action
            # set the chosen shape represenation
            # and remove the shape from sha rep
            self.cho_sha = action
            self.cho_sha_rep = sci.sparse.csr_matrix.copy(
                self.sha_rep[self.cho_sha, :])
            self.sha_rep[self.cho_sha] = sci.sparse.csr_matrix((1,
                                                                self.NUM_VOX))
            # eliminating zeros for memory management
            self.sha_rep.eliminate_zeros()

        elif self.step_typ == 2:
            if self.rot_before_mov:
                unity_action[0, 0] = action
            else:
                unity_action[0, 0] = action // (self.MOV_RES ** 2)
                unity_action[0, 1] = (action % (self.MOV_RES ** 2)) // self.MOV_RES
                unity_action[0, 2] = (action % (self.MOV_RES ** 2)) % self.MOV_RES

        else:
            if self.rot_before_mov:
                unity_action[0, 0] = action // (self.MOV_RES ** 2)
                unity_action[0, 1] = (action % (self.MOV_RES ** 2)) // self.MOV_RES
                unity_action[0, 2] = (action % (self.MOV_RES ** 2)) % self.MOV_RES
            else:
                unity_action[0, 0] = action

            self.cho_sha_rep = sci.sparse.csr_matrix((1, self.NUM_VOX))

        if self.save_actions:
            self.act_record['data'].append({
                "step": int(self.step_typ+1),
                "stepNum": int(self.step_num),
                "action": {
                    "x": int(unity_action[0, 0]),
                    "y": int(unity_action[0, 1]),
                    "z": int(unity_action[0, 2])}
                })

        # execute action
        info = self.unity_env.step(unity_action)[self.default_brain]
        self.step_typ = self._get_unity_step_typ(info)
        self.step_num = self._get_unity_step_num(info)

        # upadate the observations
        assert self.step_typ in [1, 2, 3]
        if self.step_typ == 1:
            self.box_rep = self._get_unity_obs(info, self.NUM_VOX)

        elif self.step_typ == 2:
            if self.rot_before_mov:
                pass
            else:
                self.pos_mov = self._get_unity_obs(info, self.NUM_MOV)

        else:
            if self.rot_before_mov:
                self.pos_mov = self._get_unity_obs(info, self.NUM_MOV)
            else:
                self.pos_rot = self._get_unity_obs(info, self.NUM_ROT)

        # for better memory management
        _rewards = info.rewards[0]
        _done = info.local_done[0]
        del info

        if _done and self.save_actions:
            self.act_record['data'] = self.act_record['data'][0:-1]
            _file_id_index = self.file_id_indices[
                self.file_id_index % self.num_packs]
            pack_file_name = self.file_id_lst[_file_id_index][0].replace("/", "_")
            pack_id = self.file_id_lst[_file_id_index][1]
            path_save = self.save_action_path.format(pack_file_name, pack_id)
            print("saving action to: " + path_save)
            with open(path_save, 'w') as outfile:
                json.dump(self.act_record, outfile)

        # taking care of case when the reward is -1
        if _rewards == -1:
            _rewards = 0

        return self._get_env_obs(), _rewards, _done, {}

    def _reset(self):
        """ Reset the unity environment

            Must be called before the first step function

        """

        if self.file_id_index % self.num_packs == 0:
            if self.shuffle:
                np.random.shuffle(self.file_id_indices)

        _file_id_index = self.file_id_indices[
                self.file_id_index % self.num_packs]
        self.file_id_index += 1

        # setting up the unity environment
        if self.unity_env is not None:
            self.unity_env.close()

        # garbage collection before starting a new env
        gc.collect()
        for i in range(10):
            try:
                self.unity_env = UnityEnvironment(
                    file_name=self.env_name,
                    rot_before_mov=self.rot_before_mov,
                    no_graphics=True,
                    worker_id=self.worker_id,
                    pack_file_name=self.file_id_lst[_file_id_index][0],
                    pack_id=self.file_id_lst[_file_id_index][1],
                    get_gt=self.get_gt)
            except:
                if self.unity_env is not None:
                    self.unity_env.close()
                continue
            break

        self.default_brain = self.unity_env.brain_names[0]
        info = self.unity_env.reset(train_mode=True)[self.default_brain]
        num_sha = (int)(info.vector_observations[0][2])

        if self.get_gt:
            _gt_action = self._get_unity_obs(info, (num_sha * 3 * 5) + 1)
            # _gt_action also contains the num_sha so taking from index 1
            self.gt_action = np.reshape(_gt_action[1:], (num_sha, 3, 5))

        # all the initial step observations
        self.step_typ = 1
        self.step_num = 0
        self.cho_sha_rep = sci.sparse.csr_matrix((1, self.NUM_VOX))

        # loading precomputed voxel reps
        # similar computation done in unity
        file_info = self.file_id_lst[_file_id_index][0].split('/')
        assert len(file_info) <= 2
        if(len(file_info)) == 2:
            dir_name = file_info[0]
            file_name = file_info[1]
        else:
            dir_name = ""
            file_name = file_info[0]

        precompute_file_path = (self.env_name
                          + "_Data/StreamingAssets/"
                          + dir_name
                          + "_precompute_python/"
                          + file_name
                          + "_"
                          + str(self.file_id_lst[_file_id_index][1]))
        with open(precompute_file_path) as f:
            _ = json.load(f)
            _sha_rep_sparse = _['vox']

        # +1 as there is total_sha_vox
        # _sha_rep_sparse is of the form:
        #  [sha_num_1, vox_id_1, sha_num_1, vox_id_2, ....
        #   sha_num_n, vox_id_1, sha_num_n, vox_id_2....
        #   sha_num_n, vox_id_N-1, sha_num_n, vox_id_N]
        _sha_rep_sparse = np.transpose(
                np.reshape(_sha_rep_sparse, (-1, 2)))
        _sha_rep_sparse = _sha_rep_sparse.astype(np.int64)

        self.sha_rep = sci.sparse.csr_matrix(
                    (
                        np.ones((_sha_rep_sparse.shape[1])),
                        (_sha_rep_sparse[0, :], _sha_rep_sparse[1, :])
                    ),
                    (self.MAX_NUM_SHA, self.NUM_VOX)
        )
        del _sha_rep_sparse

        info = self.unity_env.step(np.zeros((1, 3)))[self.default_brain]
        self.box_rep = self._get_unity_obs(info, self.NUM_VOX)
        # for better memory management
        del info

        return self._get_env_obs()

    def reset(self):
        obs = self._reset()
        self.initial_reset_reward = 0

        while True:
            # step_typ_str is either 'sha', 'rot' or 'mov'
            step_typ_str = self.step_typ_int_str[self.step_typ]
            if self.config[step_typ_str] is None:
                break
            else:
                action = self.config[step_typ_str](obs)
                obs, reward, done, _ = self._step(action)

            self.initial_reset_reward += reward
            assert not done, "Issue as the env is done on the reset itself"

        return obs

    def step(self, action):

        if not self.initial_reset_reward == 0:
            step_reward = self.initial_reset_reward
            self.initial_reset_reward = 0
        else:
            step_reward = 0

        obs, reward, done, _ = self._step(action)
        step_reward += reward

        if done:
            pass
        else:
            while True:
                # step_typ_str is either 'sha', 'rot' or 'mov'
                step_typ_str = self.step_typ_int_str[self.step_typ]
                if self.config[step_typ_str] is None:
                    break
                else:
                    action = self.config[step_typ_str](obs)
                    obs, reward, done, _ = self._step(action)

                step_reward += reward
                if done:
                    break

        return obs, step_reward, done, {}

    def get_gt_action(self):
        """Returns the gt action for the current environment state"""

        assert self.get_gt

        _unity_gt_action = self.gt_action[self.step_num, self.step_typ-1, :]
        assert _unity_gt_action[0] == (self.step_typ + 1)
        assert _unity_gt_action[1] == self.step_num

        unity_gt_action = _unity_gt_action[2:]
        if self.step_typ == 1:
            return (int)(unity_gt_action[0])

        elif self.step_typ == 2:
            if self.rot_before_mov:
                return (int)(unity_gt_action[0])
            else:
                return (int)(unity_gt_action[2]
                         + (unity_gt_action[1] * self.MOV_RES)
                         + (unity_gt_action[0] * (self.MOV_RES ** 2)))

        else:
            if self.rot_before_mov:
                return (int)(unity_gt_action[2]
                         + (unity_gt_action[1] * self.MOV_RES)
                         + (unity_gt_action[0] * (self.MOV_RES ** 2)))
            else:
                return (int)(unity_gt_action[0])

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if self.unity_env is not None:
            self.unity_env.close()

    @staticmethod
    def _get_unity_obs(env_info, num_obs):
        # the first two indices of vector_observations[0] are step_typ and
        # step_num
        return np.copy((env_info.vector_observations[0])[2: 2 + num_obs])

    @staticmethod
    def _get_unity_step_typ(env_info):
        # in unity env, step_type in [2, 3, 4]
        # we are changing it to [1, 2, 3]
        return (int)((env_info.vector_observations[0])[0] - 1)

    @staticmethod
    def _get_unity_step_num(env_info):
        return (int)((env_info.vector_observations[0])[1])

    @classmethod
    def _decode_agent_obs(cls, agent_obs):
        """ Decodes the observation provided by Packing environment

            Args:
                agent_obs (np.array[observation_space]): observation by the
                    packing environment
            Returns:
                dict: containing various elements explained below
                    "step_typ" (NUM_OBS): either 1, 2 or 3
                    "step_num" (NUM_OBS)
                    "box_rep" (np.array[NUM_OBS, VOX_RES, VOX_RES, VOX_RES])
                    "sha_rep" (np.array[NUM_OBS, MAX_NUM_SHA,
                        VOX_RES, VOX_RES, VOX_RES])
                    "cho_sha_rep" (np.array[NUM_OBS, VOX_RES, VOX_RES, VOX_RES])
                    "pos_mov" (np.array[NUM_OBS, MOV_RES, MOV_RES, MOV_RES])
                    "pos_rot" (np.array[NUM_OBS, NUM_ROT])
                    "sha_mask" (np.array[NUM_OBS, MAX_NUM_SHA]): 1 for the shapes
                        still left to be chosen, 0 otherwise
        """

        if agent_obs.ndim == 1:
            agent_obs = np.expand_dims(agent_obs, axis=0)

        NUM_OBS = agent_obs.shape[0]

        box_rep_size = PackingEnv.NUM_VOX
        cho_sha_rep_size = PackingEnv.NUM_VOX
        sha_rep_size = PackingEnv.NUM_VOX * PackingEnv.MAX_NUM_SHA
        pos_mov_size = PackingEnv.NUM_MOV
        pos_rot_size = PackingEnv.NUM_ROT

        # first index is step_typ
        # second index is step_num
        box_rep_start = 2
        cho_sha_rep_start = box_rep_start + box_rep_size
        sha_rep_start = cho_sha_rep_start + cho_sha_rep_size
        pos_mov_start = sha_rep_start + sha_rep_size
        pos_rot_start = pos_mov_start + pos_mov_size

        _sha_rep = np.reshape(
            agent_obs[:, sha_rep_start:sha_rep_start + sha_rep_size],
            (NUM_OBS, PackingEnv.MAX_NUM_SHA, PackingEnv.VOX_RES,
             PackingEnv.VOX_RES, PackingEnv.VOX_RES))
        _sha_rep_max = np.max(_sha_rep, axis=(2, 3, 4))
        num_sha = np.count_nonzero(_sha_rep_max, axis=1)
        sha_mask = np.zeros((NUM_OBS, PackingEnv.MAX_NUM_SHA))
        sha_mask[_sha_rep_max > 0] = 1

        return {
            "step_typ":
            agent_obs[:, 0],
            "step_num":
            agent_obs[:, 1],
            "box_rep":
            np.reshape(agent_obs[:, box_rep_start:box_rep_start + box_rep_size],
                       (NUM_OBS, PackingEnv.VOX_RES, PackingEnv.VOX_RES,
                        PackingEnv.VOX_RES)),
            "cho_sha_rep":
            np.reshape(
                agent_obs[:, cho_sha_rep_start:cho_sha_rep_start +
                          cho_sha_rep_size],
                (NUM_OBS, PackingEnv.VOX_RES, PackingEnv.VOX_RES,
                 PackingEnv.VOX_RES)),
            "sha_rep":
            _sha_rep,
            "pos_mov":
            np.reshape(agent_obs[:, pos_mov_start:pos_mov_start + pos_mov_size],
                       (NUM_OBS, PackingEnv.MOV_RES, PackingEnv.MOV_RES,
                        PackingEnv.MOV_RES)),
            "pos_rot":
            agent_obs[:, pos_rot_start:pos_rot_start + pos_rot_size],
            "num_sha":
            num_sha,
            "sha_mask":
            sha_mask
        }


def make_env(
    env_name,
    file_id_lst,
    rot_before_mov,
    worker_id,
    seed_start,
    get_gt,
    shuffle,
    output_sparse,
    config,
    save_actions,
    save_action_path):
    """
    Utility function for multiprocessed env.
    Inspired from the stable-baselines example.

    Args:
        same as those defined in PackingEnv
    """
    def _init():
        env = PackingEnv(
            env_name=env_name,
            file_id_lst=file_id_lst,
            rot_before_mov=rot_before_mov,
            worker_id=worker_id,
            seed_start=seed_start,
            get_gt=get_gt,
            shuffle=shuffle,
            output_sparse=output_sparse,
            config=config,
            save_actions=save_actions,
            save_action_path=save_action_path)
        return env
    return _init


def mul_pro_packing_env(
    num_pro,
    env_name,
    file_id_lst_lst,
    rot_before_mov,
    shuffle,
    get_gt,
    worker_id_start=0,
    config={
                'sha': None,
                'mov': None,
                'rot': None
    },
    save_actions=False,
    save_action_path="visualize/{}_{}"):

    """

    Each env is launched with a different seed for numpy to make sure
    pack orders are different after shuffling.

    Args:
        num_pro (int):
        env_name (string): eg "unity/envs/agent_2"
        file_id_lst_lst: list of file_id_lst
        rot_before_mov (bool):
        shuffle (bool):
        get_gt (bool):
        config (dict):
        Output always sparse
    """

    # either empty lists or two values per element
    assert (
        len(file_id_lst_lst[0]) == 0
        or len(file_id_lst_lst[0][0]) == 2)
    seed_start = np.random.randint(np.iinfo(np.int16).max)

    env  = SubprocVecEnv([
            make_env(
                env_name=env_name,
                file_id_lst=file_id_lst_lst[i],
                rot_before_mov=rot_before_mov,
                worker_id=(worker_id_start + i),
                seed_start=seed_start,
                shuffle=shuffle,
                get_gt=get_gt,
                output_sparse=True,
                config=config,
                save_actions=save_actions,
                save_action_path=save_action_path)
            for i in range(num_pro)
           ])
    return env
