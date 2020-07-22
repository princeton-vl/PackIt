import json
import os
import pickle
import scipy as sci
from tqdm import tqdm
from packing.packing_runner import RunnerEpisode, RunnerBeamSearch,\
        RunnerBackTrackSearch
from packing.packing_env import mul_pro_packing_env


def get_file_id_lst(env_name, pack_file_name):
    """ Returns the file_id_lst for the pack_file_name

    Args:
        env_name:
        pack_file_name (string or list(string)): path to the file(s)
            containing the pack info

    """

    if not isinstance(pack_file_name, list):
        pack_file_name = [pack_file_name]

    file_id_lst = []
    n_packs = 0
    for _pack_file_name in pack_file_name:
        _pack_file_path = \
            "{}_Data/StreamingAssets/{}".format(
                                        env_name,
                                        _pack_file_name)

        if os.path.isfile(_pack_file_path):
            with open(_pack_file_path) as f:
                _data = json.load(f)
                if 'data' in _data:
                    n_packs += len(_data['data'])
                    file_id_lst.extend(
                        [
                            [_pack_file_name, i]
                            for i in range(len(_data['data']))
                        ]
                    )
                else:
                    print("No pack data in file " + _pack_file_name)
        else:
            print("No file named exists named " + _pack_file_name)

    return file_id_lst


def dump_sup_data(file_id_lst_lst, obs, act, env_name):
    """
        file_id_lst_lst: list of file_id_lst
        obs(list[n_envs][n_episodes]): each element of the list
            is scipy.sparse.csr_matrix of dimentions
            (num_step * obs_size)
        act(list[n_envs][n_episodes]): each element is a list
    """

    for j in range(len(obs)):
        for k in range(len(obs[j])):
            assert sci.sparse.issparse(obs[j][k])

            file_id = file_id_lst_lst[j][k]
            file_info = file_id[0].split('/')
            if(len(file_info)) == 2:
                dir_name = file_info[0]
                file_name = file_info[1]
            else:
                dir_name = ""
                file_name = file_info[0]

            sup_file_path = \
                "{}_Data/StreamingAssets/{}_sup_data/{}_{}".format(
                                                        env_name,
                                                        dir_name,
                                                        file_name,
                                                        file_id[1])
            _obs = obs[j][k]
            _act = act[j][k]
            filehandler = open(sup_file_path, "wb")
            pickle.dump((_obs, _act), filehandler)
            filehandler.close()


def evaluate(
    pack_file_name,
    model,
    n_envs,
    env_name='unity/envs/agent',
    rot_before_mov=True,
    batch_size=10,
    save_sup_data=False,
    evaluate_first_n=None,
    worker_id_start=0,
    env_config={
                'sha': None,
                'mov': None,
                'rot': None
    },
    gpu_mode=True,
    beam_search=False,
    beam_size=4,
    back_track_search=False,
    budget=4):
    """ To evaluate all the packs in file name with the model
        Can also be used to save supervised data for those files

        WARNING: Keep batch_size small while saving data

    Args:
        pack_file_name (string or list(string)): path to the file(s)
            containing the pack info
        model: either rl, sup, heuristic or None
            must have a function named model.action_best
            None when save_sup_data is True
        n_envs (int): the number of parallel environments to run for the
            evaluation
        env_name (string): relative path to the env (eg-"unity/envs/agent")
            from where the function is called
        rot_before_mov (bool): for making the correct env
        batch_size (int): number of expisodes to run for each env at one go
            use batch_size = 0 for the largest batch size possible
        save_sup_data (bool):
        env_config (dict):
        gpu_mode (bool): in gpu_mode, for the left_packs, we create an env with
            multiple identicle environmnets so that input splitting is not an
            issue
        beam_search (bool): whether to do beam search on the data. when this
            is called, n_envs should be 1, save_sup_data should be False, batch_size
            should be 1, back_track_search should be False and gpu mode does not
            matter. must have only one available gpu when calling this mode for
            efficiency
        beam_size (int): number of beams to search.
        back_track_search (bool): whether to do back track search. when this
            is called, n_envs should be 1, save_sup_data should be False, batch_size
            should be 1, beam_search should be False and gpu mode does not matter.
            must have only one available gpu when calling this mode for efficiency.
        budget (int): number of back tracks allowed

    """

    reward = []
    file_id_lst = get_file_id_lst(
        env_name=env_name,
        pack_file_name=pack_file_name)
    if evaluate_first_n is not None:
        file_id_lst = file_id_lst[0: evaluate_first_n]

    n_packs = len(file_id_lst)

    if n_packs == 0:
        return []

    else:
        if beam_search:
            assert back_track_search is False
            assert n_envs == 1
            assert save_sup_data is False
            assert batch_size == 1

            for file_id in file_id_lst:
                runner = RunnerBeamSearch(
                    model=model,
                    beam_size=beam_size,
                    file_id=file_id,
                    env_name=env_name,
                    rot_before_mov=rot_before_mov,
                    worker_id_start=worker_id_start,
                    config=env_config)
                _reward = runner.run()
                reward.append(_reward)

        elif back_track_search:
            assert beam_search is False
            assert n_envs == 1
            assert save_sup_data is False
            assert batch_size == 1

            for file_id in file_id_lst:
                runner = RunnerBackTrackSearch(
                    model=model,
                    budget=budget,
                    file_id=file_id,
                    env_name=env_name,
                    rot_before_mov=rot_before_mov,
                    worker_id_start=worker_id_start,
                    config=env_config)
                _reward = runner.run()
                reward.append(_reward)

        else:
            packs_per_env = n_packs // n_envs
            n_left_packs = n_packs % n_envs
            if batch_size == 0:
                batch_size = packs_per_env

            n_batches = packs_per_env // batch_size
            n_left_batches = packs_per_env % batch_size

            # + 1 for the left batches
            for i in tqdm(range(n_batches + 1)):
                file_id_lst_lst = []
                # create temporary files
                for j in range(n_envs):
                    start_index = (j * packs_per_env) + (i * batch_size)
                    if i == n_batches:
                        end_index = start_index + n_left_batches
                    else:
                        end_index = start_index + batch_size
                    file_id_lst_lst.append(file_id_lst[start_index: end_index])

                env = mul_pro_packing_env(
                        num_pro=n_envs,
                        env_name=env_name,
                        file_id_lst_lst=file_id_lst_lst,
                        rot_before_mov=rot_before_mov,
                        shuffle=False,
                        get_gt=save_sup_data,
                        worker_id_start=worker_id_start,
                        config=env_config)

                if i == n_batches:
                    if n_left_batches != 0:
                        runner = RunnerEpisode(
                            env=env,
                            model=model,
                            n_episodes=n_left_batches)

                else:
                    runner = RunnerEpisode(
                        env=env,
                        model=model,
                        n_episodes=batch_size)

                if ((i == n_batches and n_left_batches != 0)
                        or (i != n_batches)):

                    _obs, _reward, _act = runner.run(
                            return_supervised_data=save_sup_data)
                    _reward = _reward.reshape(-1).tolist()
                    reward.extend(_reward)

                    print(_act)
                    if save_sup_data:
                        dump_sup_data(
                            file_id_lst_lst=file_id_lst_lst,
                            obs=_obs,
                            act=_act,
                            env_name=env_name)

                env.close()

            if n_left_packs != 0:
                start_index = packs_per_env * n_envs
                end_index = start_index + n_left_packs

                if gpu_mode:
                    n_envs_left = n_envs

                env = mul_pro_packing_env(
                    num_pro=n_envs_left,
                    env_name=env_name,
                    file_id_lst_lst=[file_id_lst[start_index: end_index]] * n_envs_left,
                    rot_before_mov=rot_before_mov,
                    shuffle=False,
                    get_gt=save_sup_data,
                    worker_id_start=worker_id_start,
                    config=env_config)

                runner = RunnerEpisode(
                        env=env,
                        model=model,
                        n_episodes=n_left_packs)

                _obs, _reward, _act = runner.run(
                                return_supervised_data=save_sup_data)
                _reward = _reward[0]
                _reward = _reward.reshape(-1).tolist()
                reward.extend(_reward)

                if save_sup_data:
                    dump_sup_data(
                        file_id_lst_lst=file_id_lst_lst,
                        obs=_obs,
                        act=_act,
                        env_name=env_name)

                env.close()

        print(reward)
        return reward
