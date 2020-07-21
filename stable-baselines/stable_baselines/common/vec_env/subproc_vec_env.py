from multiprocessing import Process, Pipe

import numpy as np
import scipy as sci

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                try:
                    observation, reward, done, info = env.step(data)
                except:
                    done = True
                    reward = 0
                    info = {}
                
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':             
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'get_gt_action':
                action = env.get_gt_action()
                remote.send(action)
            elif cmd == 'render':
                remote.send(env.render(**data))
            elif cmd == 'close':
                from packing.packing_env import PackingEnv
                if isinstance(env, PackingEnv):
                    env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    """
    
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for process in self.processes:
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if sci.sparse.issparse(obs[0]):
            return sci.sparse.vstack(obs), np.stack(rews), np.stack(dones), infos
        else:
            return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        if sci.sparse.issparse(obs[0]):
            return sci.sparse.vstack(obs)
        else:
            return np.stack(obs)
    
    def get_gt_action(self):
        for remote in self.remotes:
            remote.send(('get_gt_action', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', **kwargs):
        for pipe in self.remotes:
            pipe.send(('render', kwargs))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs
