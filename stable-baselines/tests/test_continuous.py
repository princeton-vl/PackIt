import subprocess
import os

import pytest

from stable_baselines.a2c import A2C
# TODO: add support for continuous actions
# from stable_baselines.acer import ACER
# from stable_baselines.acktr import ACKTR
from stable_baselines.ddpg import DDPG
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnvBox
from tests.test_common import _assert_eq


N_TRIALS = 1000
NUM_TIMESTEPS = 10000

MODEL_LIST = [
    A2C,
    # ACER,
    # ACKTR,
    DDPG,
    PPO1,
    PPO2,
    TRPO
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_manipulation(model_class):
    """
    Test if the algorithm can be loaded and saved without any issues, the environment switching
    works and that the action prediction works

    :param model_class: (BaseRLModel) A model
    """
    try:
        env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

        # create and train
        model = model_class(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=NUM_TIMESTEPS, seed=0)

        # predict and measure the acc reward
        acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            acc_reward += reward
        acc_reward = sum(acc_reward) / N_TRIALS

        # saving
        model.save("./test_model")

        del model, env

        # loading
        model = model_class.load("./test_model")

        # changing environment (note: this can be done at loading)
        env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])
        model.set_env(env)

        # predict the same output before saving
        loaded_acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            loaded_acc_reward += reward
        loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        # assert <10% diff
        assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.1, \
            "Error: the prediction seems to have changed between loading and saving"

        # learn post loading
        model.learn(total_timesteps=100, seed=0)

        # validate no reset post learning
        loaded_acc_reward = 0
        set_global_seeds(0)
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)
            loaded_acc_reward += reward
        loaded_acc_reward = sum(loaded_acc_reward) / N_TRIALS
        # assert <10% diff
        assert abs(acc_reward - loaded_acc_reward) / max(acc_reward, loaded_acc_reward) < 0.1, \
            "Error: the prediction seems to have changed between pre learning and post learning"

        # predict new values
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)

        # Free memory
        del model, env

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


def test_ddpg():
    args = ['--env-id', 'Pendulum-v0', '--num-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.ddpg.main'] + args)
    _assert_eq(return_code, 0)
