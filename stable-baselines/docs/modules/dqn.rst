.. _dqn:

.. automodule:: stable_baselines.deepq


DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_
and its extensions (Double-DQN, Dueling-DQN, Prioritized Experience Replay).

.. warning::

  The DQN model does not support ``stable_baselines.common.policies``,
  as a result it must use its own policy models (see :ref:`deepq_policies`).

Notes
-----

- Original paper: https://arxiv.org/abs/1312.5602


Can I use?
----------

-  Reccurent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌       ✔️
MultiDiscrete  ❌       ✔️
MultiBinary    ❌       ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
  from stable_baselines import DeepQ

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])

  model = DeepQ(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("deepq_cartpole")

  del model # remove to demonstrate saving and loading

  DeepQ.load("deepq_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


With Atari:

.. code-block:: python

  from stable_baselines.common.atari_wrappers import make_atari
  from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
  from stable_baselines import DeepQ

  env = make_atari('BreakoutNoFrameskip-v4')

  model = DeepQ(CnnPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("deepq_breakout")

  del model # remove to demonstrate saving and loading

  DeepQ.load("deepq_breakout")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: DeepQ
  :members:
  :inherited-members:

.. _deepq_policies:

DQN Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:


.. autoclass:: LnMlpPolicy
  :members:
  :inherited-members:


.. autoclass:: CnnPolicy
  :members:
  :inherited-members:


.. autoclass:: LnCnnPolicy
  :members:
  :inherited-members:


Custom Policy Network
---------------------

Similarly to the example given in the `examples <../guide/custom_policy.html>`_ page.
You can easily define a custom architecture for the policy network:

.. code-block:: python

  import gym

  from stable_baselines.deepq.policies import FeedForwardPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import DeepQ

  # Custom MLP policy of three layers of size 128 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             layers=[128, 128, 128],
                                             layer_norm=False,
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('LunarLander-v2')
  env = DummyVecEnv([lambda: env])

  model = DeepQ(CustomPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
