import os
import sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, './unity/')
sys.path.insert(0, './stable-baselines/')
from packing.packing_policy import PackingPolicy, sha_pol
from packing.packing_env import PackingEnv, mul_pro_packing_env
from packing.packing_evalute import evaluate, get_file_id_lst
from packing.packing_heuristic import HeuristicModel, sha_lar, mov_best, rot_best, rot_best_pos
from packing.packing_runner import RunnerBackTrackSearch

# create the tf model and loads the weights
rot_before_mov = True
tf_model_path = "log/200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1/PPO2_1/model_va"
policy_config = {
    'rot_before_mov': rot_before_mov,
    'add_bn':False,
    'add_sum_fea':True,
    'policy_weights':[1.0, 1.0, 1.0],
    'fixed_fea_config':{
        'box_fea_dim':10,
        'cho_sha_coarse_fea_dim':8,
        'cho_sha_fine_fea_dim':8
     },
     'comp_pol_config':{
        'sha_pol':sha_pol,
        'mov_pol':None,
        'rot_pol':None
     }
}

tf_config = tf.ConfigProto(
        allow_soft_placement=True)
sess = tf.Session(config=tf_config)
model = PackingPolicy(
    sess=sess,
    reuse=False,
    **policy_config
)
tf.global_variables_initializer().run(session=sess)
tf.train.Saver().restore(sess, tf_model_path)

if rot_before_mov:
    model_oracle = HeuristicModel(
        sha_lar,
        rot_best,
        mov_best)
else:
    model_oracle = HeuristicModel(
        sha_lar,
        mov_best,
        rot_best_pos)

n_envs = 1
pack_file_name = [ "pack_va/" + str(i) + "_va" for i in range(0, 130)]
env_name = 'unity/envs/packit'
file_id_lst = get_file_id_lst(
    env_name=env_name,
    pack_file_name=pack_file_name)

# runs the groundtruth actions for the particular file_id and saves the shapes
def save_actions_gt(id_to_check):
    file_id = file_id_lst[id_to_check]
    print(file_id)
    env_param = {
        'num_pro': 1,
        'env_name': env_name,
        'file_id_lst_lst': [[file_id]],
        'rot_before_mov': rot_before_mov,
        'shuffle': False,
        'get_gt': True,
        'worker_id_start': 500,
        'config': {
            'sha': None,
            'mov': None,
            'rot': None
        },
        'save_actions': True,
        'save_action_path': 'data/visualize/{}_{}_gt'
    }
    env = mul_pro_packing_env(**env_param)
    obs = env.reset()
    reward = 0

    while True:
        act = env.get_gt_action()
        obs, _reward, done, _ = env.step(act.tolist())
        reward += _reward
        if done[0]:
            print(reward)
            del env
            break

def save_actions_le(id_to_check):
    file_id = file_id_lst[id_to_check]
    print(file_id)
    env_param = {
        'num_pro': 1,
        'env_name': env_name,
        'file_id_lst_lst': [[file_id]],
        'rot_before_mov': rot_before_mov,
        'shuffle': False,
        'get_gt': False,
        'worker_id_start': 500,
        'config': {
            'sha': None,
            'mov': model_oracle.action_best,
            'rot': model_oracle.action_best
        },
        'save_actions': True,
        'save_action_path': 'data/visualize/{}_{}_le'
    }
    env = mul_pro_packing_env(**env_param)
    obs = env.reset()
    reward = 0

    while True:
        act = model.action_best(obs)
        obs, _reward, done, _ = env.step(act.tolist())
        reward += _reward
        if done[0]:
            print(reward)
            del env
            break

def save_actions_he(id_to_check):
    file_id = file_id_lst[id_to_check]
    print(file_id)
    env_param = {
        'num_pro': 1,
        'env_name': env_name,
        'file_id_lst_lst': [[file_id]],
        'rot_before_mov': rot_before_mov,
        'shuffle': False,
        'get_gt': False,
        'worker_id_start': 500,
        'config': {
            'sha': None,
            'mov': None,
            'rot': None
        },
        'save_actions': True,
        'save_action_path': 'data/visualize/{}_{}_he'
    }
    env = mul_pro_packing_env(**env_param)
    obs = env.reset()
    reward = 0

    while True:
        act = model_oracle.action_best(obs)
        obs, _reward, done, _ = env.step(act.tolist())
        reward += _reward
        if done[0]:
            print(reward)
            del env
            break

for i in range(0, 100, 1):
    save_actions_gt(i)
    save_actions_he(i)
    save_actions_le(i)
