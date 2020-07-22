import os
import sys
import argparse

import tensorflow as tf
import numpy as np
import pickle as pkl

sys.path.insert(0, './')
sys.path.insert(0, './unity/')
sys.path.insert(0, './stable-baselines/')

from packing.packing_policy import PackingPolicy, sha_pol
from packing.packing_env import PackingEnv, mul_pro_packing_env
from packing.packing_evalute import get_file_id_lst
from packing.packing_heuristic import *
from stable_baselines.ppo2 import PPO2

parser = argparse.ArgumentParser()

# only env
parser.add_argument('--id_start', type=int, default=0)
parser.add_argument('--num_tr_pack', type=int, default=200)
parser.add_argument('--num_pro', type=int, default=8, help="num of processors,\
                    matters only when learn_or_evaluate is 1.")

# only policy
parser.add_argument('--learn_sha_pol', type=int, default=1)
parser.add_argument('--learn_rot_pol', type=int, default=0)
parser.add_argument('--add_sum_fea', type=int, default=1)

# both env and policy
parser.add_argument('--rot_before_mov', type=int, default=1)
parser.add_argument('--rot_before_mov_env', type=int, default=-1)

# PPO
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--lam', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--vf_coef', type=float, default=0.5)
parser.add_argument('--ent_coef', type=float, default=0.1)
parser.add_argument('--zero_mean_advs', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=64)
parser.add_argument('--noptepochs', type=int, default=4)

# whether to learn or evaluate
# contions options for evaluation
# 1 means learn, 0 means evaluate
parser.add_argument('--learn_or_evaluate', type=int, default=1, help="1 for\
                    learn and 0 for evaluate.")
# 1 means validation 0 means test
# for evaluating test, the files should be in the folder named final inside log
parser.add_argument('--eval_va_or_te', type=int, default=1, help="1 for\
                    evaluting on the validation set and 0 for evaluating on\
                    the test set")
parser.add_argument('--model_name', type=str, default='PPO2_1/model_va',
                    help="matters only when learn_or_evaluate is 0")
# 1 means yes and 0 means no
parser.add_argument('--beam_search', type=int, default=0)
parser.add_argument('--beam_size', type=int, default=2)
# 1 means yes and 0 means no
parser.add_argument('--back_track_search', type=int, default=0)
parser.add_argument('--budget', type=int, default=4)
# start and end file id for evaluation
# end id 100 for complete test and 130 for complete validation set
parser.add_argument('--eval_start_id', type=int, default=0)
parser.add_argument('--eval_end_id', type=int, default=100)
parser.add_argument('--result_folder', type=str, default='results')


flags, unparsed = parser.parse_known_args()
assert bool(flags.learn_sha_pol)
assert not bool(flags.learn_rot_pol), "Not supported."
if flags.rot_before_mov_env == -1:
    flags.rot_before_mov_env = flags.rot_before_mov
else:
    # flags.rot_before_mov_env can make the structure between policy and env different
    # only be used for test (table 4, row 4)
    assert not bool(flags.learn_or_evaluate)

tensorboard_log = 'log/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    str(flags.num_tr_pack),
                    str(flags.gamma),
                    str(flags.lam),
                    str(flags.lr),
                    str(flags.zero_mean_advs),
                    str(flags.vf_coef),
                    str(flags.ent_coef),
                    str(flags.num_pro),
                    str(flags.noptepochs),
                    str(flags.learn_sha_pol),
                    str(flags.learn_rot_pol),
                    str(flags.rot_before_mov),
                    str(flags.add_sum_fea))
print(tensorboard_log)

if bool(flags.rot_before_mov):
    model_oracle = HeuristicModel(
        sha_lar,
        rot_best,
        mov_best)
else:
    model_oracle = HeuristicModel(
        sha_lar,
        mov_best,
        rot_best_pos)

env_name = 'unity/envs/packit'

pack_file_names = ["pack_tr/" + str(i) + "_tr" for i in range(0, flags.num_tr_pack)]
file_id_lst = get_file_id_lst(env_name, pack_file_names)
def make_env():
    return mul_pro_packing_env(
        num_pro=flags.num_pro,
        env_name=env_name,
        file_id_lst_lst=[file_id_lst] * flags.num_pro,
        rot_before_mov=bool(flags.rot_before_mov),
        shuffle=True,
        get_gt=False,
        worker_id_start=flags.id_start,
        config={
            'sha': None if bool(flags.learn_sha_pol) else model_oracle.action_best,
            'mov': model_oracle.action_best,
            'rot': model_oracle.action_best,
    })

policy_config = {
    'rot_before_mov':bool(flags.rot_before_mov),
    'add_bn':False,
    'add_sum_fea':bool(flags.add_sum_fea),
    'policy_weights':[1.0, 1.0, 1.0],
    'fixed_fea_config':{
        'box_fea_dim':10,
        'cho_sha_coarse_fea_dim':8,
        'cho_sha_fine_fea_dim':8
     },
     'comp_pol_config':{
        'sha_pol': sha_pol if bool(flags.learn_sha_pol) else None,
        'mov_pol': None,
        'rot_pol': None
     }
}

env = make_env()
model = PPO2(
    PackingPolicy,
    env,
    n_steps=flags.num_steps,
    verbose=1,
    tensorboard_log=tensorboard_log,
    nminibatches=int((flags.num_steps * flags.num_pro) / 64),
    noptepochs=flags.noptepochs,
    make_env=make_env,
    gamma=flags.gamma,
    lam=flags.lam,
    vf_coef=flags.vf_coef,
    ent_coef=flags.ent_coef,
    zero_mean_advs=bool(flags.zero_mean_advs),
    packing_id_start=flags.id_start,
    learning_rate=flags.lr,
    policy_config=policy_config,
    restore_exp=not(bool(flags.learn_or_evaluate)),
    restore_path="./{}/{}".format(tensorboard_log, flags.model_name))


if bool(flags.learn_or_evaluate):
    model.learn(flags.num_steps * flags.num_pro * 400)
else:
    if bool(flags.eval_va_or_te):
        pack_file_name_evaluate = ["pack_va/" + str(i) + "_va" for i in range(flags.eval_start_id, flags.eval_end_id)]
    else:
        pack_file_name_evaluate = ["pack_te/" + str(i) + "_te" for i in range(flags.eval_start_id, flags.eval_end_id)]
    _, _, _, rewards = model.evaluate(
        pack_file_name_evaluate,
        evaluate_first_n=None,
        beam_search=bool(flags.beam_search),
        beam_size=flags.beam_size,
        back_track_search=bool(flags.back_track_search),
        budget=flags.budget,
        rot_before_mov_env=bool(flags.rot_before_mov_env))
    if not os.path.isdir(flags.result_folder):
        os.mkdir(flags.result_folder)

    filehandler = open(
        "{}/{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(
            flags.result_folder,
            "va" if bool(flags.eval_va_or_te) else "te",
            tensorboard_log.replace("/", "_"),
            flags.model_name.replace("/", "_"),
            flags.beam_search,
            flags.beam_size,
            flags.back_track_search,
            flags.budget,
            flags.eval_start_id,
            flags.eval_end_id,
            "" if (flags.rot_before_mov_env == -1) else ("_" + str(flags.rot_before_mov_env))),
        "wb")

    pkl.dump(rewards, filehandler)
    filehandler.close()
