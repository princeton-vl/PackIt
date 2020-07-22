import sys
import argparse
import os
import pickle as pkl

sys.path.insert(0, './')
sys.path.insert(0, './unity/')
sys.path.insert(0, './stable-baselines/')

import packing.packing_heuristic as pk_he
from packing.packing_evalute import evaluate

parser = argparse.ArgumentParser()

parser.add_argument('--id_start', type=int, default=0)
parser.add_argument('--pol_1', type=str, default='sha_lar')
parser.add_argument('--pol_2', type=str, default='rot_best')
parser.add_argument('--pol_3', type=str, default='mov_best')
# 1 means yes and 0 means no
parser.add_argument('--rot_before_mov', type=int, default=1)
parser.add_argument('--eval_va_or_te', type=int, default=1)
# beam_search is only done for the shape policy
parser.add_argument('--beam_search', type=int, default=0)
parser.add_argument('--beam_size', type=int, default=2)
# back_track_search is only done for the shape policy
parser.add_argument('--back_track_search', type=int, default=0)
parser.add_argument('--budget', type=int, default=4)
# start and end file id for evaluation
# end id 100 for complete test and 130 for complete validation set
parser.add_argument('--eval_start_id', type=int, default=0)
parser.add_argument('--eval_end_id', type=int, default=100)
parser.add_argument('--result_folder', type=str, default='results')


dispatcher = {
    'sha_lar': pk_he.sha_lar,
    'sha_rand': pk_he.sha_rand,
    'sha_lar_n': pk_he.sha_lar_n,
    'sha_all_sorted': pk_he.sha_all_sorted,
    'mov_best': pk_he.mov_best,
    'mov_rand': pk_he.mov_rand,
    'rot_best_pos': pk_he.rot_best_pos,
    'rot_best': pk_he.rot_best,
    'rot_rand': pk_he.rot_rand,
    'None': None
}

flags, unparsed = parser.parse_known_args()

if bool(flags.eval_va_or_te):
    pack_file_names = ["pack_va/{}_va".format(i) for i in range(
        flags.eval_start_id, flags.eval_end_id)]
else:
    pack_file_names = ["pack_te/{}_te".format(i) for i in range(
        flags.eval_start_id, flags.eval_end_id)]

if bool(flags.beam_search) or bool(flags.back_track_search):
    n_envs = 1
    batch_size = 1
    # we do beam search only on the sha policy i.e. pol_1
    if bool(flags.rot_before_mov):
        model_env = pk_he.HeuristicModel(pol_1=None, pol_2=pk_he.rot_best,
                                         pol_3=pk_he.mov_best)
    else:
        model_env = pk_he.HeuristicModel(pol_1=None, pol_2=pk_he.mov_best,
                                         pol_3=pk_he.rot_best_pos)
    env_config = {
        'sha': None,
        'mov': model_env.action_best,
        'rot': model_env.action_best
    }
else:
    n_envs = 4
    batch_size = 20
    env_config = {
        'sha': None,
        'mov': None,
        'rot': None
    }

model = pk_he.HeuristicModel(
    pol_1=dispatcher[flags.pol_1],
    pol_2=dispatcher[flags.pol_2],
    pol_3=dispatcher[flags.pol_3])

rewards = evaluate(
    pack_file_name=pack_file_names,
    model=model,
    n_envs=n_envs,
    env_name='unity/envs/packit',
    rot_before_mov=bool(flags.rot_before_mov),
    batch_size=batch_size,
    save_sup_data=False,
    worker_id_start=flags.id_start,
    env_config=env_config,
    beam_search=bool(flags.beam_search),
    beam_size=flags.beam_size,
    back_track_search=bool(flags.back_track_search),
    budget=flags.budget)

if not os.path.isdir(flags.result_folder):
    os.mkdir(flags.result_folder)

filehandler = open(
    "{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        flags.result_folder,
        "va" if bool(flags.eval_va_or_te) else "te",
        flags.pol_1,
        flags.pol_2,
        flags.pol_3,
        flags.rot_before_mov,
        flags.beam_search,
        flags.beam_size,
        flags.back_track_search,
        flags.budget,
        flags.eval_start_id,
        flags.eval_end_id),
    "wb")
pkl.dump(rewards, filehandler)
filehandler.close()
