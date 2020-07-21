import sys
import cProfile
import pickle as pkl
sys.path.insert(0, './')
sys.path.insert(0, './unity/')
sys.path.insert(0, './stable-baselines/')

from packing.packing_evalute import evaluate
from packing.packing_heuristic import *

list_pack_file_name = [["pack_ab/{}_tr_{}_ab".format(i, j) for i in range(0, 17)] for j in range(0, 800, 50)]

configs = [
    {
        'pol1': sha_lar,
        'pol2': mov_best,
        'pol3': rot_best_pos,
        'rot_before_mov': False
    },
    {
        'pol1': sha_lar,
        'pol2': rot_best,
        'pol3': mov_best,
        'rot_before_mov': True
    }
]

for config in configs:
    for pack_file_names in list_pack_file_name:
        model = HeuristicModel(config['pol1'], config['pol2'], config['pol3'])
        rewards = evaluate(
            pack_file_name=pack_file_names,
            model=model,
            n_envs=4,
            env_name='unity/envs/agent_19_11',
            rot_before_mov=config['rot_before_mov'],
            batch_size=20,
            save_sup_data=False,
            worker_id_start=400)
        iter_number = pack_file_names[0].split("_")[-2]
        filehandler = open(
            "results_3/ablations/ab_{}_{}_{}_{}".format(iter_number, config['pol1'].__name__,
                                    config['pol2'].__name__, config['pol3'].__name__), "wb")
        pkl.dump(rewards, filehandler)
        filehandler.close()
