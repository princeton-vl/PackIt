import pickle as pkl
import os

def success_at(efficiency, threshold):
    success = [1 if x>threshold else 0 for x in efficiency]
    return (sum(success) / len(success))

# he: heuristic, le: learning, tab: table, tr: train, ev: eval
he_tab_1_row_1 = ["results/te_sha_rand_rot_best_mov_best_1_0_2_0_4_0_100"]
he_tab_1_row_2 = ["results/te_sha_lar_rot_rand_mov_best_1_0_2_0_4_0_100"]
he_tab_1_row_3 = ["results/te_sha_lar_rot_best_mov_rand_1_0_2_0_4_0_100"]
he_tab_1_row_4 = ["results/te_sha_lar_rot_best_mov_best_1_0_2_0_4_0_100"]
he_tab_2_row_2 = ["results/he_bts/te_sha_all_sorted_None_None_1_0_2_1_2_{}_{}".format(i, i+10) for i in range(0, 100, 10)]
he_tab_2_row_3 = ["results/he_bts/te_sha_all_sorted_None_None_1_0_2_1_4_{}_{}".format(i, i+10) for i in range(0, 100, 10)]
he_tab_2_row_4 = ["results/he_bts/te_sha_all_sorted_None_None_1_0_2_1_8_{}_{}".format(i, i+10) for i in range(0, 100, 10)]
he_tab_3_row_2 = ["results/he_bes/te_sha_lar_n_None_None_1_1_2_0_4_{}_{}".format(i, i+10) for i in range(0, 100, 10)]
he_tab_3_row_3 = ["results/he_bes/te_sha_lar_n_None_None_1_1_4_0_4_{}_{}".format(i, i+10) for i in range(0, 100, 10)]
he_tab_4_row_2 = ["results/te_sha_lar_mov_best_rot_best_pos_0_0_2_0_4_0_100"]

le_tab_1_row_5 = ["results/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_0_2_0_4_0_100_1"]
le_tab_2_row_6 = ["results/le_bts/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_0_2_1_2_{}_{}_1".format(i, i+10) for i in range(0, 100, 10)]
le_tab_2_row_7 = ["results/le_bts/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_0_2_1_4_{}_{}_1".format(i, i+10) for i in range(0, 100, 10)]
le_tab_2_row_8 = ["results/le_bts/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_0_2_1_8_{}_{}_1".format(i, i+10) for i in range(0, 100, 10)]
le_tab_3_row_5 = ["results/le_bes/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_1_2_0_4_{}_{}_1".format(i, i+10) for i in range(0, 100, 10)]
le_tab_3_row_6 = ["results/le_bes/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_1_4_0_4_{}_{}_1".format(i, i+10) for i in range(0, 100, 10)]
le_tab_4_row_5 = ["results/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_0_1_PPO2_1_model_va_0_2_0_4_0_100_0"]
le_tab_4_row_4 = ["results/te_log_200_1.0_0.95_1e-05_0_0.5_0.1_8_4_1_0_1_1_PPO2_1_model_va_0_2_0_4_0_100_0"]

# SELECT THE RESULTS WE WANT TO SEE
file_lst = he_tab_4_row_2
_file_lst = []
reward = []
for file_name in file_lst:
    if os.path.exists(file_name):
        _file_lst.append(file_name)
        filehandler = open(file_name, "rb")
        reward.extend(pkl.load(filehandler))
        filehandler.close()

efficiency = [x+1 if x<0 else x for x in reward]
average_efficiency = sum(efficiency) / len(efficiency)
print("Average Reward: {}".format(average_efficiency))
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.999]

for threshold in thresholds:
    print("Sucess@{}: {}".format(threshold, 100 * success_at(efficiency, threshold)))
print(len(reward))
