import itertools

# he: heuristic, le: learning, tab: table, tr: train, ev: eval
command = {
    # Figure 4
    'fig_4': 'python ablation.py',

    # evaluate the heuristic models
    'he_tab_1_row_1': 'python main_heuristic.py --eval_va_or_te 0'\
                      ' --pol_1 sha_rand',
    'he_tab_1_row_2': 'python main_heuristic.py --eval_va_or_te 0'\
                      ' --pol_2 rot_rand',
    'he_tab_1_row_3': 'python main_heuristic.py --eval_va_or_te 0'\
                      ' --pol_3 mov_rand',
    'he_tab_1_row_4': 'python main_heuristic.py --eval_va_or_te 0',
    'he_tab_4_row_2': 'python main_heuristic.py --eval_va_or_te 0'\
                      ' --rot_before_mov 0 --pol_2 mov_best'\
                      ' --pol_3 rot_best_pos',

    # train the learning based model
    'tr_le_tab_1_row_5': 'python main.py --id_start 10 --learn_or_evaluate 1',

    # evaluate the learning based model
    'ev_le_tab_1_row_5': 'python main.py --id_start 0 --learn_or_evaluate 0'\
                         ' --eval_va_or_te 0',

    # train the learning based model on PackIt Easy
    'tr_le_tab_4_row_5': 'python main.py --id_start 20 --learn_or_evaluate 1'\
                         ' --rot_before_mov 0',

    # evaluate the learning based model trained on PackIt
    # and tested on PackIt-Easy
    'ev_le_tab_4_row_4': 'python main.py --id_start 0 --learn_or_evaluate 0'\
                         ' --eval_va_or_te 0 --rot_before_mov_env 0',

    # evaluate the learning based model trained on PackIt-Easy and
    # tested on PackIt-Easy
    'ev_le_tab_4_row_5': 'python main.py --id_start 0 --learn_or_evaluate 0'\
                         ' --eval_va_or_te 0 --rot_before_mov 0',

    # heuristic with backtracks
    'he_tab_2_row_2_4': [
        'python main_heuristic.py --eval_va_or_te 0 --id_start {}'\
        ' --result_folder results/he_bts'\
        ' --pol_1 sha_all_sorted --pol_2 None --pol_3 None'\
        ' --back_track_search 1 --budget {}'\
        ' --eval_start_id {} --eval_end_id {}'.format(
            (10 * j) + (1000 * _i), i, j, j+10)\
        for (_i, i), j in itertools.product(enumerate([2, 4, 8]),
                                            list(range(0, 100, 10)))
    ],

    # heuristic with beam
    'he_tab_3_row_2_3': [
        'python main_heuristic.py --eval_va_or_te 0 --id_start {}'\
        ' --result_folder results/he_bes'\
        ' --pol_1 sha_lar_n --pol_2 None --pol_3 None'\
        ' --beam_search 1 --beam_size {}'\
        ' --eval_start_id {} --eval_end_id {}'.format(
            3000 + (10 * j) + (1000 * _i), i, j, j+10)\
        for (_i, i), j in itertools.product(enumerate([2, 4]),
                                            list(range(0, 100, 10)))
    ],

    # evaluate the learning based model with beam
    'le_tab_2_row_6_8': [
        'python main.py --eval_va_or_te 0 --id_start {}'\
        ' --result_folder results/le_bts --learn_or_evaluate 0'\
        ' --back_track_search 1 --budget {}'\
        ' --eval_start_id {} --eval_end_id {}'.format(
            (10 * j) + (1000 * _i), i, j, j+10)\
        for (_i, i), j in itertools.product(enumerate([2, 4, 8]),
                                            list(range(0, 100, 10)))
    ],

    # evaluate the learning based model with beam
    'le_tab_3_row_5_6': [
        'python main.py --eval_va_or_te 0 --id_start {}'\
        ' --result_folder results/le_bes --learn_or_evaluate 0'\
        ' --beam_search 1 --beam_size {}'\
        ' --eval_start_id {} --eval_end_id {}'.format(
            3000 + (10 * j) + (1000 * _i), i, j, j+10)\
        for (_i, i), j in itertools.product(enumerate([2, 4]),
                                            list(range(0, 100, 10)))
    ],
}

