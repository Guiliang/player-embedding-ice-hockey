import csv
import datetime
import sys
import traceback

# from random import shuffle

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os
import tensorflow as tf
import numpy as np
from support.model_tools import ExperienceReplayBuffer, compute_acc, BalanceExperienceReplayBuffer
from config.varlea_config import VaRLEACongfig
from nn_structure.varlea import VaRLEA
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv, compute_game_win_vec, compute_game_score_diff_vec, \
    read_feature_within_events
from support.data_processing_tools import get_icehockey_game_data, transfer2seq, generate_selection_matrix, \
    safely_expand_reward, generate_diff_player_cluster_id, q_values_output_mask
from support.model_tools import get_model_and_log_name, compute_rnn_acc

# from support.plot_tools import plot_players_games

General_MemoryBuffer = ExperienceReplayBuffer(capacity_number=30000)
Prediction_MemoryBuffer = BalanceExperienceReplayBuffer(capacity_number=30000)
TRAINING_ITERATIONS = 0

def gathering_data_and_run(dir_game, config, player_id_cluster_dir,
                           data_store, source_data_dir,
                           model, sess, training_flag, game_number,
                           validate_cvrnn_flag=False,
                           validate_td_flag=False,
                           validate_diff_flag=False,
                           validate_variance_flag=False,
                           validate_predict_flag=False,
                           output_decoder_all=None,
                           target_data_all=None,
                           selection_matrix_all=None,
                           q_values_all=None,
                           output_label_all=None,
                           real_label_all=None,
                           pred_target_data_all=None,
                           pred_output_prob_all=None,
                           ):
    if validate_variance_flag:
        match_q_values_players_dict = {}
        for i in range(config.Learn.player_cluster_number):
            match_q_values_players_dict.update({i: []})
    else:
        match_q_values_players_dict = None

    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)
    reward_seq = transfer2seq(data=np.expand_dims(reward, axis=-1), trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)
    # win_one_hot = compute_game_win_vec(rewards=reward)
    score_diff = compute_game_score_diff_vec(rewards=reward)

    score_difference_game = read_feature_within_events(dir_game,
                                                       source_data_dir,
                                                       'scoreDifferential',
                                                       transfer_home_number=True,
                                                       data_store=data_store)

    state_reward_input = np.concatenate([reward_seq, state_input], axis=-1)  # concatenate the sequence of state and reward.

    if config.Arch.Predict.predict_target == 'ActionGoal':
        add_pred_flag = True
        actions_all = read_feature_within_events(directory=dir_game,
                                                 data_path=source_data_dir,
                                                 feature_name='name')
        next_goal_label = []
        data_length = state_trace_length.shape[0]
        new_reward = []
        new_reward_seq = []
        new_action_seq = []
        new_state_reward_input = []
        new_state_trace_length = []
        new_team_id_seq = []
        new_player_index_seq = []
        for action_index in range(0, data_length):
            action = actions_all[action_index]
            if 'shot' in action:
                if action_index + 1 == data_length:
                    continue
                new_reward.append(reward[action_index])
                new_reward_seq.append(reward_seq[action_index])
                new_action_seq.append(action_seq[action_index])
                new_state_reward_input.append(state_reward_input[action_index])
                new_state_trace_length.append(state_trace_length[action_index])
                new_team_id_seq.append(team_id_seq[action_index])
                new_player_index_seq.append(player_index_seq[action_index])
                if actions_all[action_index + 1] == 'goal':
                    # print(actions_all[action_index+1])
                    next_goal_label.append([1, 0])
                else:
                    # print(actions_all[action_index + 1])
                    next_goal_label.append([0, 1])
        pred_target = next_goal_label
    else:
        # raise ValueError()
        add_pred_flag = False

    if add_pred_flag:
        if training_flag:
            train_mask = np.asarray([[[1]] * config.Learn.max_seq_length] * len(new_state_reward_input))
        else:
            train_mask = np.asarray([[[0]] * config.Learn.max_seq_length] * len(new_state_reward_input))

        pred_input_data = np.concatenate([np.asarray(new_player_index_seq),
                                          np.asarray(new_team_id_seq),
                                          np.asarray(new_state_reward_input),
                                          np.asarray(new_action_seq),
                                          train_mask
                                          ], axis=2)

        pred_target_data = np.asarray(np.asarray(pred_target))
        pred_trace_lengths = new_state_trace_length
        pred_selection_matrix = generate_selection_matrix(new_state_trace_length, max_trace_length=config.Learn.max_seq_length)

        if training_flag:
            for i in range(len(new_state_reward_input)):
                cache_label = np.argmax(pred_target_data[i], axis=0)
                Prediction_MemoryBuffer.push([pred_input_data[i],
                                              pred_target_data[i],
                                              pred_trace_lengths[i],
                                              pred_selection_matrix[i]
                                              ], cache_label)

    # reward_count = sum(reward)
    # print ("reward number" + str(reward_count))
    if len(state_reward_input) != len(reward) or len(state_trace_length) != len(reward):
        raise Exception('state length does not equal to reward length')

    kl_loss_game = []
    ll_game = []

    train_len = len(state_reward_input)
    train_number = 0
    s_t0 = state_reward_input[train_number]
    train_number += 1
    while True:
        # try:
        batch_return, \
        train_number, \
        s_tl, \
        print_flag = get_together_training_batch(s_t0=s_t0,
                                                 state_input=state_reward_input,
                                                 reward=reward,
                                                 player_index=player_index_seq,
                                                 train_number=train_number,
                                                 train_len=train_len,
                                                 state_trace_length=state_trace_length,
                                                 action=action_seq,
                                                 team_id=team_id_seq,
                                                 win_info=score_diff,
                                                 score_info=score_difference_game,
                                                 config=config)

        # get the batch variables
        # s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
        #                      team_id_t1, 0, 0
        s_t0_batch = [d[0] for d in batch_return]
        s_t1_batch = [d[1] for d in batch_return]
        r_t_batch = [d[2] for d in batch_return]
        trace_t0_batch = [d[3] for d in batch_return]
        trace_t1_batch = [d[4] for d in batch_return]
        action_id_t0 = [d[5] for d in batch_return]
        action_id_t1 = [d[6] for d in batch_return]
        team_id_t0_batch = [d[7] for d in batch_return]
        team_id_t1_batch = [d[8] for d in batch_return]
        player_id_t0_batch = [d[9] for d in batch_return]
        player_id_t1_batch = [d[10] for d in batch_return]
        win_id_t_batch = [d[11] for d in batch_return]
        terminal_batch = [d[-2] for d in batch_return]
        cut_batch = [d[-1] for d in batch_return]
        score_diff_t_batch = [d[11] for d in batch_return]
        score_diff_base_t0_batch = [d[12] for d in batch_return]
        outcome_data = score_diff_t_batch
        score_diff_base_t0 = score_diff_base_t0_batch
        # if training_flag:
        train_mask = np.asarray([[[1]] * config.Learn.max_seq_length] * len(s_t0_batch))
        # else:
        #     train_mask = np.asarray([[[0]] * config.Learn.max_seq_length] * len(s_t0_batch))
        # # (player_id, state ,action flag)
        for i in range(0, len(terminal_batch)):
            terminal = terminal_batch[i]
            cut = cut_batch[i]

        input_data_t0 = np.concatenate([np.asarray(player_id_t0_batch),
                                        np.asarray(team_id_t0_batch),
                                        np.asarray(s_t0_batch),
                                        np.asarray(action_id_t0),
                                        train_mask], axis=2)
        target_data_t0 = np.asarray(np.asarray(player_id_t0_batch))
        trace_lengths_t0 = trace_t0_batch
        selection_matrix_t0 = generate_selection_matrix(trace_lengths_t0,
                                                        max_trace_length=config.Learn.max_seq_length)

        input_data_t1 = np.concatenate([np.asarray(player_id_t1_batch),
                                        np.asarray(team_id_t1_batch),
                                        np.asarray(s_t1_batch),
                                        np.asarray(action_id_t1),
                                        train_mask], axis=2)
        target_data_t1 = np.asarray(np.asarray(player_id_t1_batch))
        trace_lengths_t1 = trace_t1_batch
        selection_matrix_t1 = generate_selection_matrix(trace_t1_batch,
                                                        max_trace_length=config.Learn.max_seq_length)

        if training_flag:

            if config.Learn.apply_stochastic:
                for i in range(len(input_data_t0)):
                    General_MemoryBuffer.push(
                        [input_data_t0[i], target_data_t0[i], trace_lengths_t0[i], selection_matrix_t0[i],
                         input_data_t1[i], target_data_t1[i], trace_lengths_t1[i], selection_matrix_t1[i],
                         r_t_batch[i], win_id_t_batch[i], terminal_batch[i], cut_batch[i]
                         ])
                sampled_data = General_MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                sample_input_data_t0 = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                sample_target_data_t0 = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                sample_trace_lengths_t0 = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                sample_selection_matrix_t0 = np.asarray([sampled_data[j][3] for j in range(len(sampled_data))])
                # sample_input_data_t1 = np.asarray([sampled_data[j][4] for j in range(len(sampled_data))])
                # sample_target_data_t1 = np.asarray([sampled_data[j][5] for j in range(len(sampled_data))])
                # sample_trace_lengths_t1 = np.asarray([sampled_data[j][6] for j in range(len(sampled_data))])
                # sample_selection_matrix_t1 = np.asarray([sampled_data[j][7] for j in range(len(sampled_data))])
                # sample_r_t_batch = np.asarray([sampled_data[j][8] for j in range(len(sampled_data))])
                # sample_terminal_batch = np.asarray([sampled_data[j][10] for j in range(len(sampled_data))])
                # sample_cut_batch = np.asarray([sampled_data[j][11] for j in range(len(sampled_data))])
                sampled_outcome_t = np.asarray([sampled_data[j][9] for j in range(len(sampled_data))])
                pretrain_flag = False

                for i in range(0, len(terminal_batch)):
                    batch_terminal = terminal_batch[i]
                    batch_cut = cut_batch[i]
            else:
                sample_input_data_t0 = input_data_t0
                sample_target_data_t0 = target_data_t0
                sample_trace_lengths_t0 = trace_lengths_t0
                sample_selection_matrix_t0 = selection_matrix_t0
                sampled_outcome_t = win_id_t_batch

            kld_all, ll_all = train_clvrnn_model(model, sess, config, sample_input_data_t0, sample_target_data_t0,
                           sample_trace_lengths_t0, sample_selection_matrix_t0)
            kl_loss_game += kld_all
            ll_game += ll_all

            # """we skip sampling for TD learning"""
            if config.Learn.add_sarsa:
                train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, selection_matrix_t0,
                               input_data_t1, trace_lengths_t1, selection_matrix_t1, r_t_batch, terminal, cut)

            train_score_diff(model, sess, config, input_data_t0, trace_lengths_t0, selection_matrix_t0,
                             input_data_t1, trace_lengths_t1, selection_matrix_t1, r_t_batch, outcome_data,
                             score_diff_base_t0, terminal, cut)

            if add_pred_flag:
                sampled_data = Prediction_MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                sample_pred_input_data = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                sample_pred_target_data = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                sample_pred_trace_lengths = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                sample_pred_selection_matrix = np.asarray([sampled_data[j][3] for j in range(len(sampled_data))])

                train_prediction(model, sess, config,
                                 sample_pred_input_data,
                                 sample_pred_target_data,
                                 sample_pred_trace_lengths,
                                 sample_pred_selection_matrix)

        else:
            # for i in range(0, len(r_t_batch)):
            #     if i == len(r_t_batch) - 1:
            #         if terminal or cut:
            #             print(r_t_batch[i])
            if validate_cvrnn_flag:
                output_decoder = clvrnn_validation(sess, model, input_data_t0, target_data_t0, trace_lengths_t0,
                                                   selection_matrix_t0, config)

                if output_decoder_all is None:
                    output_decoder_all = output_decoder
                    target_data_all = target_data_t0
                    selection_matrix_all = selection_matrix_t0
                else:
                    # try:
                    output_decoder_all = np.concatenate([output_decoder_all, output_decoder], axis=0)
                    # except:
                    #     print output_decoder_all.shape
                    #     print  output_decoder.shape
                    target_data_all = np.concatenate([target_data_all, target_data_t0], axis=0)
                    selection_matrix_all = np.concatenate([selection_matrix_all, selection_matrix_t0], axis=0)

            # if validate_td_flag:
            #     # validate_variance_flag = validate_variance_flag if train_number <= 500 else False
            #     q_values, match_q_values_players_dict = \
            #         td_validation(sess, model, trace_lengths_t0, selection_matrix_t0,
            #                       player_id_t0_batch, s_t0_batch, action_id_t0, input_data_t0,
            #                       train_mask, config, match_q_values_players_dict,
            #                       r_t_batch, terminal, cut, train_number, validate_variance_flag=False)
            #
            #     if q_values_all is None:
            #         q_values_all = q_values
            #     else:
            #         q_values_all = np.concatenate([q_values_all, q_values], axis=0)

            if validate_diff_flag:
                output_label, real_label = diff_validation(sess, model, input_data_t0, trace_lengths_t0,
                                                           selection_matrix_t0,
                                                           score_diff_base_t0,
                                                           config, outcome_data)
                if real_label_all is None:
                    real_label_all = real_label
                else:
                    real_label_all = np.concatenate([real_label_all, real_label], axis=0)

                if output_label_all is None:
                    output_label_all = output_label
                else:
                    output_label_all = np.concatenate([output_label_all, output_label], axis=0)

        s_t0 = s_tl
        if terminal:
            break

    if validate_predict_flag and add_pred_flag:
        input_data, pred_output_prob = prediction_validation(model, sess, config,
                                                             pred_input_data,
                                                             pred_target_data,
                                                             pred_trace_lengths,
                                                             pred_selection_matrix)
        if pred_target_data_all is None:
            pred_target_data_all = pred_target_data
        else:
            pred_target_data_all = np.concatenate([pred_target_data_all, pred_target_data], axis=0)

        if pred_output_prob_all is None:
            pred_output_prob_all = pred_output_prob
        else:
            pred_output_prob_all = np.concatenate([pred_output_prob_all, pred_output_prob], axis=0)

    return [output_decoder_all, target_data_all, selection_matrix_all,
            q_values_all, real_label_all, output_label_all,
            pred_target_data_all,
            pred_output_prob_all,
            match_q_values_players_dict]


def run_network(sess, model, config, log_dir, save_network_dir,
                training_dir_games_all, testing_dir_games_all,
                data_store, source_data_dir, player_id_cluster_dir,
                load_network_dir=None, training_file=None):
    game_number = 0
    converge_flag = False
    saver = tf.train.Saver(max_to_keep=300)

    if load_network_dir is not None:  # resume the training
        check_point_game_number = int(load_network_dir.split("--")[-1])
        game_number = check_point_game_number
        global TRAINING_ITERATIONS
        TRAINING_ITERATIONS = game_number*3000
        saver.restore(sess, load_network_dir)
        print("Successfully loaded:", load_network_dir)

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / config.Learn.number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= len(training_dir_games_all) * config.Learn.iterate_num:
            break
        # else:
        #     converge_flag = True
        for dir_game in training_dir_games_all:
            if dir_game == '.DS_Store':
                continue
            game_number += 1
            print ("\ntraining file" + str(dir_game))
            gathering_data_and_run(dir_game, config,
                                   player_id_cluster_dir, data_store, source_data_dir, model, sess,
                                   training_flag=True, game_number=game_number)
            if game_number % 100 == 1:
                # save_model(game_number, saver, sess, save_network_dir, config)
                validate_model(testing_dir_games_all, data_store, source_data_dir, config,
                               sess, model, player_id_cluster_dir,
                               train_game_number=game_number,
                               validate_cvrnn_flag=True,
                               validate_td_flag=True,
                               validate_diff_flag=True,
                               validate_pred_flag=True)


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 300 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.data_name + '-game-',
                   global_step=game_number)


# def train_win_prob(model, sess, config, input_data, trace_lengths, selection_matrix, outcome_data):
#     [
#         _,
#         win_output
#     ] = sess.run([
#         model.train_win_op,
#         model.win_output
#     ],
#         feed_dict={model.input_data_ph: input_data,
#                    model.win_target_ph: outcome_data,
#                    model.trace_length_ph: trace_lengths,
#                    model.selection_matrix_ph: selection_matrix
#                    })
#
#     output_label = np.argmax(win_output, axis=1)
#     real_label = np.argmax(outcome_data, axis=1)
#
#     correct_num = 0
#     for index in range(0, len(input_data)):
#         if output_label[index] == real_label[index]:
#             correct_num += 1
#
#     # print('accuracy of win prob is {0}'.format(str(float(correct_num)/len(input_data))))

def train_prediction(model, sess, config, input_data, target_data, trace_lengths, selection_matrix):
    [
        output_prob,
        _
    ] = sess.run([
        model.action_pred_output,
        model.train_action_pred_op],
        feed_dict={
            model.selection_matrix_ph: selection_matrix,
            model.input_data_ph: input_data,
            model.action_pred_target_ph: target_data,
            model.trace_length_ph: trace_lengths}
    )
    TP, TN, FP, FN, acc, ll, auc = compute_acc(target_data, output_prob,
                                               if_binary_result=True,
                                               if_add_ll = True,
                                               if_print=False)
    precision = float(TP) / (TP + FP) if TP > 0 else 0
    recall = float(TP) / (TP + FN) if FN > 0 else None
    if TRAINING_ITERATIONS % 20 == 0:
        print ("Prediction acc is {0} with precision {1} and recall {2}".format(acc, precision, recall))


def train_score_diff(model, sess, config, input_data_t0, trace_lengths_t0, selection_matrix_t0,
                     input_data_t1, trace_lengths_t1, selection_matrix_t1, reward_t, outcome_data,
                     score_diff_base_t0, terminal, cut):

    [readout_t1_batch] = sess.run([model.diff_output],
                                  feed_dict={model.selection_matrix_ph: selection_matrix_t1,
                                             model.input_data_ph: input_data_t1,
                                             model.trace_length_ph: trace_lengths_t1})
    y_batch = []
    for i in range(0, len(readout_t1_batch)):
        if cut and i == len(readout_t1_batch) - 1:
            y_home = readout_t1_batch[i].tolist()[0] + float(reward_t[i][0])
            y_away = readout_t1_batch[i].tolist()[1] + float(reward_t[i][1])
            y_end = readout_t1_batch[i].tolist()[2] + float(reward_t[i][2])
            # print([y_home, y_away, y_end])
            y_batch.append([y_home, y_away, y_end])
            break
        # if terminal, only equals reward
        if terminal and i == len(readout_t1_batch) - 1:
            y_home = float(reward_t[i][0])
            y_away = float(reward_t[i][1])
            y_end = float(reward_t[i][2])
            print('game is ending with {0}'.format(str([y_home, y_away, y_end])))
            y_batch.append([y_home, y_away, y_end])
            break
        else:
            y_home = readout_t1_batch[i].tolist()[0] + float(reward_t[i][0])
            # y_away = readout_t1_batch[i].tolist()[1]
            y_away = readout_t1_batch[i].tolist()[1] + float(reward_t[i][1])
            y_end = readout_t1_batch[i].tolist()[2] + float(reward_t[i][2])
            y_batch.append([y_home, y_away, y_end])

    train_list = [model.diff_output, model.diff, model.train_diff_op]

    train_outputs = \
        sess.run(
            train_list,
            feed_dict={
                model.selection_matrix_ph: selection_matrix_t0,
                model.input_data_ph: input_data_t0,
                model.trace_length_ph: trace_lengths_t0,
                model.score_diff_target_ph: y_batch
            }
        )
    if terminal or cut:
        print('the avg Q values are home {0}, away {1} '
              'and end {2} with diff {3}'.format(np.mean(train_outputs[0][:, 0]),
                                                 np.mean(train_outputs[0][:, 1]),
                                                 np.mean(train_outputs[0][:, 2]),
                                                 np.mean(train_outputs[1])))
    output_label = train_outputs[0][:, 0] - train_outputs[0][:, 1] + np.asarray(score_diff_base_t0)
    real_label = outcome_data


    return output_label, real_label


def train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, selection_matrix_t0,
                   input_data_t1, trace_lengths_t1, selection_matrix_t1, r_t_batch, terminal, cut):
    [readout_t1_batch] = sess.run([model.sarsa_output],
                                  feed_dict={model.input_data_ph: input_data_t1,
                                             model.trace_length_ph: trace_lengths_t1,
                                             model.selection_matrix_ph: selection_matrix_t1
                                             })
    # r_t_batch = safely_expand_reward(reward_batch=r_t_batch, max_trace_length=config.Learn.max_seq_length)
    y_batch = []
    # print(len(r_t_batch))
    # print(np.sum(np.asarray(r_t_batch), axis=0))
    for i in range(0, len(r_t_batch)):
        if i == len(r_t_batch) - 1:
            if terminal or cut:
                y_home = float((r_t_batch[i])[0])
                y_away = float((r_t_batch[i])[1])
                y_end = float((r_t_batch[i])[2])
                y_batch.append([y_home, y_away, y_end])
                print([y_home, y_away, y_end])
                break
        y_home = float((r_t_batch[i])[0]) + config.Learn.gamma * \
                 ((readout_t1_batch[i]).tolist())[0]
        y_away = float((r_t_batch[i])[1]) + config.Learn.gamma * \
                 ((readout_t1_batch[i]).tolist())[1]
        y_end = float((r_t_batch[i])[2]) + config.Learn.gamma * \
                ((readout_t1_batch[i]).tolist())[2]
        y_batch.append([y_home, y_away, y_end])

    # perform gradient step
    y_batch = np.asarray(y_batch)
    [
        # sarsa_y_last,
        # z_encoder_last,
        # select_index,
        # z_encoder,
        avg_diff,
        _,
        readout
    ] = sess.run(
        [
            # model.sarsa_y_last,
            # model.z_encoder_last,
            # model.select_index,
            # model.z_encoder,
            model.td_avg_diff,
            model.train_td_op,
            model.sarsa_output
        ],
        feed_dict={model.sarsa_target_ph: y_batch,
                   model.trace_length_ph: trace_lengths_t0,
                   model.input_data_ph: input_data_t0,
                   model.selection_matrix_ph: selection_matrix_t0,
                   }
    )

    # print('avg diff:{0}, avg Qs:{1}'.format(avg_diff, str(np.mean(readout, axis=0))))


def train_clvrnn_model(model, sess, config, input_data, target_data, trace_lengths, selection_matrix):
    global TRAINING_ITERATIONS
    beta_kld = TRAINING_ITERATIONS*0.00001 if TRAINING_ITERATIONS*0.00001 < 1 else 1

    [
        output_x,
        kl_loss,
        likelihood_loss,
        _
    ] = sess.run([
        model.output,
        model.kl_loss,
        model.likelihood_loss,
        model.train_general_op],
        feed_dict={model.input_data_ph: input_data,
                   model.target_data_ph: target_data,
                   model.trace_length_ph: trace_lengths,
                   model.selection_matrix_ph: selection_matrix,
                   model.kld_beta: beta_kld,
                   # model.kld_beta: 0,
                   }
    )
    output_decoder = []
    prior_output_decoder = []

    kld_all = []
    ll_all = []
    for batch_index in range(0, len(output_x)):
        output_decoder_batch = []
        prior_output_decoder_batch = []
        for trace_length_index in range(0, config.Learn.max_seq_length):
            if selection_matrix[batch_index][trace_length_index]:
                output_decoder_batch.append(output_x[batch_index][trace_length_index])
                kld_all.append(kl_loss[batch_index][trace_length_index])
                ll_all.append(likelihood_loss[batch_index][trace_length_index])
            else:
                output_decoder_batch.append(np.asarray([0] * config.Arch.CLVRNN.x_dim))
        output_decoder.append(output_decoder_batch)
        prior_output_decoder.append(prior_output_decoder_batch)
    output_decoder = np.asarray(output_decoder)

    acc = compute_rnn_acc(output_prob=output_decoder, target_label=target_data,
                          selection_matrix=selection_matrix, config=config)

    if TRAINING_ITERATIONS%20 ==0:
        print ("Embed acc is {0} with mean KLD:{1} and mean ll:{2} with beta:{3} ".format(acc,
                                                                                np.mean(kld_all),
                                                                                np.mean(ll_all),
                                                                                beta_kld
                                                                                ))
    TRAINING_ITERATIONS += 1
    # print acc
    # if cost_out > 0.0001: # TODO: we still need to consider how to define convergence
    #     converge_flag = False
    cost_out = likelihood_loss + kl_loss

    return kld_all, ll_all


def clvrnn_validation(sess, model, input_data_t0, target_data_t0,
                      trace_lengths_t0, selection_matrix_t0, config):


    run_model = [model.output]

    [
        output_x,
    ] = sess.run(run_model,
        feed_dict={model.input_data_ph: input_data_t0,
                   model.target_data_ph: target_data_t0,
                   model.trace_length_ph: trace_lengths_t0,
                   model.selection_matrix_ph: selection_matrix_t0}
    )

    output_decoder = []
    for batch_index in range(0, len(output_x)):
        output_decoder_batch = []
        for trace_length_index in range(0, config.Learn.max_seq_length):
            if selection_matrix_t0[batch_index][trace_length_index]:
                output_decoder_batch.append(output_x[batch_index][trace_length_index])
            else:
                output_decoder_batch.append(np.asarray([0] * config.Arch.CLVRNN.x_dim))
        output_decoder.append(output_decoder_batch)
    output_decoder = np.asarray(output_decoder)

    return output_decoder


def prediction_validation(model, sess, config, input_data, target_data, trace_lengths, selection_matrix):
    [
        output_prob,
        # _
    ] = sess.run([
        model.action_pred_output,
        # model.train_action_pred_op
    ],
        feed_dict={
            model.selection_matrix_ph: selection_matrix,
            model.input_data_ph: input_data,
            model.action_pred_target_ph: target_data,
            model.trace_length_ph: trace_lengths}
    )
    acc = compute_acc(output_prob, target_data, if_print=False)
    return input_data, output_prob


def td_validation(sess, model, trace_lengths_t0, selection_matrix_t0,
                  player_id_t0_batch, s_t0_batch, action_id_t0, input_data_t0, train_mask, config,
                  match_q_values_players_dict, r_t_batch, terminal, cut, train_number,
                  validate_variance_flag):
    if validate_variance_flag:
        all_player_id = generate_diff_player_cluster_id(player_id_t0_batch)

        # r_t_batch = safely_expand_reward(reward_batch=r_t_batch, max_trace_length=config.Learn.max_seq_length)
        for i in range(0, len(r_t_batch)):
            if i == len(r_t_batch) - 1:
                if terminal or cut:
                    y_home = float((r_t_batch[i])[0])
                    y_away = float((r_t_batch[i])[1])
                    y_end = float((r_t_batch[i])[2])
                    print ('reward {0} in train number {1}'.format(str([y_home, y_away, y_end]), str(train_number)))
                    break

        readout_var_all = []
        for index in range(0, len(all_player_id)):
            player_id_batch = all_player_id[index]
            match_q_values = []
            input_data_var = np.concatenate([np.asarray(player_id_batch), np.asarray(s_t0_batch),
                                             np.asarray(action_id_t0), train_mask], axis=2)
            [readout_var] = sess.run([model.sarsa_output],
                                     feed_dict={model.input_data_ph: input_data_var,
                                                model.trace_length_ph: trace_lengths_t0,
                                                model.selection_matrix_ph: selection_matrix_t0
                                                })
            for i in range(len(input_data_var)):
                match_q_values.append(readout_var[i])
                # match_q_values.append(readout_var[i * config.Learn.max_seq_length + trace_lengths_t0[i] - 1])
            match_q_values_player = match_q_values_players_dict.get(index)
            match_q_values_player += match_q_values
            match_q_values_players_dict.update({index: match_q_values_player})

            # readout_var_masked = q_values_output_mask(q_values=readout_var, trace_lengths=trace_lengths_t0,
            #                                           max_trace_length=config.Learn.max_seq_length)
            readout_var_all.append(readout_var)
        var_all = np.var(np.asarray(readout_var_all), axis=0)

        print('The mean of q values variance is {0}'.format(np.mean(var_all)))

    [readout] = sess.run([model.sarsa_output],
                         feed_dict={model.input_data_ph: input_data_t0,
                                    model.trace_length_ph: trace_lengths_t0,
                                    model.selection_matrix_ph: selection_matrix_t0
                                    })
    # readout_masked = q_values_output_mask(q_values=readout, trace_lengths=trace_lengths_t0,
    #                                       max_trace_length=config.Learn.max_seq_length)
    return readout, match_q_values_players_dict


# def win_validation(sess, model, input_data,
#                    trace_lengths, selection_matrix,
#                    config, outcome_data):
#     [
#         # _,
#         win_output
#     ] = sess.run([
#         # model.train_win_op,
#         model.win_output],
#         feed_dict={model.input_data_ph: input_data,
#                    model.win_target_ph: outcome_data,
#                    model.trace_length_ph: trace_lengths,
#                    model.selection_matrix_ph: selection_matrix
#                    })
#     output_label = np.argmax(win_output, axis=1)
#     real_label = np.argmax(outcome_data, axis=1)
#
#     # correct_num = 0
#     # for index in range(0, len(input_data)):
#     #     if output_label[index] == real_label[index]:
#     #         correct_num += 1
#     #
#     return output_label, real_label


def diff_validation(sess, model, input_data, trace_lengths,
                    selection_matrix_t0,
                    score_diff_base_t0,
                    config, outcome_data):
    if config.Learn.diff_apply_rl:
        train_outputs = sess.run([model.diff_output],
                                 feed_dict={model.input_data_ph: input_data,
                                            model.trace_length_ph: trace_lengths,
                                            model.selection_matrix_ph: selection_matrix_t0})
        if train_outputs[0].shape[0] > 1:
            output_label = train_outputs[0][:, 0] - train_outputs[0][:, 1] + np.asarray(score_diff_base_t0)
            real_label = outcome_data
        else:
            output_label = [train_outputs[0][0][0] - train_outputs[0][0][1] + score_diff_base_t0[0]]
            real_label = outcome_data
    else:
        outcome_data = [[outcome_data[i]] for i in range(len(outcome_data))]
        [
            # _,
            diff_output
        ] = sess.run([
            # model.train_win_op,
            model.read_out],
            feed_dict={model.rnn_input_ph: input_data,
                       model.y_ph: outcome_data,
                       model.trace_lengths_ph: trace_lengths
                       })
        if diff_output.shape[0] > 1:
            # shape = diff_output.shape
            output_label = np.squeeze(diff_output, axis=1)
            real_label = np.squeeze(outcome_data, axis=1)
        else:
            output_label = diff_output[0]
            real_label = outcome_data[0]

    # correct_num = 0
    # for index in range(0, len(input_data)):
    #     if output_label[index] == real_label[index]:
    #         correct_num += 1
    #
    return output_label, real_label


def validate_model(testing_dir_games_all, data_store, source_data_dir, config, sess, model,
                   player_id_cluster_dir, train_game_number, validate_cvrnn_flag, validate_td_flag,
                   validate_diff_flag, validate_pred_flag, file_writer=None):
    output_decoder_all = None
    target_data_all = None
    selection_matrix_all = None
    q_values_all = None
    validate_variance_flag = False
    pred_target_data_all = None
    pred_output_prob_all = None

    if validate_diff_flag:
        real_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1
        output_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1

    print('validating model')
    for dir_index in range(0, len(testing_dir_games_all)):

        real_label_all = None
        output_label_all = None

        dir_game = testing_dir_games_all[dir_index]
        print('validating game {0}'.format(str(dir_game)))
        if dir_game == '.DS_Store':
            continue

        [output_decoder_all,
         target_data_all,
         selection_matrix_all,
         q_values_all,
         real_label_all,
         output_label_all,
         pred_target_data_all,
         pred_output_prob_all,
         match_q_values_players_dict] = gathering_data_and_run(dir_game, config,
                                                               player_id_cluster_dir,
                                                               data_store,
                                                               source_data_dir,
                                                               model, sess,
                                                               training_flag=False,
                                                               game_number=None,
                                                               validate_cvrnn_flag=validate_cvrnn_flag,
                                                               validate_td_flag=validate_td_flag,
                                                               validate_diff_flag=validate_diff_flag,
                                                               validate_variance_flag=validate_variance_flag,
                                                               validate_predict_flag=validate_pred_flag,
                                                               output_decoder_all=output_decoder_all,
                                                               target_data_all=target_data_all,
                                                               selection_matrix_all=selection_matrix_all,
                                                               q_values_all=q_values_all,
                                                               output_label_all=output_label_all,
                                                               real_label_all=real_label_all,
                                                               pred_target_data_all=pred_target_data_all,
                                                               pred_output_prob_all=pred_output_prob_all
                                                               )
        # validate_variance_flag = False
        # if match_q_values_players_dict is not None:
        #     plot_players_games(match_q_values_players_dict, train_game_number)

        if validate_diff_flag:
            real_label_record[dir_index][:len(real_label_all)] = real_label_all[:len(real_label_all)]
            output_label_record[dir_index][:len(output_label_all)] = output_label_all[:len(output_label_all)]

    if validate_cvrnn_flag:
        acc = compute_rnn_acc(output_prob=output_decoder_all, target_label=target_data_all,
                              selection_matrix=selection_matrix_all, config=config, if_print=True)
        print ("testing acc is {0}".format(str(acc)))
        if file_writer is not None:
            file_writer.write("testing acc is {0}\n".format(str(acc)))
    # if validate_td_flag:
    #     print ("testing avg qs is {0}".format(str(np.mean(q_values_all, axis=0))))
    #     if file_writer is not None:
    #         file_writer.write("testing avg qs is {0}\n".format(str(np.mean(q_values_all, axis=0))))

    if validate_diff_flag:
        # print ('general real label is {0}'.format(str(np.sum(real_label_record, axis=1))))
        # print ('general output label is {0}'.format(str(np.sum(output_label_record, axis=1))))
        for i in range(0, output_label_record.shape[1]):
            real_outcome_record_step = real_label_record[:, i]
            model_output_record_step = output_label_record[:, i]
            diff_sum = 0
            total_number = 0
            print_flag = True
            for win_index in range(0, len(real_outcome_record_step)):
                if model_output_record_step[win_index] == -100 or real_outcome_record_step[win_index] == -100:
                    print_flag = True
                    continue
                diff = abs(model_output_record_step[win_index] - real_outcome_record_step[win_index])
                diff_sum += diff
                total_number += 1
            if print_flag:
                if i % 100 == 0 and total_number > 0:
                    print('diff of time {0} is {1}'.format(str(i), str(float(diff_sum) / total_number)))
                    if file_writer is not None:
                        file_writer.write(
                            'diff of time {0} is {1}\n'.format(str(i), str(float(diff_sum) / total_number)))
    if validate_pred_flag:
        TP, TN, FP, FN, acc, ll, auc = compute_acc(pred_target_data_all, pred_output_prob_all,
                                                   if_binary_result=True, if_print=False, if_add_ll = True)
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        print ("Prediction acc is {0} with precision {1} and recall {2}".format(acc, precision, recall))



def run():
    local_test_flag = False
    box_msg = '' # We can try adding pre-game box scores.
    predict_action = '_predict_nex_goal'  # Our prediction target is expected goal
    embed_mode = '_embed_random'  # player embedding model

    running_number = 0  # running_number is [0, 1, 2, 3, 4] for 5-fold cross validation

    load_icehockey_game_name = ''  # you can load a break point
    player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
    predicted_target = '_PlayerLocalId'  # we reidentify the player id

    icehockey_clvrnn_config_path = "../environment_settings/" \
                                  "icehockey_varlea{0}{2}_config{1}{3}.yaml".format(predicted_target,
                                                                                       box_msg,
                                                                                       predict_action,
                                                                                       embed_mode)
    icehockey_varlea_config = VaRLEACongfig.load(icehockey_clvrnn_config_path)
    Prediction_MemoryBuffer.set_cache_memory(cache_number=icehockey_varlea_config.Arch.Predict.output_size)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_varlea_config,
                                                        model_catagoery='varlea', running_number=running_number,
                                                        date_msg='')
    if len(load_icehockey_game_name) > 0:
        load_network_dir = saved_network_dir+'/'+load_icehockey_game_name
    else:
        load_network_dir = None

    source_data_dir = icehockey_varlea_config.Learn.save_mother_dir + '/oschulte/Galen/2018-2019/'  # you source data (before pre-preprocessing)
    data_store_dir = icehockey_varlea_config.Learn.save_mother_dir + '/oschulte/Galen/Ice-hockey-data/2018-2019/' # your source data (after pre-preprocessing)
    dir_games_all = os.listdir(data_store_dir)
    # shuffle(dir_games_all)  # randomly shuffle the list
    if running_number == 0:
        training_dir_games_all = dir_games_all[
                                 0: len(dir_games_all) / 5 * 4 - running_number * len(dir_games_all) / 5]
    else:
        training_dir_games_all = dir_games_all[
                                 0: len(dir_games_all) / 5 * 4 - running_number * len(dir_games_all) / 5] \
                                 + dir_games_all[-running_number * len(dir_games_all) / 5:]

    test_validate_dir_games_all = [item for item in dir_games_all if item not in training_dir_games_all]

    testing_dir_games_all = test_validate_dir_games_all[:len(test_validate_dir_games_all)/2]
    validate_dir_games_all = test_validate_dir_games_all[len(test_validate_dir_games_all) / 2:]
    tmp_testing_dir_games_all = testing_dir_games_all[-10:] # TODO: it is a small running testing, not the real one
    number_of_total_game = len(dir_games_all)
    icehockey_varlea_config.Learn.number_of_total_game = number_of_total_game

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sess = tf.Session()
    clvrnn = VaRLEA(config=icehockey_varlea_config,
                    train_flag=True,
                    extra_prediction_flag=True,
                    deterministic_decoder=True)
    clvrnn()
    sess.run(tf.global_variables_initializer())

    if not local_test_flag:
        if not os.path.exists(saved_network_dir):
            os.mkdir(saved_network_dir)
        # save the training and testing dir list
        if os.path.exists(saved_network_dir + '/training_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/training_file_dirs_all.csv',
                      saved_network_dir + '/bak_training_file_dirs_all_{0}.csv'
                      .format(datetime.date.today().strftime("%Y%B%d")))
        if os.path.exists(saved_network_dir + '/testing_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/testing_file_dirs_all.csv',
                      saved_network_dir + '/bak_testing_file_dirs_all_{0}.csv'
                      .format(datetime.date.today().strftime("%Y%B%d")))
        if os.path.exists(saved_network_dir + '/validate_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/validate_file_dirs_all.csv',
                      saved_network_dir + '/bak_validate_file_dirs_all_{0}.csv'
                      .format(datetime.date.today().strftime("%Y%B%d")))
        with open(saved_network_dir + '/training_file_dirs_all.csv', 'wb') as f:
            for dir in training_dir_games_all:
                f.write(dir + '\n')
        with open(saved_network_dir + '/validate_file_dirs_all.csv', 'wb') as f:
            for dir in validate_dir_games_all:
                f.write(dir + '\n')
        with open(saved_network_dir + '/testing_file_dirs_all.csv', 'wb') as f:
            for dir in testing_dir_games_all:
                f.write(dir + '\n')
    print('training the model.')

    run_network(sess=sess, model=clvrnn, config=icehockey_varlea_config, log_dir=log_dir,
                save_network_dir=saved_network_dir, data_store=data_store_dir, source_data_dir=source_data_dir,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=tmp_testing_dir_games_all,
                player_id_cluster_dir=player_id_cluster_dir, load_network_dir=load_network_dir)


if __name__ == '__main__':
    run()
