import os
import pickle
import random

import datetime
import tensorflow as tf
import numpy as np
import json
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from nn_structure.varlea import VaRLEA
from nn_structure.cvrnn import CVRNN
from support.data_processing_tools import get_icehockey_game_data, generate_selection_matrix, transfer2seq, \
    read_feature_within_events, read_features_within_events


# from support.plot_tools import plot_game_Q_values


class ExperienceReplayBuffer:
    def __init__(self, capacity_number):
        self.capacity = capacity_number
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[random.randint(0, len(self.memory) - 1)]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BalanceExperienceReplayBuffer:
    def __init__(self, capacity_number):
        self.capacity = capacity_number
        self.memory = []
        self.cache_number = None

    def set_cache_memory(self, cache_number):
        self.cache_number = cache_number
        for i in range(0, cache_number):
            self.memory.append([])

    def push(self, transition, cache_label):
        memory_selected = self.memory[cache_label]
        memory_selected.append(transition)
        if len(self.memory[cache_label]) > self.capacity:
            del self.memory[cache_label][random.randint(0, len(self.memory) - 1)]

    def sample(self, batch_size):
        return_samples = []
        for i in range(0, batch_size):
            cache_label = random.randint(0, self.cache_number - 1)
            sampled_point = random.sample(self.memory[cache_label], 1)
            return_samples.append(sampled_point[0])
        return return_samples

    def __len__(self):
        return len(self.memory)


def load_nn_model(saver, sess, saved_network_dir):
    # saver = tf.train.Saver()
    # merge = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    # sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(saved_network_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
        # game_number_checkpoint = check_point_game_number % config.number_of_total_game
        # game_number = check_point_game_number
        # game_starting_point = 0
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find the network: {0}", format(saved_network_dir))


def get_data_name(config, model_catagoery, model_number):
    player_info = ''
    if config.Learn.apply_box_score:
        player_info += '_box'
    if config.Learn.apply_pid:
        player_info += '_pid'
    if model_catagoery == 'cvrnn':
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        data_name = "model_{1}_three_cut_cvrnn_Qs_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}{12}_{13}".format(config.Learn.save_mother_dir,
                                                                                       model_number,
                                                                                       str(config.Learn.feature_type),
                                                                                       str(config.Learn.batch_size),
                                                                                       str(config.Learn.iterate_num),
                                                                                       str(config.Learn.learning_rate),
                                                                                       str(config.Learn.model_type),
                                                                                       str(config.Learn.max_seq_length),
                                                                                       str(
                                                                                           config.Arch.CVRNN.latent_dim),
                                                                                       str(config.Arch.CVRNN.y_dim),
                                                                                       str(config.Arch.CVRNN.x_dim),
                                                                                       str(
                                                                                           config.Arch.CVRNN.hidden_dim),
                                                                                       player_info,
                                                                                       config.Learn.embed_mode
                                                                                       )
    elif model_catagoery == 'varlea':
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        data_name = "model_{1}_three_cut_varlea_Qs_feature{2}_latent{8}_x{9}_ys{10}_ya{11}_yr{12}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{13}" \
                  "{14}_{15}".format(config.Learn.save_mother_dir,
                                               model_number,
                                               str(config.Learn.feature_type),
                                               str(config.Learn.batch_size),
                                               str(config.Learn.iterate_num),
                                               str(config.Learn.learning_rate),
                                               str(config.Learn.model_type),
                                               str(config.Learn.max_seq_length),
                                               str(config.Arch.CLVRNN.latent_a_dim),
                                               str(config.Arch.CLVRNN.x_dim),
                                               str(config.Arch.CLVRNN.y_s_dim),
                                               str(config.Arch.CLVRNN.y_a_dim),
                                               str(config.Arch.CLVRNN.y_r_dim),
                                               str(config.Arch.CLVRNN.hidden_dim),
                                               player_info,
                                               config.Learn.embed_mode
                                           )
    elif model_catagoery == 'caernn':
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        data_name = "model_{1}_three_cut_caernn_Qs_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}{12}_" \
                    "{13}".format(config.Learn.save_mother_dir,
                                  model_number,
                                  str(config.Learn.feature_type),
                                  str(config.Learn.batch_size),
                                  str(config.Learn.iterate_num),
                                  str(config.Learn.learning_rate),
                                  str(config.Learn.model_type),
                                  str(config.Learn.max_seq_length),
                                  str(config.Arch.CAERNN.latent_dim),
                                  str(config.Arch.CAERNN.y_dim),
                                  str(config.Arch.CAERNN.x_dim),
                                  str(config.Arch.CAERNN.hidden_dim),
                                  player_info,
                                  config.Learn.embed_mode
                                  )
    elif model_catagoery == 'cvae':
        data_name = "model_{1}_three_cut_cvae_Qs_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                                model_number,
                                                                str(config.Learn.feature_type),
                                                                str(config.Learn.batch_size),
                                                                str(config.Learn.iterate_num),
                                                                str(config.Learn.learning_rate),
                                                                str(config.Learn.model_type),
                                                                None,
                                                                str(config.Arch.CVAE.latent_dim),
                                                                str(config.Arch.CVAE.x_dim),
                                                                str(config.Arch.CVAE.y_dim),
                                                                None,
                                                                player_info
                                                                )
    elif model_catagoery == 'vhe':
        data_name = "model_{1}_three_cut_vhe_Qs_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                                model_number,
                                                                str(config.Learn.feature_type),
                                                                str(config.Learn.batch_size),
                                                                str(config.Learn.iterate_num),
                                                                str(config.Learn.learning_rate),
                                                                str(config.Learn.model_type),
                                                                None,
                                                                str(config.Arch.CVAE.latent_dim),
                                                                str(config.Arch.CVAE.x_dim),
                                                                str(config.Arch.CVAE.y_dim),
                                                                None,
                                                                player_info
                                                                )
    elif model_catagoery == 'encoder':
        data_name = "model_{1}_three_cut_encoder_Qs_feature{2}_embed{8}_in{9}_out{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                                model_number,
                                                                str(config.Learn.feature_type),
                                                                str(config.Learn.batch_size),
                                                                str(config.Learn.iterate_num),
                                                                str(config.Learn.learning_rate),
                                                                str(config.Learn.model_type),
                                                                None,
                                                                str(config.Arch.Encoder.embed_dim),
                                                                str(config.Arch.Encoder.input_dim),
                                                                str(config.Arch.Encoder.output_dim),
                                                                None,
                                                                player_info,
                                                                # lstm_msg
                                                                )
    elif model_catagoery == 'multi_agent':
        data_name = "model_{1}_three_cut_multi_agent_Qs_feature{2}_latent{8}_hidden{9}_lstm{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                                model_number,
                                                                str(config.Learn.feature_type),
                                                                str(config.Learn.batch_size),
                                                                str(config.Learn.iterate_num),
                                                                str(config.Learn.learning_rate),
                                                                str(config.Learn.model_type),
                                                                None,
                                                                str(config.Arch.Episodic.latent_dim),
                                                                str(config.Arch.Episodic.h_size),
                                                                str(config.Arch.Episodic.lstm_layer_num),
                                                                None,
                                                                player_info,
                                                                # lstm_msg
                                                                )
    elif model_catagoery == 'lstm_Qs':
        data_name = "model_{1}_three_cut_lstm_Qs_feature{2}_{8}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                    "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                            model_number,
                                            str(config.Learn.feature_type),
                                            str(config.Learn.batch_size),
                                            str(config.Learn.iterate_num),
                                            str(config.Learn.learning_rate),
                                            str(config.Learn.model_type),
                                            str(
                                                config.Learn.max_seq_length),
                                            config.Learn.predict_target,
                                            None,
                                            str(config.Arch.LSTM.h_size),
                                            str(
                                                config.Arch.Dense.hidden_size),
                                            player_info
                                            )
    elif model_catagoery == 'lstm_diff':
        data_name = "model_{1}_three_cut_lstm_Qs_feature{2}_{8}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                    "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                            model_number,
                                            str(config.Learn.feature_type),
                                            str(config.Learn.batch_size),
                                            str(config.Learn.iterate_num),
                                            str(config.Learn.learning_rate),
                                            str(config.Learn.model_type),
                                            str(
                                                config.Learn.max_seq_length),
                                            config.Learn.predict_target,
                                            None,
                                            str(config.Arch.LSTM.h_size),
                                            str(
                                                config.Arch.Dense.hidden_size),
                                            player_info
                                            )

    return data_name


def get_model_and_log_name(config, model_catagoery, train_flag=False,
                           embedding_tag=None, running_number=None, date_msg='', focus_condition='', reward_msg=''):
    if train_flag:
        train_msg = 'Train_'
    else:
        train_msg = ''

    player_info = ''
    if config.Learn.apply_box_score:
        player_info = '_box'

    if model_catagoery == 'cvrnn':  # TODO: add more parameters
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Learn.rnn_skip_player:
            player_info += '_skip'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target
        if not config.Learn.apply_stochastic:
            player_info += '_deter'
        log_dir = "{0}/oschulte/Galen/icehockey-models/cvrnn_log_NN" \
                  "/{1}cvrnn{16}_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}" \
                  "{12}_{13}_r{14}{15}".format(config.Learn.save_mother_dir,
                                           train_msg,
                                           str(config.Learn.feature_type),
                                           str(config.Learn.batch_size),
                                           str(config.Learn.iterate_num),
                                           str(config.Learn.learning_rate),
                                           str(config.Learn.model_type),
                                           str(config.Learn.max_seq_length),
                                           str(config.Arch.CVRNN.latent_dim),
                                           str(config.Arch.CVRNN.y_dim),
                                           # TODO: reorder x_dim and y_dim
                                           str(config.Arch.CVRNN.x_dim),
                                           str(config.Arch.CVRNN.hidden_dim),
                                           player_info,
                                           config.Learn.embed_mode,
                                           str(running_number),
                                               date_msg,
                                               focus_condition
                                           )

        saved_network = "{0}/oschulte/Galen/icehockey-models/cvrnn_saved_NN/" \
                        "{1}cvrnn{16}_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}" \
                        "{12}_{13}_r{14}{15}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            str(config.Arch.CVRNN.latent_dim),
            str(config.Arch.CVRNN.y_dim),
            # TODO: reorder x_dim and y_dim
            str(config.Arch.CVRNN.x_dim),
            str(config.Arch.CVRNN.hidden_dim),
            player_info,
            config.Learn.embed_mode,
            str(running_number),
            date_msg,
            focus_condition
        )
    if model_catagoery == 'caernn':  # TODO: add more parameters
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target
        log_dir = "{0}/oschulte/Galen/icehockey-models/caernn_log_NN" \
                  "/{1}caernn_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}" \
                  "{12}_{13}_r{14}{15}".format(config.Learn.save_mother_dir,
                                           train_msg,
                                           str(config.Learn.feature_type),
                                           str(config.Learn.batch_size),
                                           str(config.Learn.iterate_num),
                                           str(config.Learn.learning_rate),
                                           str(config.Learn.model_type),
                                           str(config.Learn.max_seq_length),
                                           str(config.Arch.CAERNN.latent_dim),
                                           str(config.Arch.CAERNN.y_dim),
                                           # TODO: reorder x_dim and y_dim
                                           str(config.Arch.CAERNN.x_dim),
                                           str(config.Arch.CAERNN.hidden_dim),
                                           player_info,
                                           config.Learn.embed_mode,
                                           str(running_number),
                                               date_msg
                                           )

        saved_network = "{0}/oschulte/Galen/icehockey-models/caernn_saved_NN/" \
                        "{1}caernn_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}" \
                        "{12}_{13}_r{14}{15}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            str(config.Arch.CAERNN.latent_dim),
            str(config.Arch.CAERNN.y_dim),
            # TODO: reorder x_dim and y_dim
            str(config.Arch.CAERNN.x_dim),
            str(config.Arch.CAERNN.hidden_dim),
            player_info,
            config.Learn.embed_mode,
            str(running_number),
            date_msg
        )
    elif 'varlea' in model_catagoery:  # TODO: add more parameters
        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Learn.rnn_skip_player:
            player_info += '_skip'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target
        if not config.Learn.apply_stochastic:
            player_info += '_deter'
        log_dir = "{0}/oschulte/Galen/icehockey-models/varlea_log_NN" \
                  "/{1}varlea_log_feature{2}_latent{8}_x{9}_ys{10}_ya{11}_yr{12}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{13}" \
                  "{14}_{15}_r{16}{17}{18}".format(config.Learn.save_mother_dir,
                                               train_msg,
                                               str(config.Learn.feature_type),
                                               str(config.Learn.batch_size),
                                               str(config.Learn.iterate_num),
                                               str(config.Learn.learning_rate),
                                               str(config.Learn.model_type),
                                               str(config.Learn.max_seq_length),
                                               str(config.Arch.CLVRNN.latent_a_dim),
                                               str(config.Arch.CLVRNN.x_dim),
                                               str(config.Arch.CLVRNN.y_s_dim),
                                               str(config.Arch.CLVRNN.y_a_dim),
                                               str(config.Arch.CLVRNN.y_r_dim),
                                               str(config.Arch.CLVRNN.hidden_dim),
                                               player_info,
                                               config.Learn.embed_mode,
                                               str(running_number),
                                               date_msg,
                                                   reward_msg
                                           )

        saved_network = "{0}/oschulte/Galen/icehockey-models/varlea_saved_NN/" \
                        "{1}clvrnn_saved_networks_feature{2}_latent{8}_x{9}_ys{10}_ya{11}_yr{12}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{13}" \
                        "{14}_{15}_r{16}{17}{18}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            str(config.Arch.CLVRNN.latent_a_dim),
            str(config.Arch.CLVRNN.x_dim),
            str(config.Arch.CLVRNN.y_s_dim),
            str(config.Arch.CLVRNN.y_a_dim),
            str(config.Arch.CLVRNN.y_r_dim),
            str(config.Arch.CLVRNN.hidden_dim),
            player_info,
            config.Learn.embed_mode,
            str(running_number),
            date_msg,
            reward_msg
        )
    elif model_catagoery == 'de_embed':
        if embedding_tag is not None:
            train_msg += 'validate{0}_'.format(str(embedding_tag))

        log_dir = "{0}/oschulte/Galen/icehockey-models/de_log_NN" \
                  "/{1}de_embed_log_feature{2}_{8}_embed{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}_r{13}".format(config.Learn.save_mother_dir,
                                                train_msg,
                                                str(config.Learn.feature_type),
                                                str(config.Learn.batch_size),
                                                str(config.Learn.iterate_num),
                                                str(config.Learn.learning_rate),
                                                str(config.Learn.model_type),
                                                str(config.Learn.max_seq_length),
                                                config.Learn.predict_target,
                                                str(config.Arch.Encode.latent_size),
                                                str(config.Arch.LSTM.h_size),
                                                str(config.Arch.Dense.hidden_node_size),
                                                player_info,
                                                str(running_number)
                                                )

        saved_network = "{0}/oschulte/Galen/icehockey-models/de_model_saved_NN/" \
                        "{1}de_embed_saved_networks_feature{2}_{8}_embed{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}_r{13}".format(config.Learn.save_mother_dir,
                                                      train_msg,
                                                      str(config.Learn.feature_type),
                                                      str(config.Learn.batch_size),
                                                      str(config.Learn.iterate_num),
                                                      str(config.Learn.learning_rate),
                                                      str(config.Learn.model_type),
                                                      str(config.Learn.max_seq_length),
                                                      config.Learn.predict_target,
                                                      str(config.Arch.Encode.latent_size),
                                                      str(config.Arch.LSTM.h_size),
                                                      str(config.Arch.Dense.hidden_node_size),
                                                      player_info,
                                                      str(running_number)
                                                      )
    elif model_catagoery == 'mdn_Qs':
        log_dir = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_log_NN" \
                  "/{1}mdn_log_feature{2}_{8}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}_r{13}".format(config.Learn.save_mother_dir,
                                                train_msg,
                                                str(config.Learn.feature_type),
                                                str(config.Learn.batch_size),
                                                str(config.Learn.iterate_num),
                                                str(config.Learn.learning_rate),
                                                str(config.Learn.model_type),
                                                str(config.Learn.max_seq_length),
                                                config.Learn.predict_target,
                                                None,
                                                str(config.Arch.LSTM.h_size),
                                                str(config.Arch.Dense.hidden_size),
                                                player_info,
                                                str(running_number)
                                                )

        saved_network = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_model_saved_NN/" \
                        "{1}mdn_embed_saved_networks_feature{2}_{8}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}_r{13}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            player_info,
            str(running_number)
        )

    elif model_catagoery == 'lstm_Qs':
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''
        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_Qs_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                    train_msg,
                                                    str(config.Learn.feature_type),
                                                    str(config.Learn.batch_size),
                                                    str(config.Learn.iterate_num),
                                                    str(config.Learn.learning_rate),
                                                    str(config.Learn.model_type),
                                                    str(config.Learn.max_seq_length),
                                                    config.Learn.predict_target,
                                                    None,
                                                    str(config.Arch.LSTM.h_size),
                                                    str(config.Arch.Dense.hidden_size),
                                                    player_info,
                                                    player_id_info,
                                                    str(running_number)
                                                    )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_Qs_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}_r{14}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            player_info,
            player_id_info,
            str(running_number)
        )

    elif model_catagoery == 'lstm_win':
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''
        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_win_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                    train_msg,
                                                    str(config.Learn.feature_type),
                                                    str(config.Learn.batch_size),
                                                    str(config.Learn.iterate_num),
                                                    str(config.Learn.learning_rate),
                                                    str(config.Learn.model_type),
                                                    str(config.Learn.max_seq_length),
                                                    config.Learn.predict_target,
                                                    None,
                                                    str(config.Arch.LSTM.h_size),
                                                    str(config.Arch.Dense.hidden_size),
                                                    player_info,
                                                    player_id_info,
                                                    str(running_number)
                                                    )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_win_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}_r{14}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            player_info,
            player_id_info,
            str(running_number)
        )

    elif model_catagoery == 'lstm_diff':

        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''

        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_diff_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                    train_msg,
                                                    str(config.Learn.feature_type),
                                                    str(config.Learn.batch_size),
                                                    str(config.Learn.iterate_num),
                                                    str(config.Learn.learning_rate),
                                                    str(config.Learn.model_type),
                                                    str(config.Learn.max_seq_length),
                                                    config.Learn.predict_target,
                                                    None,
                                                    str(config.Arch.LSTM.h_size),
                                                    str(config.Arch.Dense.hidden_size),
                                                    player_info,
                                                    player_id_info,
                                                    str(running_number)
                                                    )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_diff_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}_r{14}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            player_info,
            player_id_info,
            str(running_number)
        )
    elif model_catagoery == 'vhe':  # TODO: add more parameters

        if config.Learn.apply_lstm:
            lstm_msg = '_lstm'
        else:
            lstm_msg = ''

        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target

        log_dir = "{0}/oschulte/Galen/icehockey-models/vhe_saved_NN" \
                  "/{1}vhe_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        None,
                                                                        str(config.Arch.CVAE.latent_dim),
                                                                        str(config.Arch.CVAE.x_dim),
                                                                        str(config.Arch.CVAE.y_dim),
                                                                        None,
                                                                        player_info,
                                                                        lstm_msg,
                                                                        str(running_number)
                                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/vhe_saved_NN/" \
                        "{1}vhe_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                              train_msg,
                                                                              str(config.Learn.feature_type),
                                                                              str(config.Learn.batch_size),
                                                                              str(config.Learn.iterate_num),
                                                                              str(config.Learn.learning_rate),
                                                                              str(config.Learn.model_type),
                                                                              None,
                                                                              str(config.Arch.CVAE.latent_dim),
                                                                              str(config.Arch.CVAE.x_dim),
                                                                              str(config.Arch.CVAE.y_dim),
                                                                              None,
                                                                              player_info,
                                                                              lstm_msg,
                                                                              str(running_number)
                                                                              )

    elif model_catagoery == 'cvae':  # TODO: add more parameters

        if config.Learn.apply_lstm:
            lstm_msg = '_lstm'
        else:
            lstm_msg = ''

        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target

        log_dir = "{0}/oschulte/Galen/icehockey-models/cvae_saved_NN" \
                  "/{1}cvae_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        None,
                                                                        str(config.Arch.CVAE.latent_dim),
                                                                        str(config.Arch.CVAE.x_dim),
                                                                        str(config.Arch.CVAE.y_dim),
                                                                        None,
                                                                        player_info,
                                                                        lstm_msg,
                                                                        str(running_number)
                                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/cvae_saved_NN/" \
                        "{1}cvae_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                              train_msg,
                                                                              str(config.Learn.feature_type),
                                                                              str(config.Learn.batch_size),
                                                                              str(config.Learn.iterate_num),
                                                                              str(config.Learn.learning_rate),
                                                                              str(config.Learn.model_type),
                                                                              None,
                                                                              str(config.Arch.CVAE.latent_dim),
                                                                              str(config.Arch.CVAE.x_dim),
                                                                              str(config.Arch.CVAE.y_dim),
                                                                              None,
                                                                              player_info,
                                                                              lstm_msg,
                                                                              str(running_number)
                                                                              )
    elif model_catagoery == 'auto_encoder':

        if config.Learn.apply_lstm:
            lstm_msg = '_lstm'
        else:
            lstm_msg = ''

        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target

        log_dir = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN" \
                  "/{1}auto_encoder_log_feature{2}_embed{8}_in{9}_out{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        None,
                                                                        str(config.Arch.Encoder.embed_dim),
                                                                        str(config.Arch.Encoder.input_dim),
                                                                        str(config.Arch.Encoder.output_dim),
                                                                        None,
                                                                        player_info,
                                                                        lstm_msg,
                                                                        str(running_number)
                                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN/" \
                        "{1}auto_encoder_saved_networks_feature{2}_embed{8}_in{9}_out{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                              train_msg,
                                                                              str(config.Learn.feature_type),
                                                                              str(config.Learn.batch_size),
                                                                              str(config.Learn.iterate_num),
                                                                              str(config.Learn.learning_rate),
                                                                              str(config.Learn.model_type),
                                                                              None,
                                                                              str(config.Arch.Encoder.embed_dim),
                                                                              str(config.Arch.Encoder.input_dim),
                                                                              str(config.Arch.Encoder.output_dim),
                                                                              None,
                                                                              player_info,
                                                                              lstm_msg,
                                                                              str(running_number)
                                                                              )

    elif model_catagoery == 'encoder':

        if config.Learn.apply_lstm:
            lstm_msg = '_lstm'
        else:
            lstm_msg = ''

        if config.Learn.integral_update_flag:
            player_info += '_integral'
        if config.Arch.Predict.predict_target is not None:
            player_info += '_' + config.Arch.Predict.predict_target

        log_dir = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN" \
                  "/{1}encoder_log_feature{2}_embed{8}_in{9}_out{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        None,
                                                                        str(config.Arch.Encoder.embed_dim),
                                                                        str(config.Arch.Encoder.input_dim),
                                                                        str(config.Arch.Encoder.output_dim),
                                                                        None,
                                                                        player_info,
                                                                        lstm_msg,
                                                                        str(running_number)
                                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN/" \
                        "{1}encoder_saved_networks_feature{2}_embed{8}_in{9}_out{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}_r{14}".format(config.Learn.save_mother_dir,
                                                                              train_msg,
                                                                              str(config.Learn.feature_type),
                                                                              str(config.Learn.batch_size),
                                                                              str(config.Learn.iterate_num),
                                                                              str(config.Learn.learning_rate),
                                                                              str(config.Learn.model_type),
                                                                              None,
                                                                              str(config.Arch.Encoder.embed_dim),
                                                                              str(config.Arch.Encoder.input_dim),
                                                                              str(config.Arch.Encoder.output_dim),
                                                                              None,
                                                                              player_info,
                                                                              lstm_msg,
                                                                              str(running_number)
                                                                              )

    elif model_catagoery == 'lstm_prediction':

        predict_target = config.Learn.predict_target
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''

        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_predict_log_NN" \
                  "/{1}lstm_predict_{13}_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}{14}_r{15}".format(config.Learn.save_mother_dir,
                                                        train_msg,
                                                        str(config.Learn.feature_type),
                                                        str(config.Learn.batch_size),
                                                        str(config.Learn.iterate_num),
                                                        str(config.Learn.learning_rate),
                                                        str(config.Learn.model_type),
                                                        str(config.Learn.max_seq_length),
                                                        config.Learn.predict_target,
                                                        None,
                                                        str(config.Arch.LSTM.h_size),
                                                        str(config.Arch.Dense.dense_layer_size),
                                                        player_info,
                                                        predict_target,
                                                        player_id_info,
                                                        str(running_number)
                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_predict_model_saved_NN/" \
                        "{1}lstm_predict_{13}_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}{14}_r{15}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.dense_layer_size),
            player_info,
            predict_target,
            player_id_info,
            str(running_number)
        )
    elif model_catagoery == 'multi_agent':

        predict_target = config.Learn.predict_target
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''

        log_dir = "{0}/oschulte/Galen/icehockey-models/multi_agent_log_NN" \
                  "/{1}multi_agent_{13}_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}{14}_r{15}".format(config.Learn.save_mother_dir,
                                                        train_msg,
                                                        str(config.Learn.feature_type),
                                                        str(config.Learn.batch_size),
                                                        str(config.Learn.iterate_num),
                                                        str(config.Learn.learning_rate),
                                                        str(config.Learn.model_type),
                                                        str(config.Learn.max_seq_length),
                                                        config.Learn.predict_target,
                                                        None,
                                                        str(config.Arch.Episodic.h_size),
                                                        str(config.Arch.Dense.dense_layer_size),
                                                        player_info,
                                                        predict_target,
                                                        player_id_info,
                                                        str(running_number)

                                                        )

        saved_network = "{0}/oschulte/Galen/icehockey-models/multi_agent_model_saved_NN/" \
                        "{1}multi_agent_{13}_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}{14}_r{15}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.Episodic.h_size),
            str(config.Arch.Dense.dense_layer_size),
            player_info,
            predict_target,
            player_id_info,
            str(running_number)
        )

    return saved_network, log_dir


def compute_rnn_acc(target_label, output_prob, selection_matrix,
                    config, bounding_id_trace_length=None,
                    keep_flags_all=None, if_print=False, if_add_ll=False):
    total_number = 0
    correct_number = 0
    ll_sum = 0
    correct_output_all = {}
    bounding_ids_index = None
    for batch_index in range(0, len(selection_matrix)):
        if keep_flags_all is not None:
            if not keep_flags_all[batch_index]:
                continue

        if bounding_id_trace_length is not None and batch_index - bounding_id_trace_length - 1 > 0:
            bounding_ids = set()
            # if batch_index - bounding_id_trace_length - 1 > 0:
            bounding_end_index = batch_index - bounding_id_trace_length - 1
            for i in range(batch_index, bounding_end_index, -1):
                for trace_length_index in range(0, config.Learn.max_seq_length):
                    if selection_matrix[batch_index][trace_length_index]:
                        bounding_ids.add(np.argmax(target_label[i][trace_length_index]))
            bounding_ids_index = sorted(list(bounding_ids), reverse=True)


        for trace_length_index in range(0, config.Learn.max_seq_length):
            test_flag = False
            if selection_matrix[batch_index][trace_length_index] and trace_length_index+1 == config.Learn.max_seq_length:
                test_flag = True
            elif selection_matrix[batch_index][trace_length_index] and not selection_matrix[batch_index][trace_length_index+1]:
                test_flag = True

            if test_flag:
                sub_output_prob_bound = None
                total_number += 1
                if bounding_ids_index is not None:
                    sub_output_prob_bound = output_prob[batch_index][trace_length_index][bounding_ids_index]
                    sub_output_prob_bound = np.exp(sub_output_prob_bound) / sum(np.exp(sub_output_prob_bound))
                    sub_target_prob_bound = target_label[batch_index][trace_length_index][bounding_ids_index]
                    target_prediction_bound = np.argmax(sub_target_prob_bound)
                    output_prediction = bounding_ids_index[np.argmax(sub_output_prob_bound)]
                    bounding_ids_index = None
                else:
                    output_prediction = np.argmax(output_prob[batch_index][trace_length_index])
                # print output_prediction
                target_prediction = np.argmax(target_label[batch_index][trace_length_index])
                if if_add_ll:
                    if sub_output_prob_bound is not None:
                        likelihood = sub_output_prob_bound * sub_target_prob_bound
                        ll = np.log(likelihood[target_prediction_bound] + 1e-10)
                    else:
                        likelihood = output_prob[batch_index][trace_length_index] * \
                                     target_label[batch_index][trace_length_index]
                        ll = np.log(likelihood[target_prediction] + 1e-10)
                    ll_sum += ll
                if output_prediction == target_prediction:
                    correct_number += 1
                    if correct_output_all.get(output_prediction) is None:
                        correct_output_all.update({output_prediction: 1})
                    else:
                        number = correct_output_all.get(output_prediction) + 1
                        correct_output_all.update({output_prediction: number})

    if if_print:
        print(correct_output_all)
    if if_add_ll:
        return float(correct_number) / total_number, float(ll_sum) / total_number
    else:
        return float(correct_number) / total_number


def compute_mae(target_actions_prob, output_actions_prob, if_print=False):
    total_number = 0
    total_mae = 0
    for batch_index in range(0, len(target_actions_prob)):
        total_number += 1
        mae = abs(output_actions_prob[batch_index] - target_actions_prob[batch_index])
        # print mae
        total_mae += mae
    print('prediction scale is :' + str(np.sum(output_actions_prob, axis=0)))
    return total_mae / float(total_number)


def compute_acc(target_label, output_prob,
                bounding_id_trace_length=None,
                keep_flags_all=None,
                if_print=False,
                if_binary_result=False,
                if_add_ll=False):
    total_number = 0
    correct_number = 0
    ll_sum = 0
    correct_output_all = {}

    if if_binary_result:
        TP = 0
        TN = 0
        FP = 0
        FN = 0

    for batch_index in range(0, len(target_label)):
        if keep_flags_all is not None:
            if not keep_flags_all[batch_index]:
                continue
        total_number += 1
        sub_output_prob_bound = None
        sub_target_prob_bound = None
        if bounding_id_trace_length is not None and batch_index - bounding_id_trace_length - 1 > 0:  # TODO: any other ideas?
            bounding_ids = set()
            # if batch_index - bounding_id_trace_length - 1 > 0:
            bounding_end_index = batch_index - bounding_id_trace_length - 1
            for i in range(batch_index, bounding_end_index, -1):
                bounding_ids.add(np.argmax(target_label[i]))
            bounding_ids_index = sorted(list(bounding_ids), reverse=True)
            # temp = output_prob[batch_index][bounding_ids_index]
            sub_output_prob_bound = output_prob[batch_index][bounding_ids_index]
            sub_output_prob_bound = np.exp(sub_output_prob_bound) / sum(np.exp(sub_output_prob_bound))
            sub_target_prob_bound = target_label[batch_index][bounding_ids_index]
            target_prediction_bound =  np.argmax(sub_target_prob_bound)
            bounding_output_prediction_index = np.argmax(sub_output_prob_bound)
            output_prediction = bounding_ids_index[bounding_output_prediction_index]
        else:
            output_prediction = np.argmax(output_prob[batch_index])

        target_prediction = np.argmax(target_label[batch_index])

        if if_binary_result:
            if target_prediction == 0 and output_prediction == 0:  # argmax(prob)=0 indicates score
                TP += 1
            elif target_prediction == 1 and output_prediction == 1:
                TN += 1
            elif target_prediction == 0 and output_prediction == 1:
                FN += 1
            elif target_prediction == 1 and output_prediction == 0:
                FP += 1
            else:
                raise ValueError('It is not binary result')
        if if_add_ll:
            if sub_output_prob_bound is not None:
                likelihood = sub_output_prob_bound * sub_target_prob_bound
                ll = np.log(likelihood[target_prediction_bound] + 1e-10)
            else:
                likelihood = output_prob[batch_index] * target_label[batch_index]
                ll = np.log(likelihood[target_prediction] + 1e-10)
            ll_sum += ll

        if output_prediction == target_prediction:
            correct_number += 1
            if correct_output_all.get(output_prediction) is None:
                correct_output_all.update({output_prediction: 1})
            else:
                number = correct_output_all.get(output_prediction) + 1
                correct_output_all.update({output_prediction: number})

    if if_print:
        print(correct_output_all)

    if if_binary_result:
        auc_score = roc_auc_score(y_true=target_label[:, 0], y_score=output_prob[:, 0])
        if if_add_ll:
            return TP, TN, FP, FN, float(correct_number) / total_number, float(ll_sum) / total_number, auc_score
        else:
            return TP, TN, FP, FN, float(correct_number) / total_number, auc_score
    else:
        if if_add_ll:
            return float(correct_number) / total_number, float(ll_sum) / total_number
        else:
            return float(correct_number) / total_number


def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu) ** 2
    value = (1 / tf.sqrt(2 * np.pi * (var ** 2))) * tf.exp((-1 / (2 * (var ** 2))) * value)
    return value


def normal_td(mu1, mu2, var1, var2, y):
    """compute td error between two normal distribution"""
    mu_diff = (mu2 - mu1)
    var_diff = (var1 ** 2 + var2 ** 2) ** 0.5
    # https://stats.stackexchange.com/questions/186463/distribution-of-difference-between-two-normal-distributions
    com1 = (var_diff ** -1) * ((2 / np.pi) ** 0.5)
    com2 = tf.cosh(y * mu_diff / (var_diff ** 2))
    com3 = tf.exp(-1 * (y ** 2 + mu_diff ** 2) / (2 * var_diff ** 2))
    # return com1, com2, com3
    return com1 * com2 * com3


def get_embedding_model_output(sess_nn, output_list, feed_dict):
    output = sess_nn.run(output_list,
                         feed_dict=feed_dict)

    return output


def prepare_embedding_game_data(data_store,
                                dir_game, config,
                                player_id_cluster_dir,
                                model_category,
                                player_basic_info_dir,
                                game_date_dir,
                                player_box_score_dir):

    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config,
        player_id_cluster_dir=player_id_cluster_dir,
        player_basic_info_dir=player_basic_info_dir,
        game_date_dir=game_date_dir,
        player_box_score_dir=player_box_score_dir
    )

    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    reward_seq = transfer2seq(data=np.expand_dims(reward, axis=-1), trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)

    state_reward_input = np.concatenate([reward_seq, state_input],
                                        axis=-1)  # concatenate the sequence of state and reward.
    state_zero_input = []
    for trace_index in range(0, len(state_trace_length)):
        trace_length = state_trace_length[trace_index]
        trace_length = trace_length - 1
        if trace_length > 9:
            trace_length = 9
        state_zero_input.append(state_input[trace_index, trace_length, :])

    train_mask = np.asarray([[[0]] * config.Learn.max_seq_length] * len(player_index))
    input_data = np.concatenate([player_index_seq,
                                 team_id_seq,
                                 state_input,
                                 action_seq,
                                 train_mask],
                                axis=2)

    trace_lengths = []
    for trace_length in state_trace_length:
        trace_length = trace_length if trace_length <= config.Learn.max_seq_length else config.Learn.max_seq_length
        trace_lengths.append(trace_length)
    trace_lengths = np.asarray(trace_lengths)
    selection_matrix = generate_selection_matrix(trace_lengths,
                                                 max_trace_length=config.Learn.max_seq_length)
    target_model_label = player_index_seq

    return input_data, target_model_label, trace_lengths, train_mask, selection_matrix


def compute_game_embedding(sess_nn, model, data_store,
                           dir_game, config,
                           player_id_cluster_dir,
                           model_category,
                           player_basic_info_dir=None,
                           game_date_dir=None,
                           player_box_score_dir=None,
                           focus_condition=None
                           ):
    input_data, target_model_label, \
    trace_lengths, train_mask, \
    selection_matrix = prepare_embedding_game_data(data_store,
                                                   dir_game, config,
                                                   player_id_cluster_dir,
                                                   model_category,
                                                   player_basic_info_dir,
                                                   game_date_dir,
                                                   player_box_score_dir,
                                                   focus_condition)

    player_index = np.argmax(sio.loadmat(data_store + "/" + dir_game + "/"
                                         + 'player_index_game_{0}-playsequence-wpoi.mat'.
                                         format(str(dir_game)))['player_index'], axis=1)

    if model_category == "cvrnn" or model_category == "varlea" or model_category == "caernn":
        feed_dict = {model.input_data_ph: input_data,
                     model.trace_length_ph: trace_lengths,
                     model.selection_matrix_ph: selection_matrix
                     }
        [output_encoder] = get_embedding_model_output(sess_nn, [model.z_encoder_output], feed_dict)
        output_player_encoding = output_encoder
    elif model_category == "multi_agent":
        feed_dict={model.positive_state_input_ph:input_data,
                   model.positive_trace_lengths_ph: trace_lengths}
        [output_encoder] = get_embedding_model_output(sess_nn, [model.episode_embedding], feed_dict)
        output_player_encoding = output_encoder
    elif model_category == "clvrnn-prior":
        feed_dict = {model.input_data_ph: input_data,
                     model.trace_length_ph: trace_lengths,
                     model.selection_matrix_ph: selection_matrix
                     }
        [output_encoder] = get_embedding_model_output(sess_nn, [model.z_prior_output], feed_dict)
        output_player_encoding = output_encoder

    elif model_category == "cvae" or model_category == "vhe":
        if config.Learn.apply_lstm:
            x_ph_input = []
            for trace_index in range(0, len(trace_lengths)):
                trace_length = trace_lengths[trace_index]
                trace_length = trace_length - 1
                if trace_length > 9:
                    trace_length = 9
                x_ph_input.append(input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
            x_ph_input = np.asarray(x_ph_input)

            feed_dict = {model.x_ph: x_ph_input,
                         model.y_ph: input_data[:, :, config.Arch.CVAE.x_dim:],
                         model.trace_lengths_ph: trace_lengths,
                         model.train_flag_ph: train_mask, }
        else:
            feed_dict = {model.x_ph: input_data[:, : config.Arch.CVAE.x_dim],
                         model.train_flag_ph: train_mask,
                         model.y_ph: input_data[:, config.Arch.CVAE.x_dim:]}
        [output_encoder] = get_embedding_model_output(sess_nn, [model.z], feed_dict)
        output_player_encoding = output_encoder

    elif model_category == "encoder":
        if config.Learn.apply_lstm:
            feed_dict = {model.input_ph: input_data[:, :, config.Arch.Encoder.output_dim:],
                         model.trace_lengths_ph: trace_lengths}
        else:
            feed_dict = {model.input_ph: input_data[:, config.Arch.Encoder.output_dim:], }
        [output_encoder] = get_embedding_model_output(sess_nn, [model.embedding], feed_dict)
        output_player_encoding = output_encoder

    # elif model_category == "lstm_prediction":
    #     feed_dict = {model.rnn_input_ph: input_data,
    #                  model.trace_lengths_ph: trace_lengths}
    #     [output_encoder] = get_embedding_model_output(sess_nn, [model.read_out], feed_dict)
    #     output_player_encoding = output_encoder
    else:
        raise ValueError('Unknown category {0}'.format(model_category))

    return output_player_encoding, player_index


def compute_game_ids(sess_nn, model, data_store,
                     dir_game, config,
                     player_id_cluster_dir,
                     model_category,
                     player_basic_info_dir=None,
                     game_date_dir=None,
                     player_box_score_dir=None
                     ):
    input_data, target_model_label, \
    trace_lengths, train_mask, \
    selection_matrix = prepare_embedding_game_data(data_store,
                                                   dir_game, config,
                                                   player_id_cluster_dir,
                                                   model_category,
                                                   player_basic_info_dir,
                                                   game_date_dir,
                                                   player_box_score_dir)
    feed_dict = {model.input_data_ph: input_data,
                 model.trace_length_ph: trace_lengths,
                 model.selection_matrix_ph: selection_matrix
                 }
    model_kl = [model.kl_loss]
    [kl_loss_output] = get_embedding_model_output(sess_nn, model_kl, feed_dict)
    kl_loss = []

    model_output = [model.output]
        # if model_category == "cvrnn" else [model.prior_recon_output]
    [output_x] = get_embedding_model_output(sess_nn, model_output, feed_dict)

    output_model_prob = []
    for batch_index in range(0, len(output_x)):
        output_decoder_batch = []
        for trace_length_index in range(0, config.Learn.max_seq_length):
            if selection_matrix[batch_index][trace_length_index]:
                output_decoder_batch.append(output_x[batch_index][trace_length_index])
                kl_loss.append(kl_loss_output[batch_index][trace_length_index])
            else:
                if model_category == "cvrnn":
                    output_decoder_batch.append(np.asarray([0] * config.Arch.CVRNN.x_dim))
                elif model_category == "varlea":
                    output_decoder_batch.append(np.asarray([0] * config.Arch.CLVRNN.x_dim))
        output_model_prob.append(output_decoder_batch)
    output_model_prob = np.asarray(output_model_prob)

    return output_model_prob, target_model_label, selection_matrix, np.mean(kl_loss)



def get_model_prediction_output(sess_nn, model, config, pred_input_data, pred_trace_lengths, model_category):
    if model_category == "cvrnn" or model_category == "varlea" or model_category == "caernn":
        pred_selection_matrix = generate_selection_matrix(pred_trace_lengths,
                                                          max_trace_length=config.Learn.max_seq_length)
        [readout_pred_output] = sess_nn.run([
            model.action_pred_output],
            feed_dict={model.input_data_ph: pred_input_data,
                       model.trace_length_ph: pred_trace_lengths,
                       model.selection_matrix_ph: pred_selection_matrix
                       })
    elif model_category == 'cvae' or model_category == 'vhe':
        pred_train_mask = np.asarray([0] * len(pred_input_data))
        if config.Learn.apply_lstm:
            x_ph_input = []
            for trace_index in range(0, len(pred_trace_lengths)):
                trace_length = pred_trace_lengths[trace_index]
                trace_length = trace_length - 1
                if trace_length > 9:
                    trace_length = 9
                x_ph_input.append(pred_input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
            x_ph_input = np.asarray(x_ph_input)

            feed_dict = {model.x_ph: x_ph_input,
                         model.y_ph: pred_input_data[:, :, config.Arch.CVAE.x_dim:],
                         model.trace_lengths_ph: pred_trace_lengths,
                         model.train_flag_ph: pred_train_mask,
                         }
        else:
            feed_dict = {model.x_ph: pred_input_data[:, : config.Arch.CVAE.x_dim],
                         model.train_flag_ph: pred_train_mask,
                         model.y_ph: pred_input_data[:, config.Arch.CVAE.x_dim:]
                         }

        [
            readout_pred_output,
        ] = sess_nn.run([
            model.predict_output],
            feed_dict=feed_dict
        )

    elif model_category == 'lstm_prediction':
        [readout_pred_output] = sess_nn.run([model.read_out],
                                            feed_dict={model.rnn_input_ph: pred_input_data,
                                                       model.trace_lengths_ph: pred_trace_lengths})
    elif model_category == 'encoder':
        if config.Learn.apply_lstm:
            feed_dict = {model.input_ph: pred_input_data[:, :, config.Arch.Encoder.output_dim:],
                         model.trace_lengths_ph: pred_trace_lengths
                         }
        else:
            feed_dict = {model.input_ph: pred_input_data[:, config.Arch.Encoder.output_dim:]}
        [
            readout_pred_output,
        ] = sess_nn.run([
            model.prediction_prob],
            feed_dict=feed_dict
        )
    elif model_category == 'multi_agent':
        feed_dict = {
            model.positive_state_input_ph: pred_input_data,
            model.positive_trace_lengths_ph: pred_trace_lengths}
        [
            readout_pred_output,
        ] = sess_nn.run([
            model.predict_output],
            feed_dict=feed_dict
        )

    return readout_pred_output


def compute_prediction_game(sess_nn, model,
                            data_store,
                            source_data_dir,
                            dir_game, config,
                            player_id_cluster_dir,
                            model_category,
                            prediction_target,
                            player_basic_info_dir=None,
                            game_date_dir=None,
                            player_box_score_dir=None
                            ):
    state_trace_length, state_input, reward, actions, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game,
        config=config,
        player_id_cluster_dir=player_id_cluster_dir,
        player_basic_info_dir=player_basic_info_dir,
        game_date_dir=game_date_dir,
        player_box_score_dir=player_box_score_dir)

    trace_lengths = []
    for trace_length in state_trace_length:
        trace_length = trace_length if trace_length <= config.Learn.max_seq_length else config.Learn.max_seq_length
        trace_lengths.append(trace_length)
    state_trace_length = np.asarray(trace_lengths)

    action_seq = transfer2seq(data=actions, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)

    state_zero_trace = [1] * len(state_trace_length)
    state_zero_input = []
    for trace_index in range(0, len(state_trace_length)):
        trace_length = state_trace_length[trace_index]
        trace_length = trace_length - 1
        if trace_length > 9:
            trace_length = 9
        state_zero_input.append(state_input[trace_index, trace_length, :])
    state_zero_input = np.asarray(state_zero_input)

    # if config.Arch.Predict.predict_target == 'ActionGoal':
    if prediction_target == 'ActionGoal':
        actions_all = read_feature_within_events(directory=dir_game,
                                                 data_path=source_data_dir,
                                                 feature_name='name')
        next_goal_label = []
        data_length = state_trace_length.shape[0]
        new_reward = []
        new_action_seq = []
        new_state_input = []
        new_state_trace_length = []
        new_team_id_seq = []
        new_player_index_seq = []
        new_action = []
        new_state_zero_input = []
        new_state_zero_trace = []
        new_state_zero_trace = []
        new_team_id = []
        new_player_index = []
        for action_index in range(0, data_length):
            action = actions_all[action_index]
            if 'shot' in action:
                if action_index + 1 == data_length:
                    continue
                new_reward.append(reward[action_index])
                new_action.append(actions[action_index])
                new_state_zero_input.append(state_zero_input[action_index])
                new_state_zero_trace.append(state_zero_trace[action_index])
                new_team_id.append(team_id[action_index])
                new_player_index.append(player_index[action_index])
                new_action_seq.append(action_seq[action_index])
                new_state_input.append(state_input[action_index])
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
    elif prediction_target == 'Action':
        add_pred_flag = True
        pred_target = actions[1:, :]
        new_reward = reward[:-1]
        new_action_seq = action_seq[:-1, :, :]
        new_state_input = state_input[:-1, :, :]
        new_state_trace_length = state_trace_length[:-1]
        new_team_id_seq = team_id_seq[:-1, :, :]
    else:
        # raise ValueError()
        add_pred_flag = False

    if model_category == "cvrnn" or model_category == 'varlea':
        train_mask = np.asarray([[[1]] * config.Learn.max_seq_length] * len(new_state_input))
        if config.Learn.predict_target == 'PlayerLocalId':
            pred_input_data = np.concatenate([np.asarray(new_player_index_seq),
                                              np.asarray(new_team_id_seq),
                                              np.asarray(new_state_input),
                                              np.asarray(new_action_seq),
                                              train_mask], axis=2)
            pred_target_data = np.asarray(np.asarray(pred_target))
            pred_trace_lengths = new_state_trace_length
        else:
            pred_input_data = np.concatenate([np.asarray(new_player_index_seq), np.asarray(new_state_input),
                                              np.asarray(new_action_seq), train_mask], axis=2)
            pred_target_data = np.asarray(np.asarray(pred_target))
            pred_trace_lengths = new_state_trace_length

        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)
    elif model_category == "caernn":
        pred_input_data = np.concatenate([np.asarray(new_player_index_seq),
                                          np.asarray(new_team_id_seq),
                                          np.asarray(new_state_input),
                                          np.asarray(new_action_seq),
                                          ], axis=2)
        pred_target_data = np.asarray(np.asarray(pred_target))
        pred_trace_lengths = new_state_trace_length

        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)
    elif model_category == 'cvae' or model_category == 'vhe':

        if config.Learn.apply_lstm:
            pred_input_data = np.concatenate([np.asarray(new_player_index_seq),
                                              np.asarray(new_team_id_seq),
                                              np.asarray(new_state_input),
                                              np.asarray(new_action_seq)], axis=2)
            pred_target_data = np.asarray(pred_target)
            pred_trace_lengths = new_state_trace_length
        else:

            pred_input_data = np.concatenate([np.asarray(new_player_index),
                                              np.asarray(new_team_id),
                                              np.asarray(new_state_zero_input),
                                              np.asarray(new_action), ], axis=1)
            pred_target_data = np.asarray(pred_target)
            pred_trace_lengths = new_state_zero_trace
        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)
    elif model_category == 'lstm_prediction':

        if config.Learn.apply_pid:
            pred_input_data = np.concatenate([np.asarray(new_action_seq),
                                              np.asarray(new_state_input),
                                              np.asarray(new_player_index_seq)],
                                             axis=2)
        else:
            pred_input_data = np.concatenate([np.asarray(new_action_seq),
                                              np.asarray(new_state_input)],
                                             axis=2)
        pred_trace_lengths = new_state_trace_length
        pred_target_data = pred_target
        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)
    elif model_category == 'encoder':
        if config.Learn.apply_lstm:
            pred_input_data = np.concatenate([np.asarray(new_player_index_seq),
                                              np.asarray(new_team_id_seq),
                                              np.asarray(new_state_input),
                                              np.asarray(new_action_seq)], axis=2)
            pred_target_data = np.asarray(pred_target)
            pred_trace_lengths = new_state_trace_length
        else:
            pred_input_data = np.concatenate([np.asarray(new_player_index),
                                              np.asarray(new_team_id),
                                              np.asarray(new_state_zero_input),
                                              np.asarray(new_action)], axis=1)
            pred_target_data = np.asarray(pred_target)
            pred_trace_lengths = new_state_zero_trace

        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)

    elif model_category == 'multi_agent':
        if config.Learn.apply_pid:
            pred_input_data = np.concatenate([np.asarray(new_state_input),
                                              np.asarray(new_action_seq),
                                              np.asarray(new_player_index_seq)
                                              ], axis=2)
        else:
            pred_input_data = np.concatenate([np.asarray(new_state_input),
                                              np.asarray(new_action_seq)], axis=2)
        pred_target_data = np.asarray(pred_target)
        pred_trace_lengths = new_state_trace_length
        readout_pred_output = get_model_prediction_output(sess_nn, model, config, pred_input_data,
                                                          pred_trace_lengths, model_category)

    else:
        raise ValueError('unknown model {0}'.format(model_category))

    return readout_pred_output, pred_target_data


def compute_game_values(sess_nn, model, data_store, dir_game, config,
                        player_id_cluster_dir,
                        model_category
                        ):
    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)

    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)

    trace_lengths = []
    for trace_length in state_trace_length:
        trace_length = trace_length if trace_length <= config.Learn.max_seq_length else config.Learn.max_seq_length
        trace_lengths.append(trace_length)
    state_trace_length = np.asarray(trace_lengths)

    if model_category == "cvrnn" or model_category == "varlea":
        train_mask = np.asarray([[[1]] * config.Learn.max_seq_length] * len(player_index))
        if config.Learn.predict_target == 'PlayerLocalId':
            input_data = np.concatenate([player_index_seq,
                                         team_id_seq,
                                         state_input,
                                         action_seq,
                                         train_mask],
                                        axis=2)
            trace_lengths = state_trace_length
            selection_matrix_t0 = generate_selection_matrix(trace_lengths,
                                                            max_trace_length=config.Learn.max_seq_length)
        else:
            input_data = np.concatenate([player_index, state_input,
                                         action, train_mask], axis=2)
            trace_lengths = state_trace_length
            selection_matrix_t0 = generate_selection_matrix(trace_lengths,
                                                            max_trace_length=config.Learn.max_seq_length)
        if model_category == "cvrnn":
            [readout_next_Q, readout_accumu_Q] = sess_nn.run([model.sarsa_output,
                                                              model.diff_output],
                                                             feed_dict={model.input_data_ph: input_data,
                                                                        model.trace_length_ph: trace_lengths,
                                                                        model.selection_matrix_ph: selection_matrix_t0
                                                                        })
        elif model_category == 'varlea':
            [readout_accumu_Q] = sess_nn.run([model.diff_output],
                                             feed_dict={model.input_data_ph: input_data,
                                                        model.trace_length_ph: trace_lengths,
                                                        model.selection_matrix_ph: selection_matrix_t0
                                                        })
            readout_next_Q = None
    elif model_category == "caernn":

        input_data = np.concatenate([player_index_seq,
                                     team_id_seq,
                                     state_input,
                                     action_seq],
                                    axis=2)
        trace_lengths = state_trace_length
        selection_matrix_t0 = generate_selection_matrix(trace_lengths, max_trace_length=config.Learn.max_seq_length)

        [readout_accumu_Q] = sess_nn.run([model.diff_output],
                                                         feed_dict={model.input_data_ph: input_data,
                                                                    model.trace_length_ph: trace_lengths,
                                                                    model.selection_matrix_ph: selection_matrix_t0
                                                                    })
        readout_next_Q = None
    elif model_category == 'cvae' or model_category == 'vhe':
        pred_train_mask = np.asarray([0] * len(player_index_seq))
        # if config.Learn.apply_lstm:
        input_data = np.concatenate([np.asarray(player_index_seq),
                                     np.asarray(team_id_seq),
                                     np.asarray(state_input),
                                     np.asarray(action_seq)], axis=2)
        trace_lengths = state_trace_length

        x_ph_input = []
        for trace_index in range(0, len(state_trace_length)):
            trace_length = state_trace_length[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)

        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: input_data[:, :, config.Arch.CVAE.x_dim:],
                     model.trace_lengths_ph: trace_lengths,
                     model.train_flag_ph: pred_train_mask,
                     }
        # else:
        #     input_data = np.concatenate([np.asarray(player_index_seq),
        #                                  np.asarray(team_id_seq),
        #                                  np.asarray(state_input),
        #                                  np.asarray(action_seq), ], axis=1)
        #
        #     feed_dict = {model.x_ph: input_data[:, : config.Arch.CVAE.x_dim],
        #                  model.train_flag_ph: pred_train_mask,
        #                  model.y_ph: input_data[:, config.Arch.CVAE.x_dim:]
        #                  }

        [readout_next_Q, readout_accumu_Q] = sess_nn.run([model.q_values_sarsa,
                                                          model.q_values_diff],
                                                         feed_dict=feed_dict)

    elif model_category == 'encoder':
        # if config.Learn.apply_lstm:
        pred_input_data = np.concatenate([np.asarray(player_index_seq),
                                          np.asarray(team_id_seq),
                                          np.asarray(state_input),
                                          np.asarray(action_seq)], axis=2)
        pred_trace_lengths = state_trace_length

        feed_dict = {model.input_ph: pred_input_data[:, :, config.Arch.Encoder.output_dim:],
                     model.trace_lengths_ph: pred_trace_lengths
                     }
        [readout_next_Q, readout_accumu_Q] = sess_nn.run([model.q_values_sarsa,
                                                          model.q_values_diff],
                                                         feed_dict=feed_dict)
        # else:
        #     pred_input_data = np.concatenate([np.asarray(player_index),
        #                                       np.asarray(team_id),
        #                                       np.asarray(state_zero_input),
        #                                       np.asarray(action)], axis=1)
        #     pred_target_data = np.asarray(np.asarray(pred_target))
        #     pred_trace_lengths = state_zero_trace
    elif model_category == 'multi_agent':
        if config.Learn.apply_pid:
            pred_input_data = np.concatenate([np.asarray(state_input),
                                              np.asarray(action_seq),
                                              np.asarray(player_index_seq)], axis=2)
        else:
            pred_input_data = np.concatenate([np.asarray(state_input),
                                              np.asarray(action_seq)], axis=2)

        pred_trace_lengths = state_trace_length
        feed_dict = {model.positive_state_input_ph: pred_input_data,
                     model.positive_trace_lengths_ph: pred_trace_lengths}
        [readout_accumu_Q] = sess_nn.run([model.score_diff_output], feed_dict=feed_dict)
        readout_next_Q = None


    elif model_category == 'lstm_Qs':

        if config.Learn.apply_pid:
            input_data = np.concatenate([np.asarray(player_index_seq),
                                         np.asarray(state_input),
                                         np.asarray(action_seq)], axis=2)
        else:
            input_data = np.concatenate([np.asarray(state_input),
                                         np.asarray(action_seq)], axis=2)
        trace_lengths = state_trace_length
        [readout_next_Q] = sess_nn.run([model.read_out],
                                       feed_dict={model.rnn_input_ph: input_data,
                                                  model.trace_lengths_ph: trace_lengths})
        readout_accumu_Q = None
    elif model_category == 'lstm_diff':
        if config.Learn.apply_pid:
            input_data = np.concatenate([np.asarray(player_index_seq),
                                         np.asarray(state_input), np.asarray(action_seq)], axis=2)
        else:
            input_data = np.concatenate([np.asarray(state_input), np.asarray(action_seq)], axis=2)
        trace_lengths = state_trace_length
        [readout_accumu_Q] = sess_nn.run([model.read_out],
                                         feed_dict={model.rnn_input_ph: input_data,
                                                    model.trace_lengths_ph: trace_lengths})
        readout_next_Q = None
    else:
        raise ValueError('unknown model {0}'.format(model_category))

    return readout_next_Q, readout_accumu_Q


def validate_games_prediction(config,
                              data_store_dir,
                              source_data_dir,
                              dir_all,
                              model_nn,
                              sess_nn,
                              model_path,
                              player_basic_info_dir=None,
                              game_date_dir=None,
                              player_box_score_dir=None,
                              prediction_type='gamebygame',
                              model_number=None,
                              player_id_cluster_dir=None,
                              saved_network_dir=None,
                              model_category=None,
                              file_writer=None):
    # sess_nn = tf.InteractiveSession()
    if model_category == 'cvrnn':
        prediction_target = config.Arch.Predict.predict_target
    #     cvrnn = CVRNN(config=config, extra_prediction_flag=True)
    #     cvrnn()
    #     model_nn = cvrnn
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    if model_category == 'cvae' or model_category == 'vhe':
        prediction_target = config.Arch.Predict.predict_target
    #     cvae = CVAE_NN(config=config)
    #     cvae()
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_nn = cvae
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    elif model_category == 'lstm_prediction':
        prediction_target = config.Learn.predict_target
    #     model_nn = Td_Prediction_NN(config=config)
    #     model_nn.initialize_ph()
    #     model_nn.build()
    #     model_nn.call()
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    elif model_category == 'encoder':
        prediction_target = config.Arch.Predict.predict_target
    #     encoder = Encoder_NN(config=config)
    #     encoder()
    #     model_nn = encoder
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    elif model_category == 'multi_agent':
        prediction_target = config.Arch.Predict.predict_target
    elif model_category == 'varlea':
        prediction_target = config.Arch.Predict.predict_target
    elif model_category == 'caernn':
        prediction_target = config.Arch.Predict.predict_target

    if model_number is not None:
        saver = tf.train.Saver()
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        raise ValueError('please provide a model number or no model will be loaded')
    output_decoder_all = None
    target_data_all = None

    if prediction_type == 'gamebygame':
        # for game_name_dir in dir_all[:5]:
        for game_name_dir in dir_all:
            print('working for game {0}'.format(game_name_dir))
            game_name = game_name_dir.split('.')[0]
            # game_time_all = get_game_time(data_path, game_name_dir)
            output_prediction_prob, \
            target_prediction, = compute_prediction_game(sess_nn=sess_nn,
                                                         model=model_nn,
                                                         data_store=data_store_dir,
                                                         source_data_dir=source_data_dir,
                                                         dir_game=game_name,
                                                         config=config,
                                                         player_id_cluster_dir=player_id_cluster_dir,
                                                         model_category=model_category,
                                                         prediction_target=prediction_target,
                                                         player_basic_info_dir=player_basic_info_dir,
                                                         game_date_dir=game_date_dir,
                                                         player_box_score_dir=player_box_score_dir)
            if output_decoder_all is None:
                output_decoder_all = output_prediction_prob
                target_data_all = target_prediction
            else:
                # try:
                output_decoder_all = np.concatenate([output_decoder_all, output_prediction_prob], axis=0)
                target_data_all = np.concatenate([target_data_all, target_prediction], axis=0)
        TP, TN, FP, FN, acc, ll, auc = compute_acc(output_prob=output_decoder_all,
                                                   target_label=target_data_all,
                                                   if_print=True,
                                                   if_binary_result=True,
                                                   if_add_ll=True)
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        prediction_results_str = "prediction testing precision is {6}, recall is {7}, f1 is {8}, acc is {0}, ll is {5} " \
                                 "with TP:{1}, TN:{2}, FP:{3}, FN:{4}\n".format(str(acc),
                                                                                str(TP),
                                                                                str(TN),
                                                                                str(FP),
                                                                                str(FN),
                                                                                str(ll),
                                                                                str(precision),
                                                                                str(recall),
                                                                                str(f1)
                                                                                )

        print (prediction_results_str)
        if file_writer is not None:
            file_writer.write(prediction_results_str)
        return precision, recall, f1, acc, ll, auc

    elif prediction_type == 'spatial_simulation':

        spatial_projection_input_data = sio.loadmat(data_store_dir + '_input.mat')['simulate_data']
        spatial_projection_trace_data = sio.loadmat(data_store_dir + '_trace.mat')['simulate_data']
        all_prediction_output = []

        for x_index in range(0, spatial_projection_input_data.shape[0]):

            if model_category == 'cvrnn':
                pass

            x_coord_data_ouput = get_model_prediction_output(sess_nn=sess_nn, model=model_nn,
                                                             config=config,
                                                             pred_input_data=spatial_projection_input_data[x_index],
                                                             pred_trace_lengths=spatial_projection_trace_data[x_index],
                                                             model_category=model_category)
            all_prediction_output.append(x_coord_data_ouput)

        return np.asarray(all_prediction_output)


def validate_games_embedding(config,
                             data_store_dir,
                             dir_all,
                             player_basic_info_dir,
                             game_date_dir,
                             player_box_score_dir,
                             model_number=None,
                             player_id_cluster_dir=None,
                             saved_network_dir=None,
                             model_category=None,
                             source_data_path=None,
                             focus_condition=None
                             ):
    sess_nn = tf.InteractiveSession()
    if 'varlea' in model_category:
        varlea = VaRLEA(config=config, train_flag=False)
        varlea()
        model_nn = varlea
        sess_nn.run(tf.global_variables_initializer())
        model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    # elif model_category == 'caernn':
    #     caernn = CAERNN(config=config)
    #     caernn()
    #     model_nn = caernn
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    elif model_category == 'cvrnn':
        cvrnn = CVRNN(config=config)
        cvrnn()
        model_nn = cvrnn
        sess_nn.run(tf.global_variables_initializer())
        model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)

    # elif model_category == 'encoder':
    #     encoder = Encoder_NN(config=config)
    #     encoder()
    #     model_nn = encoder
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    # elif model_category == 'multi_agent':
    #     model_nn =  Multi_Agent_NN(config=config)
    #     model_nn.call()
    #     sess_nn.run(tf.global_variables_initializer())
    #     model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    else:
        raise ValueError('unkown model {0}'.format(model_category))
    if model_number is not None:
        saver = tf.train.Saver()
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        raise ValueError('please provide a model number or no model will be loaded')

    all_embedding = None
    all_player_index = None
    location_features_all = []
    action_features_all = []
    manpower_features_all = []
    period_feature_all = []
    for game_name_dir in dir_all:
        print('working for game {0}'.format(game_name_dir))
        game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir)
        embedding, player_index = compute_game_embedding(sess_nn=sess_nn,
                                                         model=model_nn,
                                                         data_store=data_store_dir,
                                                         dir_game=game_name,
                                                         config=config,
                                                         player_id_cluster_dir=player_id_cluster_dir,
                                                         model_category=model_category,
                                                         player_basic_info_dir=player_basic_info_dir,
                                                         game_date_dir=game_date_dir,
                                                         player_box_score_dir=player_box_score_dir,
                                                         focus_condition=focus_condition
                                                         )

        location_features_events = read_features_within_events(directory=str(game_name_dir) + '-playsequence-wpoi.json',
                                                   data_path=source_data_path,
                                                   feature_name_list=['xAdjCoord','yAdjCoord'])
        location_features_all += location_features_events[:len(embedding)]

        action_features_events = read_features_within_events(directory=str(game_name_dir) + '-playsequence-wpoi.json',
                                                   data_path=source_data_path,
                                                   feature_name_list=['name'])
        action_features_all += action_features_events[:len(embedding)]

        manpower_features_events = read_features_within_events(directory=str(game_name_dir) + '-playsequence-wpoi.json',
                                                   data_path=source_data_path,
                                                   feature_name_list=['manpowerSituation'])
        manpower_features_all += manpower_features_events[:len(embedding)]

        period_features_events = read_features_within_events(directory=str(game_name_dir) + '-playsequence-wpoi.json',
                                                   data_path=source_data_path,
                                                   feature_name_list=['period'])
        period_feature_all += period_features_events[:len(embedding)]


        if all_embedding is None:
            all_embedding = embedding
            all_player_index = player_index
        else:
            all_embedding = np.concatenate([all_embedding, embedding], axis=0)
            all_player_index = np.concatenate([all_player_index, player_index], axis=0)
        # all_player_index.append(player_index)
        # all_embedding.append(embedding)
        # print(np.sum(embedding - np.flip(embedding, axis=0)))
        # print(np.mean(embedding - np.flip(embedding, axis=0)))
    tf.reset_default_graph()
    sess_nn.close()
    return all_embedding, all_player_index, \
           location_features_all, action_features_all, \
           manpower_features_all, period_feature_all


def validate_model_initialization(sess_nn,
                                  model_category,
                                  config,
                                  ):
    if model_category == 'cvrnn':
        cvrnn = CVRNN(config=config, extra_prediction_flag=True)
        cvrnn()
        model_nn = cvrnn
        sess_nn.run(tf.global_variables_initializer())
    if model_category == 'varlea':
        varlea = VaRLEA(config=config, extra_prediction_flag=True, train_flag=False)
        varlea()
        model_nn = varlea
        sess_nn.run(tf.global_variables_initializer())

    return model_nn


def validate_games_player_id(config,
                             data_store_dir,
                             dir_all,
                             model_nn,
                             sess_nn,
                             model_path,
                             player_basic_info_dir,
                             game_date_dir,
                             player_box_score_dir,
                             data_store,
                             apply_bounding=False,
                             model_number=None,
                             player_id_cluster_dir=None,
                             saved_network_dir=None,
                             model_category=None,
                             file_writer=None,
                             player_sparse_presence_all=None):
    if model_number is not None:
        saver = tf.train.Saver()
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        raise ValueError('please provide a model number or no model will be loaded')
    # return 0, 0
    output_decoder_all = None
    target_data_all = None
    selection_matrix_all = None
    kl_loss_all = []
    keep_flags_all = []
    for dir_game in dir_all:
    # for dir_game in ['16037']:
        print('working for game {0}'.format(dir_game))
        # game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir

        output_decoder, \
        player_index_seq, \
        selection_matrix, kl_loss = compute_game_ids(sess_nn=sess_nn,
                                                     model=model_nn,
                                                     data_store=data_store_dir,
                                                     dir_game=dir_game,
                                                     config=config,
                                                     player_id_cluster_dir=player_id_cluster_dir,
                                                     model_category=model_category,
                                                     player_basic_info_dir=player_basic_info_dir,
                                                     game_date_dir=game_date_dir,
                                                     player_box_score_dir=player_box_score_dir)
        skip_flags = [True]*len(output_decoder)
        if player_sparse_presence_all is not None:
            player_index_name = 'player_index_game_{0}-playsequence-wpoi.mat'.format(dir_game)
            player_index_all = sio.loadmat(data_store + "/" + dir_game + "/" + player_index_name)['player_index']
            for i in range(len(player_index_all)):
                player_index = np.argmax(player_index_all[i])
                if player_index not in player_sparse_presence_all:
                    skip_flags[i]= False
                else:
                    print 'find player'
        keep_flags_all += skip_flags

        if kl_loss is not None:
            kl_loss_all.append(kl_loss)
        if output_decoder_all is None:
            output_decoder_all = output_decoder
            target_data_all = player_index_seq
            if selection_matrix is not None:
                selection_matrix_all = selection_matrix
        else:
            # try:
            output_decoder_all = np.concatenate([output_decoder_all, output_decoder], axis=0)
            target_data_all = np.concatenate([target_data_all, player_index_seq], axis=0)
            if selection_matrix is not None:
                selection_matrix_all = np.concatenate([selection_matrix_all, selection_matrix], axis=0)

    if apply_bounding:
        bounding_id_trace_length = 10
    else:
        bounding_id_trace_length = None

    if selection_matrix_all is not None:
        acc, ll = compute_rnn_acc(output_prob=output_decoder_all, target_label=target_data_all,
                                  selection_matrix=selection_matrix_all, config=config,
                                  bounding_id_trace_length=bounding_id_trace_length,
                                  keep_flags_all=keep_flags_all, if_print=True,
                                  if_add_ll=True)
    else:
        acc, ll = compute_acc(target_label=target_data_all, output_prob=output_decoder_all,
                              bounding_id_trace_length=bounding_id_trace_length,
                              keep_flags_all=keep_flags_all, if_add_ll=True)
    if len(kl_loss_all) > 0:
        kl_loss_all_mean = np.mean(kl_loss_all)
    else:
        kl_loss_all_mean = None

    print ("testing acc is {0} with ll {1} and kl {2}".format(str(acc), str(ll), kl_loss_all_mean))
    if file_writer is not None:
        file_writer.write("testing acc is {0} with ll {1} and kl {2}\n".format(str(acc), str(ll), np.mean(kl_loss_all)))

    return acc, ll


def compute_games_Q_values(config, data_store_dir, dir_all,
                           model_nn, sess_nn, model_path,
                           model_number=None,
                           player_id_cluster_dir=None,
                           model_category=None,
                           return_values_flag=False,
                           apply_cv=False,
                           running_number=None):
    if model_number is not None:
        saver = tf.train.Saver()
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        raise ValueError('please provide a model number or no model will be loaded')

    model_next_Q_values_all = []
    model_accumu_Q_value_all = []
    for game_name_dir in dir_all:
    # for game_name_dir in ['16276']:
        print('working for game {0}'.format(game_name_dir))
        game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir)
        readout_next_Q, readout_accumu_Q = compute_game_values(sess_nn=sess_nn,
                                                               model=model_nn,
                                                               data_store=data_store_dir,
                                                               dir_game=game_name,
                                                               config=config,
                                                               player_id_cluster_dir=player_id_cluster_dir,
                                                               model_category=model_category)
        # plot_game_Q_values(model_value)
        if readout_next_Q is not None:
            model_next_Q_value_json = {}
            for value_index in range(0, len(readout_next_Q)):
                model_next_Q_value_json.update({value_index: {'home': float(readout_next_Q[value_index][0]),
                                                              'away': float(readout_next_Q[value_index][1]),
                                                              'end': float(readout_next_Q[value_index][2])}})
            model_next_Q_values_all.append(model_next_Q_value_json)
        if readout_accumu_Q is not None:
            model_accumu_Q_value_json = {}
            for value_index in range(0, len(readout_accumu_Q)):
                model_accumu_Q_value_json.update({value_index: {'home': float(readout_accumu_Q[value_index][0]),
                                                                'away': float(readout_accumu_Q[value_index][1]),
                                                                'end': float(readout_accumu_Q[value_index][2])}})
            model_accumu_Q_value_all.append(model_accumu_Q_value_json)

        data_name = get_data_name(config=config, model_catagoery=model_category, model_number=model_number)
        game_store_dir = game_name_dir.split('.')[0]
        if not return_values_flag:
            if apply_cv:
                if readout_next_Q is not None:
                    with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs')
                              + '_r' + str(running_number), 'w') as outfile:
                        json.dump(model_next_Q_value_json, outfile)
                if readout_accumu_Q is not None:
                    with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs')
                              + '_r' + str(running_number), 'w') as outfile:
                        json.dump(model_accumu_Q_value_json, outfile)
            else:
                if readout_next_Q is not None:
                    with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs'),
                              'w') as outfile:
                        json.dump(model_next_Q_value_json, outfile)
                if readout_accumu_Q is not None:
                    with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs'),
                              'w') as outfile:
                        json.dump(model_accumu_Q_value_json, outfile)

    if return_values_flag:
        return model_next_Q_values_all, model_accumu_Q_value_all
    else:
        return data_name


if __name__ == '__main__':
    # compute_acc(target_label=np.array([[1, 0, 0], [2], 3]),
    #             output_prob=np.array([1, 2, 3]),
    #             bounding_id_trace_length=2)

    print('testing normal_td')
    sess = tf.Session()
    mu1 = tf.constant([0.480746])
    mu2 = tf.constant([0.47201255])
    var1 = tf.constant([0.3566948])
    var2 = tf.constant([0.35223854])
    y = tf.constant([0.1])
    # mu1 = tf.constant([[0.480746, 0.11928552, 0.3999685],
    #                    [0.47201255, 0.12002791, 0.40795958],
    #                    [0.48492602, 0.11869837, 0.39637566]])
    # mu2 = tf.constant([[0.47201255, 0.12002791, 0.40795958],
    #                    [0.48492602, 0.11869837, 0.39637566],
    #                    [0.4928479, 0.11631709, 0.39083505]])
    # var1 = tf.constant([[0.3566948, 0.69558066, 0.86941123],
    #                     [0.35223854, 0.6946951, 0.82976174],
    #                     [0.3467184, 0.68256944, 0.8444052]])
    # var2 = tf.constant([[0.35223854, 0.6946951, 0.82976174],
    #                     [0.3467184, 0.68256944, 0.8444052],
    #                     [0.34353644, 0.68014973, 0.8500393]])
    # y = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # y = y + 1e-10
    # print(sess.run(y))
    com1_tf, com2_tf, com3_tf = normal_td(mu1,
                                          mu2,
                                          var1,
                                          var2,
                                          y)
    com1, com2, com3 = sess.run([com1_tf, com2_tf, com3_tf])
    print(com1)
    print(com2)
    print(com3)
    print(com1 * com2 * com3)
