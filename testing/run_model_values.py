import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import json
import os
import tensorflow as tf
from random import shuffle
from config.clvrnn_config import CLVRNNCongfig
from config.cvrnn_config import CVRNNCongfig
from support.model_tools import compute_games_Q_values, get_model_and_log_name, validate_model_initialization, \
    get_data_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    model_type = 'clvrnn'
    player_info = ''
    model_number = 12301
    local_test_flag = False
    if model_type == 'cvrnn':
        embed_mode = '_embed_random'
        predicted_target = '_PlayerLocalId_predict_nex_goal'
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../environment_settings/icehockey_cvrnn{0}_config{1}{2}.yaml".format(
            predicted_target, player_info, embed_mode)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_type == 'clvrnn':
        embed_mode = '_embed_random_v2'
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_clvrnn_config_path = "../environment_settings/icehockey_clvrnn{0}_config{1}{2}.yaml". \
            format(predicted_target, player_info, embed_mode)
        icehockey_model_config = CLVRNNCongfig.load(icehockey_clvrnn_config_path)
    else:
        raise ValueError('incorrect model type {0}'.format(model_type))

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)

    sess_nn = tf.InteractiveSession()
    model_nn = validate_model_initialization(sess_nn=sess_nn, model_category=model_type,
                                             config=icehockey_model_config)
    running_numbers = [0]
    # running_numbers = [0, 1, 2, 3, 4]

    # cv_record_all_model_next_Q_values = []
    # cv_record_all_model_accumu_Q_values = []
    #
    # for dir_game_index in range(0, len(dir_games_all)):
    #     game_cv_record = {}
    #     for running_number in running_numbers:
    #         game_cv_record.update({running_number: None})
    #     cv_record_all_model_next_Q_values.append({dir_game_index: game_cv_record})
    #     cv_record_all_model_accumu_Q_values.append({dir_game_index: game_cv_record})

    for running_number in running_numbers:
        print('handing games for running number {0}'.format(str(running_number)))
        saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                            model_catagoery=model_type,
                                                            running_number=running_number)
        if model_type == 'lstm_Qs' or model_type == 'lstm_diff':
            model_path = saved_network_dir +'/Ice-Hockey-game--{0}'.format(model_number)
        else:
            model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
        # dir_games_all = ['16276']
        compute_games_Q_values(config=icehockey_model_config,
                               data_store_dir=data_store_dir,
                               dir_all=dir_games_all,
                               model_nn=model_nn,
                               sess_nn=sess_nn,
                               model_path=model_path,
                               model_number=model_number,
                               player_id_cluster_dir=player_id_cluster_dir,
                               model_category=model_type,
                               return_values_flag=False,
                               apply_cv=True,
                               running_number=running_number)
        # for dir_game_index in range(0, len(dir_games_all)):
        #     cv_record_all_model_next_Q_values[dir_game_index].update(
        #         {running_number: model_next_Q_values_all[dir_game_index]})
        #     cv_record_all_model_accumu_Q_values[dir_game_index].update(
        #         {running_number: model_accumu_Q_value_all[dir_game_index]})

    # dir_games_all = dir_games_all[1:]
    for dir_game_index in range(0, len(dir_games_all)):
        data_name = get_data_name(config=icehockey_model_config,
                                  model_catagoery=model_type,
                                  model_number=model_number)
        game_name_dir = dir_games_all[dir_game_index]
        game_store_dir = game_name_dir.split('.')[0]
        game_all_next_Qs_values = {}
        game_all_accumu_Qs_values = {}
        for running_number in running_numbers:
            if model_type != 'lstm_diff' and model_type != 'multi_agent' and model_type != 'clvrnn' and model_type != 'caernn':
                with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs')
                          + '_r'+str(running_number), 'r') as outfile:
                    cv_next_Qs_game_values = json.load(outfile)
                game_all_next_Qs_values.update({running_number: cv_next_Qs_game_values})
                os.remove(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs')
                          + '_r'+str(running_number))

            if model_type != 'lstm_Qs':
                with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs')
                          + '_r'+str(running_number), 'r') as outfile:
                    cv_accumu_Qs_game_values = json.load(outfile)
                game_all_accumu_Qs_values.update({running_number: cv_accumu_Qs_game_values})
                os.remove(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs')
                          + '_r'+str(running_number))

        if model_type != 'lstm_diff' and model_type != 'multi_agent' and model_type != 'clvrnn':
            with open(data_store_dir + "/" + game_store_dir + "/"
                      + data_name.replace('Qs', 'next_Qs')+'_cv', 'w') as outfile:
                json.dump(game_all_next_Qs_values, outfile)

        if model_type != 'lstm_Qs':
            with open(data_store_dir + "/" + game_store_dir + "/"
                      + data_name.replace('Qs', 'accumu_Qs')+'_cv', 'w') as outfile:
                json.dump(game_all_accumu_Qs_values, outfile)
