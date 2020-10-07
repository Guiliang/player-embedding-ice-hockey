import os
import tensorflow as tf
import numpy as np
from config.varlea_config import VaRLEACongfig
from config.cvrnn_config import CVRNNCongfig
from support.model_tools import get_model_and_log_name, get_data_name, validate_games_player_id, \
    validate_model_initialization

if __name__ == '__main__':
    local_test_flag = False
    model_category = 'varlea'
    model_number =5401
    player_info = ''
    apply_bounding = False
    if apply_bounding:
        msg_bounding = '_bound'
    else:
        msg_bounding = ''
    apply_sparse_number = None
    if apply_sparse_number is not None:
        sparse_msg = '_sparse_{0}'.format(apply_sparse_number)
    else:
        sparse_msg = ''

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if model_category == 'cvrnn':
        if apply_bounding:
            msg_bounding = '_bound'
        embed_mode = '_embed_random'
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn_PlayerLocalId_predict_nex_goal_config_embed_random.yaml"
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_category == 'varlea':
        if apply_bounding:
            msg_bounding = '_bound'
        embed_mode = '_embed_random_v2'
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_varlea_config_path = "../../environment_settings/icehockey_varlea_PlayerLocalId_predict_nex_goal_config_embed_random.yaml"
        icehockey_model_config = VaRLEACongfig.load(icehockey_varlea_config_path)
    else:
        raise ValueError("uknown model catagoery {0}".format(model_category))
    sess_nn = tf.InteractiveSession()
    model_nn = validate_model_initialization(sess_nn=sess_nn, model_category=model_category,
                                             config=icehockey_model_config)

    acc_all = []
    ll_all = []

    # running_numbers = [0,1,2,3,4]
    running_numbers = [0]

    with open('./results/player_id_acc_' + model_category + '_'
              + str(model_number) + player_info + msg_bounding+sparse_msg+'_q_cv', 'wb') as file_writer:

        for running_number in running_numbers:
            saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                                model_catagoery=model_category,
                                                                running_number=running_number)

            testing_dir_games_all = []
            # with open('../../sport_resource/ice_hockey_201819/testing_file_dirs_all.csv', 'rb') as f:
            with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
                testing_dir_all = f.readlines()
            for testing_dir in testing_dir_all:
                testing_dir_games_all.append(str(int(testing_dir)))
            model_data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
            source_data_store = '/Local-Scratch/oschulte/Galen/2018-2019/'

            # data_name = get_data_name(icehockey_model_config, model_category, model_number)

            print(model_category + '_' + str(model_number) + player_info)

            if local_test_flag:
                data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
            else:
                data_store_dir = icehockey_model_config.Learn.save_mother_dir \
                                 + "/oschulte/Galen/Ice-hockey-data/2018-2019/"

            model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)

            if apply_sparse_number == 100:
                player_sparse_presence_all = [19, 75, 82, 115, 183, 285, 311, 313, 320, 323, 388, 416, 423, 457, 464, 488, 500, 527, 543, 548, 555, 634, 649, 656, 658, 665, 735, 746, 761, 768, 772, 788, 795, 799, 817, 822, 848, 865, 867, 872, 898, 928, 957, 958, 960, 961, 965, 966, 974, 993, 995]
            elif apply_sparse_number == 200:
                player_sparse_presence_all = [15, 19, 38, 40, 42, 75, 82, 101, 115, 183, 234, 285, 311, 313, 320, 323, 331, 346, 388, 416, 422, 423, 451, 453, 457, 464, 465, 488, 500, 527, 543, 548, 553, 555, 572, 615, 634, 635, 639, 649, 656, 658, 665, 711, 719, 735, 746, 761, 762, 768, 772, 788, 791, 793, 795, 799, 813, 817, 822, 848, 852, 865, 867, 872, 888, 892, 898, 915, 918, 928, 957, 958, 960, 961, 965, 966, 974, 993, 995]
            elif apply_sparse_number == 300:
                player_sparse_presence_all = [12, 15, 19, 38, 40, 42, 74, 75, 82, 87, 101, 115, 130, 157, 170, 183, 234, 285, 310, 311, 313, 320, 323, 331, 346, 361, 364, 388, 416, 422, 423, 434, 442, 451, 453, 457, 464, 465, 468, 476, 488, 495, 500, 511, 519, 527, 534, 538, 543, 548, 553, 555, 571, 572, 573, 615, 634, 635, 639, 642, 649, 656, 658, 665, 692, 711, 719, 735, 746, 754, 761, 762, 768, 772, 788, 791, 793, 795, 799, 813, 817, 822, 827, 848, 852, 855, 857, 858, 865, 867, 872, 888, 890, 892, 898, 914, 915, 918, 928, 957, 958, 960, 961, 964, 965, 966, 974, 993, 995]
            else:
                player_sparse_presence_all = None

            ac, ll = validate_games_player_id(config=icehockey_model_config,
                                              data_store_dir=data_store_dir,
                                              dir_all=testing_dir_games_all,
                                              model_nn=model_nn,
                                              sess_nn=sess_nn,
                                              model_path=model_path,
                                              player_basic_info_dir='../../sport_resource/ice_hockey_201819/player_info_2018_2019.json',
                                              game_date_dir='../../sport_resource/ice_hockey_201819/game_dates_2018_2019.json',
                                              player_box_score_dir='../../sport_resource/ice_hockey_201819/Scale_NHL_players_game_summary_201819.csv',
                                              data_store=data_store_dir,
                                              apply_bounding=apply_bounding,
                                              model_number=model_number,
                                              player_id_cluster_dir=player_id_cluster_dir,
                                              saved_network_dir=saved_network_dir,
                                              model_category=model_category,
                                              file_writer=file_writer,
                                              player_sparse_presence_all=player_sparse_presence_all)
            acc_all.append(ac)
            ll_all.append(ll)

        acc_all = np.asarray(acc_all)
        ac_avg_str = "avg testing acc is {0} with variance {1}\n".format(str(np.mean(acc_all)),
                                                                         str(np.var(acc_all)))
        print(ac_avg_str)
        file_writer.write(ac_avg_str)

        ll_all = np.asarray(ll_all)
        ll_avg_str = "avg testing ll is {0} with variance {1} \n".format(str(np.mean(ll_all)),
                                                                         str(np.var(ll_all)))
        print(ll_avg_str)
        file_writer.write(ll_avg_str)
