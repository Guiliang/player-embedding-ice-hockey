import json
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from support.ice_hockey_data_config import player_position_index_dict
from sklearn.decomposition import PCA


def compute_embedding_average(cluster_dict, player_stats_info, interest_features=[]):
    cluster_sum_dict = {}
    cluster_count_dict = {}
    for id in cluster_dict.keys():
        player_stat_dict = player_stats_info.get(str(id))
        if player_stat_dict is None:
            print(id)
            continue
        feature_list = []
        for feature in interest_features:
            feature_list.append(float(player_stat_dict.get(feature)))
        cluster_id = cluster_dict.get(id)

        if cluster_sum_dict.get(cluster_id):
            cluster_sum_list = cluster_sum_dict.get(cluster_id)
            cluster_sum_list = [x + y for x, y in zip(cluster_sum_list, feature_list)]
            cluster_sum_dict.update({cluster_id: cluster_sum_list})
            count_number = cluster_count_dict.get(cluster_id)
            cluster_count_dict.update({cluster_id: count_number + 1})
        else:
            cluster_sum_dict.update({cluster_id: feature_list})
            cluster_count_dict.update({cluster_id: 1})

    for cluster_id in cluster_sum_dict.keys():
        cluster_sum_list = cluster_sum_dict.get(cluster_id)
        count_number = cluster_count_dict.get(cluster_id)
        average_values = [str(interest_features[i]) + ':' + str(round(cluster_sum_list[i] / count_number, 2))
                          for i in range(len(cluster_sum_list))]
        print('cluster {0}:{1}'.format(str(cluster_id), str(average_values)))


def plot_embeddings(data, cluster_number, player_cluster_mapping,
                    legend_cluster_mapping, model_msg='', size=2, if_print=True):
    num_cluster = np.max(cluster_number, axis=0)
    plt.figure()
    for cluster in range(0, num_cluster + 1):
        indices = [i for i, x in enumerate(cluster_number) if x == cluster]
        if legend_cluster_mapping is not None:
            plot_label = legend_cluster_mapping[player_cluster_mapping[cluster]]
        else:
            plot_label = player_cluster_mapping[cluster]
        plt.scatter(data[indices, 0], data[indices, 1], s=size, label=plot_label)

        if if_print:
            x_plot = data[indices, 0].tolist()
            y_plot = data[indices, 1].tolist()
            max_x_index = indices[x_plot.index(max(x_plot))]  # 'find the special player'
            max_y_index = indices[y_plot.index(max(y_plot))]
            min_x_index = indices[x_plot.index(min(x_plot))]  # 'find the special player'
            min_y_index = indices[y_plot.index(min(y_plot))]
            print('cluster {0}, max_x index {1}, max_y index {2}, '
                  'min_x_index {3}, min_y_index{4}'.format(str(cluster), str(max_x_index),
                                                           str(max_y_index), str(min_x_index),
                                                           str(min_y_index)))
    # plt.show()
    plt.legend(fontsize=15, loc='lower right')
    # plt.show()
    plt.savefig('./plots/player_enc_z_cluster{0}_{1}.png'.format(str(num_cluster + 1), model_msg))


def get_features_cluster(game_features_all, cluster_type, all_encoder_values, cluster_selected=None):
    if cluster_type == 'od-zone':
        feature_cluster_index_dict = {'DZ': 0, 'OZ': 1, 'NZ': 2}
    elif cluster_type == 'manpower':
        feature_cluster_index_dict = {'evenStrength': 0, 'shortHanded': 1, 'powerPlay': 2}
    elif cluster_type == 'period':
        feature_cluster_index_dict = {1:1, 2:2, 3:3}
    else:
        feature_cluster_index_dict = {}

    feature_cluster_list = []
    embedding_selected_list = []
    cluster_id = 0
    for i in range(len(game_features_all)):
        game_features = game_features_all[i]
        if cluster_type == 'od-zone':
            if game_features['xAdjCoord']<-30:
                location_cluster_index = feature_cluster_index_dict.get('DZ')
            elif game_features['xAdjCoord']<15:
                location_cluster_index = feature_cluster_index_dict.get('NZ')
            else:
                location_cluster_index = feature_cluster_index_dict.get('OZ')
            feature_cluster_list.append(location_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])
        elif cluster_type == 'manpower' or cluster_type == 'period':
            manpower_cluster_index = feature_cluster_index_dict.get(game_features['manpowerSituation'])
            feature_cluster_list.append(manpower_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])
        elif 'action' in cluster_type:
            action = game_features['name']
            if cluster_selected is not None:
                if action not in cluster_selected:
                    continue
            if feature_cluster_index_dict.get(action) is None:
                feature_cluster_index_dict.update({action: cluster_id})
                cluster_id += 1
            action_cluster_index = feature_cluster_index_dict.get(action)
            feature_cluster_list.append(action_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])

    player_cluster_mapping = {v: k for k, v in feature_cluster_index_dict.iteritems()}
    print(player_cluster_mapping)

    cluster_count_dict = {}
    for cluster_id in feature_cluster_list:
        if cluster_count_dict.get(player_cluster_mapping[cluster_id]) is not None:
            cluster_count_dict[player_cluster_mapping[cluster_id]] += 1
        else:
            cluster_count_dict.update({player_cluster_mapping[cluster_id]: 1})
    cluster_count_dict = sorted(cluster_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    print cluster_count_dict
    return feature_cluster_list, player_cluster_mapping, embedding_selected_list


def get_player_cluster(player_index_list, player_basic_info_dir,
                       cluster_type, all_encoder_values, cluster_selected=None):
    # if player_basic_info_dir is None:
    #     player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)

    player_index_position_pair = {}
    player_index_name_pair = {}
    for player_info in player_basic_info.values():
        index = player_info.get('index')
        position = player_info.get('position')
        player_name = player_info.get('first_name')+" "+player_info.get('last_name')
        player_index_position_pair.update({index: position})
        player_index_name_pair.update({index: player_name})

    if cluster_type == 'position':
        player_cluster_index_dict = { u'C': 0, u'D': 1, u'LW': 2, u'RW': 3, u'G': 4}
    else:
        player_cluster_index_dict = {}
    cluster_id = 0

    player_cluster_list = []
    embedding_selected_list = []
    for i in range(0, len(player_index_list)):
        player_index = player_index_list[i]
        if cluster_type == 'position':
            position = player_index_position_pair.get(player_index)
            # if player_cluster_index_dict.get(position) is None:
            #     player_cluster_index_dict.update({position: cluster_id})
            #     cluster_id += 1
            player_cluster_index = player_cluster_index_dict.get(position)
            player_cluster_list.append(player_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])
        elif cluster_type == 'pindex':
            if player_cluster_index_dict.get(player_index) is None:
                player_cluster_index_dict.update({player_index: cluster_id})
                cluster_id += 1
            player_cluster_index = player_cluster_index_dict.get(player_index)
            player_cluster_list.append(player_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])
        elif cluster_type == 'name':
            player_name = player_index_name_pair.get(player_index)
            if cluster_selected is not None:
                if player_name not in cluster_selected:
                    continue
            if player_cluster_index_dict.get(player_name) is None:
                player_cluster_index_dict.update({player_name: cluster_id})
                cluster_id += 1
            player_cluster_index = player_cluster_index_dict.get(player_name)
            player_cluster_list.append(player_cluster_index)
            embedding_selected_list.append(all_encoder_values[i])
        else:
            raise ValueError('unknown {0}'.format(cluster_type))

    player_cluster_mapping = {v: k for k, v in player_cluster_index_dict.iteritems()}
    print(player_cluster_mapping)

    cluster_count_dict = {}
    for cluster_id in player_cluster_list:
        if cluster_count_dict.get(player_cluster_mapping[cluster_id]) is not None:
            cluster_count_dict[player_cluster_mapping[cluster_id]] += 1
        else:
            cluster_count_dict.update({player_cluster_mapping[cluster_id]: 1})

    cluster_count_dict = sorted(cluster_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    print cluster_count_dict
    return player_cluster_list, player_cluster_mapping, embedding_selected_list


def aggregate_positions_within_cluster(player_basic_info, cluster_number):
    player_positions = [0] * len(player_basic_info)
    for player_info in player_basic_info.values():
        index = player_info.get('index')
        position = player_info.get('position')
        player_positions[index] = position

    cluster_position_pairs = zip(cluster_number, player_positions)

    cluster_position_count_dict = {}

    for cluster_position_pair in cluster_position_pairs:
        if cluster_position_count_dict.get(cluster_position_pair[0]):
            cluster_count = cluster_position_count_dict.get(cluster_position_pair[0])
            cluster_count.update({cluster_position_pair[1]: cluster_count[cluster_position_pair[1]] + 1})
            cluster_position_count_dict.update({cluster_position_pair[0]: cluster_count})
        else:
            cluster_count = {'C': 0, 'RW': 0, 'LW': 0, 'D': 0, 'G': 0}
            cluster_count.update({cluster_position_pair[1]: 1})
            cluster_position_count_dict.update({cluster_position_pair[0]: cluster_count})
    print(cluster_position_count_dict)
    for cluster_id in cluster_position_count_dict.keys():
        print('cluster {0} with counts {1}'.format(str(cluster_id), str(cluster_position_count_dict.get(cluster_id))))
    return cluster_position_count_dict


def dimensional_reduction(embeddings, dr_method, perplexity=30):
    if dr_method == 'PCA':
        dr_embedding = PCA(n_components=2).fit_transform(embeddings)
        print ('finish pca')
    elif dr_method == 'TSNE':
        dr_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)
        print ('finish t-sne')
    else:
        raise ValueError('unknown {0}'.format(dr_method))
    return dr_embedding
