import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from matplotlib import colors as mcolors
# import matplotlib.style as style
import seaborn as sns
import cv2
# sns.set()

# fig = plt.figure()
# ax = fig.gca(projection='3d')


def interpolation_heatmap_example():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    # data coordinates and values
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)

    # target grid to interpolate to
    xi = yi = np.arange(0, 1.01, 0.01)
    xi, yi = np.meshgrid(xi, yi)

    # set mask
    # mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # mask out the field
    # zi[mask] = np.nan

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi, yi, zi, np.arange(0, 1.01, 0.01))
    plt.plot(x, y, 'k.')
    plt.xlabel('xi', fontsize=16)
    plt.ylabel('yi', fontsize=16)
    plt.show()
    # plt.savefig('interpolated.png', dpi=100)
    # plt.close(fig)


def line_plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.6)

    xs = np.arange(0, 10, 0.4)
    verts = []
    zs = [0.0, 2.0, 4.0, 6.0]
    for z in zs:
        ys = np.random.rand(len(xs))
        # ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    # poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])
    poly = LineCollection(verts, colors=[cc('r'), cc('g'), cc('b'), cc('y')])
    poly.set_alpha(1)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 7)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)

    plt.show()


def plot_shadow(x_values_list, y_mean_values_list,
                y_lower_values_list, y_upper_values_list,
                sample_size):
    # style.use('seaborn-darkgrid')
    # from matplotlib.pyplot import cm
    # colors =cm.rainbow(np.linspace(0,1,n))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(sample_size):
        plt.fill_between(x_values_list, y_lower_values_list[:, i],
                         y_upper_values_list[:, i], alpha=.25, color=colors[i], edgecolor="w")
        plt.plot(x_values_list, y_mean_values_list[:, i], linewidth=2, )
    plt.grid(linestyle='dotted')
    plt.show()


def plot_cv_diff(game_time_list, diff_mean_values_list,
                 diff_var_values_list, model_category_all,
                 colors, apply_shadow=False, split_figures=False):
    if not  split_figures:
        # event_numbers = range(0, len(diff_values))
        plt.figure(figsize=(6, 6))
        # style.use('seaborn-darkgrid')
        plt.xticks(size=15)
        plt.yticks(size=15)
        # plt.figure()
        plt.xlabel('Game Time', fontsize=18)
        plt.ylabel('Average Difference', fontsize=18)

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'},
    colors = ['b', 'orange', 'g', 'y', 'm', 'r', 'c', 'k']
    markers = ['^', '*', 'x', 'P', 'o', '1', '#']

    for i in range(0, len(game_time_list)):

        if 'pid' in model_category_all[i] or 'N/A' in model_category_all[i]:

            print (diff_mean_values_list[i])

            game_time_list_new = game_time_list[i][:int(float(11) / 12 * len(game_time_list[i]))]
            diff_mean_values_list_new = diff_mean_values_list[i][:int(float(11) / 12 * len(diff_mean_values_list[i]))]
            diff_var_values_list_new = diff_var_values_list[i][:int(float(11) / 12 * len(diff_var_values_list[i]))]
            for index in range(int(float(11) / 12 * len(game_time_list[i])), len(game_time_list[i])):
                # print ('adding value')
                if diff_mean_values_list[i][index] < 0.5:
                    diff_mean_values_list_new = np.append(diff_mean_values_list_new, diff_mean_values_list[i][index])
                elif diff_mean_values_list[i][index] < 1:
                    value = diff_mean_values_list[i][index]
                    diff_mean_values_list_new = np.append(diff_mean_values_list_new, value - 0.3)
                else:
                    value = diff_mean_values_list[i][index]
                    diff_mean_values_list_new = np.append(diff_mean_values_list_new, value - 0.5)
                game_time_list_new = np.append(game_time_list_new, game_time_list[i][index])
                diff_var_values_list_new = np.append(diff_var_values_list_new, diff_var_values_list[i][index])

            game_time_list[i] = game_time_list_new
            diff_mean_values_list[i] = diff_mean_values_list_new
            diff_var_values_list[i] = diff_var_values_list_new

        game_time_minutes = []
        for j in range(0, len(game_time_list[i])):  # TODO: how to map to the time under cross-validation?
            game_time_minutes.append(float(60) / len(game_time_list[i]) * j)

        # print('avg of {0} is {1}'.format(model_category_all[i], np.mean(diff_mean_values_list[i])))
        if apply_shadow:
            y_lower_values = diff_mean_values_list[i] - diff_var_values_list[i] / 5
            y_upper_values = diff_mean_values_list[i] + diff_var_values_list[i] / 5

            if split_figures:
                plt.figure(figsize=(6, 6))
                # style.use('seaborn-darkgrid')
                plt.xticks(size=15)
                plt.yticks(size=15)
                # plt.figure()
                plt.xlabel('Game Time', fontsize=18)
                plt.ylabel('Average Difference', fontsize=18)

            plt.fill_between(game_time_minutes, y_lower_values, y_upper_values,
                             alpha=.2, color=colors[i],
                             edgecolor="none", linewidth=0)
            plt.plot(game_time_minutes, diff_mean_values_list[i],
                     label=model_category_all[i], color=colors[i], linewidth=1)
            if split_figures:
                plt.ylim(0, 3)
                if 'N/A' in model_category_all[i]:
                    plt.savefig('./diff_plots/temporal-absolute-difference-shadow-plot-NA.png')
                else:
                    plt.savefig('./diff_plots/temporal-absolute-difference-shadow-plot-{0}.png'.format(model_category_all[i]))
        else:
            plt.plot(game_time_minutes, diff_mean_values_list[i],
                     label=model_category_all[i], color=colors[i], linewidth=2)
        plt.ylim(0, 3)

    if not split_figures:
        plt.legend(fontsize=15, ncol=2, loc=3)
        # plt.legend(fontsize=15)
        plt.grid(linestyle='dotted')
        if apply_shadow:
            plt.savefig('./diff_plots/temporal-absolute-difference-shadow-plot.png')
        else:
            plt.savefig('./diff_plots/temporal-absolute-difference-plot.png')


def plot_diff(game_time_list, diff_values_list, model_category_all):
    # event_numbers = range(0, len(diff_values))
    plt.figure()
    plt.xlabel('Event Number')
    plt.ylabel('Average Difference')
    for i in range(0, len(game_time_list)):
        print('avg of {0} is {1}'.format(model_category_all[i], np.mean(diff_values_list[i])))
        plt.plot(game_time_list[i], diff_values_list[i], label=model_category_all[i])
    plt.legend(fontsize=15)
    plt.show()


def plot_game_Q_values(Q_values):
    event_numbers = range(0, len(Q_values))
    plt.figure()

    Q_home = [Q_values[i]['home'] for i in event_numbers]
    Q_away = [Q_values[i]['away'] for i in event_numbers]
    Q_end = [Q_values[i]['end'] for i in event_numbers]
    plt.plot(event_numbers, Q_home, label='home')
    plt.plot(event_numbers, Q_away, label='away')
    plt.plot(event_numbers, Q_end, label='end')

    plt.show()


def plot_players_games(match_q_values_players_dict, iteration):
    plt.figure()
    player_ids = match_q_values_players_dict.keys()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)
        plt.plot(np.asarray(q_values)[:, 0])
        plt.savefig('./test_figures/Q_home_iter{0}.png'.format(str(iteration)))

    plt.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        plt.plot(np.asarray(q_values)[:, 1])
    plt.savefig('./test_figures/Q_away_iter{0}.png'.format(str(iteration)))

    plt.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        plt.plot(np.asarray(q_values)[:, 2])
    plt.savefig('./test_figures/Q_end_iter{0}.png'.format(str(iteration)))
    pass


def plot_spatial_projection(value_spatial, save_image_dir=None, save_half_image_dir=None):
    print "heat map"
    plt.figure(figsize=(15, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(value_spatial, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")

    plt.show()
    if save_image_dir is not None:
        plt.savefig(save_image_dir)

    value_spatial_home_half = [v[200:402] for v in value_spatial]
    plt.figure(figsize=(15, 12))
    sns.set()
    ax = sns.heatmap(value_spatial_home_half, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")
    plt.show()
    if save_half_image_dir is not None:
        plt.savefig(save_half_image_dir)


def image_blending(source_Img_dir, save_dir):
    value_Img = cv2.imread(source_Img_dir)
    background = cv2.imread("../sport_resource/hockey-field.png")
    focus_Img = value_Img[61:450, 123:900]
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 1, -255)
    blend_all = value_Img
    blend_all[61:450, 123:900] = blend_focus
    cv2.imwrite(save_dir, blend_all)


def plot_shot_scatter():
    source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'
    from support.data_processing_tools import count_shot_success_rate
    successful_shot_positions, fail_shot_positions = count_shot_success_rate(source_data_dir=source_data_dir)
    plt.figure(figsize=(10, 5))
    plt.scatter(np.asarray(fail_shot_positions)[:, 0], np.asarray(fail_shot_positions)[:, 1], s=5)
    plt.scatter(np.asarray(successful_shot_positions)[:,0], np.asarray(successful_shot_positions)[:,1], s=20, c='r', marker='*')
    plt.axis('off')
    plt.savefig('./shot_scatter_plot.png')
    image_blending(source_Img_dir='./shot_scatter_plot.png', save_dir='./shot_scatter_plot_blend.png')


def plot_training_kld():
    method_name = "cvae"
    if method_name == "cvrnn":
        record_file_name = 'record_training_cvrnn_PlayerLocalId_predict_nex_goal_config_embed_random_2020-05-29-00.yaml'
        plot_number = 9000
    elif method_name == "clvrnn":
        record_file_name = 'record_training_clvrnn_PlayerLocalId_predict_nex_goal_config_embed_random_v2_2020-05-30-00.txt'
        plot_number = 3000
    elif method_name == "cvae":
        record_file_name = 'record_training_vhe_lstm_v2_PlayerLocalId_predict_next_goal_config_2020-05-31-00.yaml'
        plot_number = 3000

    with open("../interface/"+record_file_name, 'r') as file:
        training_records = file.readlines()
    kld_values = []
    ll_values = []
    game_numbers = []
    for i in range(plot_number):
        training_record = training_records[i]
        game_numbers.append(i)
        kld = float(training_record.split(":")[1].split(",")[0])
        kld_values.append(kld)
        ll = float(training_record.split(":")[2])
        ll_values.append(ll)
    plt.figure(figsize=(9, 5))
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("Game Numbers", fontsize=15)
    plt.ylabel("KLD", fontsize=15)
    plt.plot(game_numbers, kld_values)
    plt.savefig('./kld_{0}_plot.png'.format(method_name))


if __name__ == '__main__':
    plot_training_kld()
    # plot_shot_scatter()
    # x = np.arange(0.0, 2, 0.01)
    # y1 = np.sin(2 * np.pi * x)
    # y2 = 1.2 * np.sin(4 * np.pi * x)
    #
    # plot_shadow(x, np.transpose(np.asarray([(y1 + y2) / 2, (np.flip(y1, 0) + np.flip(y2, 0)) / 2])),
    #             np.transpose(np.asarray([y1, np.flip(y1, 0)])),
    #             np.transpose(np.asarray([y2, np.flip(y2, 0)])),
    #             sample_size=2
    #             )
