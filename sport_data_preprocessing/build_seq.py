import scipy.io as sio
import numpy as np
import os


def construct_seq_state_action_feature(s_f, a_f, lt):
    rows = s_f.shape[0]
    feature_num = s_f.shape[1]
    action_num = a_f.shape[1]
    state_feature_seq = np.zeros([rows, 10, feature_num])
    action_feature_seq = np.zeros([rows, 10, action_num])
    for i in range(rows):
        trace = lt[0][i]
        if trace > 10:
            trace = 10
        idx = trace - 1
        cnt = 0
        while (idx >= 0):
            state_feature_seq[i, idx, :] = s_f[i - cnt, :]
            action_feature_seq[i, idx, :] = a_f[i - cnt, :]
            idx = idx - 1
            cnt = cnt + 1
    return state_feature_seq, action_feature_seq


def process_seq_all(save_data_dir, start_folder='16752'):
    # start_flag = False
    folder_all = os.listdir(save_data_dir)
    for folder in folder_all:
        if folder == '.DS_Store':
            continue
        # if folder == start_folder:
        #     start_flag = True
        # if not start_flag:
        #     continue
        folder_path = save_data_dir + '/' + folder
        file_all = os.listdir(folder_path)
        state_feature_path = None
        action_feature_path = None
        lt_path = None
        for file in file_all:
            if "state_feature_seq" in file:
                continue
            if "action_feature_seq" in file:
                continue
            if "state" in file:
                state_feature_path = folder_path + '/' + file
            if "action" in file:
                action_feature_path = folder_path + '/' + file
            if "lt" in file:
                lt_path = folder_path + '/' + file
            # if 'player'
        if state_feature_path is None:
            print ('wrong files in {0}'.format(folder))
            continue
        if lt_path is None:
            print ('wrong files in {0}'.format(folder))
            continue
        state_feature = sio.loadmat(state_feature_path)
        state_feature = state_feature["state_feature"]

        action_feature = sio.loadmat(action_feature_path)
        action_feature = action_feature["action"]

        lt = sio.loadmat(lt_path)
        lt = lt["lt"]
        print(folder)
        state_feature_seq, action_feature_seq = construct_seq_state_action_feature(state_feature, action_feature, lt)
        sio.savemat(folder_path + '/' + 'state_feature_seq_' + folder + '.mat',
                    {'state_feature_seq': state_feature_seq})
        sio.savemat(folder_path + '/' + 'action_feature_seq' + folder + '.mat',
                    {'action_feature_seq': action_feature_seq})


if __name__ == '__main__':
    save_data_dir = '/Local_Scratch/oschulte/Galen/Ice-hockey-data/2018-2019'
    process_seq_all(save_data_dir)
