import json
import os
import pickle

import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from data_config import action_all, interested_raw_features, interested_compute_features, teamList


class Preprocess:
    def __init__(self, hockey_data_dir, save_data_dir, player_basic_info_dict, team_info_dict, game_date_dict):
        self.hockey_data_dir = hockey_data_dir
        self.save_data_dir = save_data_dir
        self.player_basic_info_dict = player_basic_info_dict
        self.team_info_list = team_info_dict
        self.game_date_dict_all = game_date_dict

    def complement_data(self):
        files_all = os.listdir(self.hockey_data_dir)
        for file in files_all:
            print(file)
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            file_name = file.split('.')[0]
            game_name = file_name.split('-')[0]
            save_game_dir = self.save_data_dir + '/' + game_name
            with open(self.hockey_data_dir + file) as f:
                data = json.load(f)
            events = data.get('events')
            gameId = data.get('gameId')

            for game_date in self.game_date_dict_all:
                if str(game_date.get('gameid')) == gameId:
                    home_team_id = str(game_date.get('team2id'))
                    away_team_id = str(game_date.get('team1id'))
                    break
            home_away_list = []
            if len(events) == 0:
                print('skip wrong file'.format(file_name))
                continue
            for event in events:
                if event.get('teamId') == home_team_id:
                    home_away_list.append(1)
                elif event.get('teamId') == away_team_id:
                    home_away_list.append(0)
                else:
                    raise ValueError('wrong home away id')

            sio.savemat(save_game_dir + "/" + "home_away_identifier_game_" + file_name + ".mat",
                        {'home_away': np.asarray(home_away_list)})

    def get_events(self, data_dir):
        with open(self.hockey_data_dir + data_dir) as f:
            data = json.load(f)
            events = data.get('events')
            gameId = data.get('gameId')
        return events, gameId

    def get_player_name(self, data_dir):
        players_info_dict = {}
        with open(self.hockey_data_dir + data_dir) as f:
            data = json.load(f)
            rosters = data.get('rosters')
        for teamId in rosters.keys():
            teamName = None
            for teaminfo in self.team_info_list:
                if teaminfo.get('teamid') == int(teamId):
                    teamName = teaminfo.get('shorthand')
                    break
            assert teamName is not None
            players = rosters.get(teamId)
            if len(players) == 0:
                continue
            for player_info in players:
                first_name = player_info.get('firstName')
                last_name = player_info.get('lastName')
                position = player_info.get('position')
                id = player_info.get('id')
                players_info_dict.update(
                    {int(id): {'first_name': first_name, 'last_name': last_name, 'position': position,
                               'teamId': int(teamId), 'teamName': teamName}})

        return players_info_dict

    def generate_player_information(self, store_player_info_dir):
        files_all = os.listdir(self.hockey_data_dir)
        players_info_dict_all = {}
        for file in files_all:
            print("### handling player name in file: ", file)
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            players_info_dict = self.get_player_name(file)
            players_info_dict_all.update(players_info_dict)

        player_global_index = 0
        for player_id in players_info_dict_all.keys():
            player_info = players_info_dict_all.get(player_id)
            player_info.update({'index': player_global_index})
            # players_info_dict_all.update({player_id: player_info})
            player_global_index += 1

        with open(store_player_info_dir, 'w') as f:
            json.dump(players_info_dict_all, f)

    def action_one_hot(self, action):
        one_hot = [0] * len(action_all)  # total 33 action
        idx = action_all.index(action)
        one_hot[idx] = 1
        return one_hot

    def team_one_hot(self, teamId):
        teamId_int = int(teamId)
        one_hot = [0] * 31  # total 31 team
        idx = teamList.index(teamId_int)
        one_hot[idx] = 1
        return one_hot

    def home_away_one_hot(self, home_away):
        one_hot = [0] * 2
        if home_away == 'H':
            one_hot[0] = 1
        elif home_away == 'A':
            one_hot[1] = 1
        return one_hot


    def get_duration(self, events, idx):
        gameTime_now = events[idx].get('gameTime')
        if idx == 0:
            duration = 0
        else:
            gameTime_pre = events[idx-1].get('gameTime')
            duration = gameTime_now - gameTime_pre
        return duration

    def get_time_remain(self, events, idx):
        gameTime = events[idx].get('gameTime')
        return 3600. - gameTime

    def is_goal(self, events, idx):
        action = events[idx].get('name')
        if idx == 0:
            return False
        elif action == 'goal':
            return True
        else:
            return False

    def is_switch_possession(self, events, idx):  # compare with former timestamp
        switch = False
        if idx == 0:
            switch = False
        else:
            team_pre = events[idx - 1].get('teamId')
            team_now = events[idx].get('teamId')
            if team_pre == team_now:
                switch = False
            else:
                switch = True
        return switch

    def is_home_away(self, events, idx, gameId):

        for game_date in self.game_date_dict_all:
            if str(game_date.get('gameid')) == gameId:
                home_team_id = str(game_date.get('team2id'))
                away_team_id = str(game_date.get('team1id'))
                break
        event = events[idx]
        if event.get('teamId') == home_team_id:
            return 'H'
        elif event.get('teamId') == away_team_id:
            return 'A'
        else:
            raise ValueError('wrong home away id')


    def is_switch_home_away(self, events, idx, gameId):  # compare with next timestamp
        switch = False
        if idx == len(events) - 1:
            switch = False
        else:
            h_a_now = self.is_home_away(events, idx, gameId)
            h_a_next = self.is_home_away(events, idx + 1, gameId)
            if h_a_now == h_a_next:
                switch = False
            else:
                switch = True
        return switch

    def get_velocity(self, coord_next, coord_now, duration):
        v = (float(coord_next) - float(coord_now)) / float(duration)
        return v

    def compute_v_x(self, events, idx, duration, gameId):
        v_x = float(0)
        if idx == len(events) - 1 or duration == 0:
            v_x = float(0)
        else:
            coord_next = events[idx + 1].get('xAdjCoord')
            coord_now = events[idx].get('xAdjCoord')
            if self.is_switch_home_away(events, idx, gameId):
                coord_next = -coord_next
            v_x = self.get_velocity(coord_next, coord_now, duration)
        return v_x

    def compute_v_y(self, events, idx, duration, gameId):
        v_y = float(0)
        if idx == len(events) - 1 or duration == 0:
            v_y = float(0)
        else:
            coord_next = events[idx + 1].get('yAdjCoord')
            coord_now = events[idx].get('yAdjCoord')
            if self.is_switch_home_away(events, idx, gameId):
                coord_next = -coord_next
            v_y = self.get_velocity(coord_next, coord_now, duration)
        return v_y

    def compute_angle2gate(self, events, idx):
        x_goal = 89
        y_goal = 0
        xAdj = events[idx].get('xAdjCoord')
        yAdj = events[idx].get('yAdjCoord')
        tant = (y_goal - yAdj) / (x_goal - xAdj)
        return tant

    def process_game_events(self, events, gameId):
        rewards_game = []
        state_feature_game = []
        action_game = []
        team_game = []
        lt_game = []
        player_id_game = []
        player_index_game = []

        lt = 0
        # reward = []
        for idx in range(0, len(events)):
            event = events[idx]
            teamId = event.get('teamId')
            teamId = int(teamId)
            action = event.get('name')

            if self.is_switch_possession(events, idx) or self.is_goal(events, idx-1):
                lt = 1
            else:
                lt = lt + 1
            try:
                action_one_hot_vector = self.action_one_hot(action)
            except:
                print('skip wrong action: {0} with index {1}'.format(action, str(idx)))
                continue
            team_one_hot_vector = self.team_one_hot(teamId)
            features_all = []
            # add raw features
            for feature_name in interested_raw_features:
                feature_value = event.get(feature_name)
                if feature_name == 'manpowerSituation':
                    if feature_value == 'powerPlay':
                        features_all.append(1.)
                    elif feature_value == 'evenStrength':
                        features_all.append(0.)
                    elif feature_value == 'shortHanded':
                        features_all.append(-1.)
                elif feature_name == 'outcome':
                    if feature_value == 'successful':
                        features_all.append(1.)
                    elif feature_value == 'undetermined':
                        features_all.append(0.)
                    elif feature_value == 'failed':
                        features_all.append(-1.)
                else:
                    features_all.append(float(feature_value))
            # add compute features
            for feature_name in interested_compute_features:
                if feature_name == 'velocity_x':
                    duration = self.get_duration(events, idx)
                    v_x = self.compute_v_x(events, idx, duration, gameId)
                    features_all.append(v_x)
                elif feature_name == 'velocity_y':
                    duration = self.get_duration(events, idx)
                    v_y = self.compute_v_y(events, idx, duration, gameId)
                    features_all.append(v_y)
                elif feature_name == 'time_remain':
                    time_remain = self.get_time_remain(events, idx)
                    features_all.append(time_remain)
                elif feature_name == 'duration':
                    duration = self.get_duration(events, idx)
                    features_all.append(duration)
                elif feature_name == 'home_away':
                    h_a = self.is_home_away(events, idx, gameId)
                    home_away_one_hot_vector = self.home_away_one_hot(h_a)
                    features_all += home_away_one_hot_vector
                elif feature_name == 'angle2gate':
                    angle2gate = self.compute_angle2gate(events, idx)
                    features_all.append(angle2gate)
            if action == 'goal' and h_a == 'H':
                print ('home goal')
                rewards_game.append(1)
            elif action == 'goal' and h_a == 'A':
                print ('away goal')
                rewards_game.append(-1)
            else:
                rewards_game.append(0)
            # rewards_game.append(reward)

            player_id = event.get('playerId')
            player_index = self.player_basic_info_dict.get(player_id).get('index')
            player_id_one_hot = [0] * len(self.player_basic_info_dict)
            player_id_one_hot[player_index] = 1
            player_index_game.append(player_id_one_hot)

            state_feature_game.append(np.asarray(features_all))
            action_game.append(np.asarray(action_one_hot_vector))
            team_game.append(np.asarray(team_one_hot_vector))
            lt_game.append(lt)
            player_id_game.append(player_id)
        return state_feature_game, action_game, team_game, lt_game, rewards_game, player_id_game, player_index_game

    def scale_allgame_features(self):
        files_all = os.listdir(self.hockey_data_dir)
        features_allgame = None
        for file in files_all:
            print("### Scale file: ", file)
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            events, gameId = self.get_events(file)
            state_feature_game, action_feature_game, _, _, _, _, _ = self.process_game_events(events, gameId)
            if len(state_feature_game) == 0:
                continue
            # game_features = np.concatenate([np.asarray(state_feature_game), np.asarray(action_feature_game)], axis=1)
            game_features = np.asarray(state_feature_game)
            if features_allgame is None:
                features_allgame = game_features
            else:
                features_allgame = np.concatenate([features_allgame, game_features], axis=0)

        scaler = preprocessing.StandardScaler().fit(features_allgame)
        print("### Scaler ###")
        print(scaler.mean_)
        print(scaler.scale_)

        with open('./feature_mean.txt', 'w') as f:
            f.write(str(scaler.mean_))
        with open('./feature_var.txt', 'w') as f:
            f.write(str(scaler.var_))

        with open('feature_scaler.pkl', 'w') as f:
            pickle.dump(scaler, f)

        return scaler

    def process_all(self, scaler):
        files_all = os.listdir(self.hockey_data_dir)
        wrong_files = []
        for file in files_all:
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            file_name = file.split('.')[0]
            game_name = file_name.split('-')[0]
            save_game_dir = self.save_data_dir + '/' + game_name
            events, gameId = self.get_events(file)
            state_feature_game, action_game, team_game, \
            lt_game, rewards_game, player_id_game, player_index_game = self.process_game_events(events, gameId)
            # try:
            game_features = np.asarray(state_feature_game)
            # game_features = np.concatenate([np.asarray(state_feature_game), np.asarray(action_game)], axis=1)
            if len(game_features) == 0:
                print 'skip wrong file {0}'.format(file)
                wrong_files.append(file)
                continue
            feature_game_scale = scaler.transform(game_features)
            state_feature_game_scale = feature_game_scale
            # state_feature_game_scale = feature_game_scale[:, :len(state_feature_game[0])]
            # action_feature_game_scale = feature_game_scale[:, len(state_feature_game[0]):]
            # except:
            #     print 'skip wrong file {0}'.format(file)
            #     wrong_files.append(file)
            #     continue
            if not os.path.exists(save_game_dir):
                os.mkdir(save_game_dir)
            print('Processing file {0}'.format(file))
            # save data to mat
            sio.savemat(save_game_dir + "/" + "reward_" + file_name + ".mat", {'reward': np.asarray(rewards_game)})
            sio.savemat(save_game_dir + "/" + "state_feature_" + file_name + ".mat",
                        {'state_feature': np.asarray(state_feature_game_scale)})
            sio.savemat(save_game_dir + "/" + "action_" + file_name + ".mat",
                        {'action': np.asarray(action_game)})
            sio.savemat(save_game_dir + "/" + "lt_" + file_name + ".mat", {'lt': np.asarray(lt_game)})
            sio.savemat(save_game_dir + "/" + "team_" + file_name + ".mat", {'team': np.asarray(team_game)})
            sio.savemat(save_game_dir + "/" + "player_id_game_" + file_name + ".mat",
                        {'player_id': np.asarray(player_id_game)})
            sio.savemat(save_game_dir + "/" + "player_index_game_" + file_name + ".mat",
                        {'player_index': np.asarray(player_index_game)})

        return wrong_files


if __name__ == '__main__':
    hockey_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
    # hockey_data_dir = '/cs/oschulte/2019-icehockey-data/2018-2019/'
    save_data_dir = '/cs/oschulte/Galen/Ice-hockey-data/2018-2019'
    prep = Preprocess(hockey_data_dir=hockey_data_dir, save_data_dir=save_data_dir)
    scaler = prep.scale_allgame_features()
    prep.process_all(scaler)
