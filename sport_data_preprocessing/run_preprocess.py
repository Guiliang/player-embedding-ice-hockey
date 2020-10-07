import json

from preprocess import Preprocess
from build_seq import process_seq_all

if __name__ == '__main__':
    hockey_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'
    save_data_dir = '/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019'
    player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    team_info_dir = '../sport_resource/ice_hockey_201819/teams_NHL.json'
    game_date_dir = '../sport_resource/ice_hockey_201819/game_dates_2018_2019.json'

    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info_dict = json.load(f)

    with open(team_info_dir, 'rb') as f:
        team_info_dict = json.load(f)

    with open(game_date_dir, 'rb') as f:
        game_date_dict = json.load(f)

    prep = Preprocess(hockey_data_dir=hockey_data_dir, save_data_dir=save_data_dir,
                      player_basic_info_dict=player_basic_info_dict, team_info_dict=team_info_dict,
                      game_date_dict=game_date_dict)
    scaler = prep.scale_allgame_features()
    prep.process_all(scaler=scaler)
    prep.complement_data()
    scaler = prep.scale_allgame_features()
    wrong_files = prep.process_all(scaler)
    process_seq_all(save_data_dir=save_data_dir)
    # print 'wrong files skipped are {0}'.format(str(wrong_files))

    # prep.generate_player_information(store_player_info_dir='../sport_resource/ice_hockey_201819/player_info_2018_2019.json')
