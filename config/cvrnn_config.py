import yaml
from support.config_tools import InitWithDict


class CVRNNCongfig(object):
    def __init__(self, init):
        self.Learn = CVRNNCongfig.Learn(init["Learn"])
        self.Arch = CVRNNCongfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        save_mother_dir = None
        batch_size = None
        keep_prob = None
        learning_rate = None
        number_of_total_game = None
        max_seq_length = None
        feature_type = None
        iterate_num = None
        model_type = None
        action_number = None
        predict_target = None
        gamma = None
        player_cluster_number = None
        data_name = None
        player_Id_style = None
        sport = None
        position_max_length = None
        apply_stochastic = None
        apply_box_score = None
        diff_apply_rl = None
        integral_update_flag = None
        embed_mode = ''

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.CVRNN = CVRNNCongfig.Arch.CVRNN(init["CVRNN"])
            self.SARSA = CVRNNCongfig.Arch.SARSA(init["SARSA"])
            self.WIN = CVRNNCongfig.Arch.WIN(init["WIN"])
            self.Predict = CVRNNCongfig.Arch.Predict(init["Predict"])

        class CVRNN(InitWithDict):
            hidden_dim = None
            latent_dim = None
            x_dim = None
            y_dim = None

        class SARSA(InitWithDict):
            lstm_layer_num = None
            h_size = None
            dense_layer_number = None
            dense_layer_size = None

        class WIN(InitWithDict):
            lstm_layer_num = None
            h_size = None
            dense_layer_number = None
            dense_layer_size = None

        class Predict(InitWithDict):
            lstm_layer_num = None
            h_size = None
            dense_layer_number = None
            dense_layer_size = None
            predict_target = None
            output_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return CVRNNCongfig(config)
