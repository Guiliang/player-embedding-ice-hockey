import yaml
from support.config_tools import InitWithDict


class VaRLEACongfig(object):
    def __init__(self, init):
        self.Learn = VaRLEACongfig.Learn(init["Learn"])
        self.Arch = VaRLEACongfig.Arch(init["Arch"])

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
            self.CLVRNN = VaRLEACongfig.Arch.CLVRNN(init["CLVRNN"])
            self.SARSA = VaRLEACongfig.Arch.SARSA(init["SARSA"])
            self.WIN = VaRLEACongfig.Arch.WIN(init["WIN"])
            self.Predict = VaRLEACongfig.Arch.Predict(init["Predict"])

        class CLVRNN(InitWithDict):
            hidden_dim = None
            latent_a_dim = None
            latent_s_dim = None
            x_dim = None
            y_s_dim = None
            y_a_dim = None
            y_r_dim = None

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
        return VaRLEACongfig(config)
