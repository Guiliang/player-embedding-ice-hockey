import tensorflow as tf

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


class LadderVariationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self, config, train_flag):
        self.config = config
        x_dim = self.config.Arch.CLVRNN.x_dim
        y_s_dim = self.config.Arch.CLVRNN.y_s_dim
        y_a_dim = self.config.Arch.CLVRNN.y_a_dim
        y_r_dim = self.config.Arch.CLVRNN.y_r_dim
        h_dim = self.config.Arch.CLVRNN.hidden_dim
        z_s_dim = self.config.Arch.CLVRNN.latent_s_dim
        z_a_dim = self.config.Arch.CLVRNN.latent_a_dim
        z_r_dim = self.config.Arch.CLVRNN.latent_r_dim
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_y_list = [y_a_dim, y_r_dim, y_s_dim]
        self.n_z = [z_r_dim, z_a_dim, z_s_dim]
        self.bn_train_flag = train_flag

        # self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.n_h, state_is_tuple=True)

        embed_dim = self.n_z[0]

        self.output_dim_list = [self.n_z[2], self.n_z[2], self.n_z[1], self.n_z[1], self.n_z[0], self.n_z[0],
                                self.n_z[2], self.n_z[2], self.n_z[1], self.n_z[1], self.n_z[0], self.n_z[0],
                                self.n_x, embed_dim, embed_dim]

        self.e_mus, self.e_sigmas = [0] * 3, [0] * 3
        self.p_mus, self.p_sigmas = [0] * 3, [0] * 3
        self.d_mus, self.d_sigmas = [0] * 3, [0] * 3

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        # enc_mu, enc_sigma, dec_mu, dec_sigma, dec_x, prior_mu, prior_sigma
        return sum(self.output_dim_list)
        # return self.n_h

    def precision_weighted_sampler(self, musigma1, musigma2, n_z):

        def precision_weighted(musigma1, musigma2):
            mu1, sigma1 = musigma1
            mu2, sigma2 = musigma2
            sigma1__2 = 1 / tf.square(sigma1)
            sigma2__2 = 1 / tf.square(sigma2)
            mu = (mu1 * sigma1__2 + mu2 * sigma2__2) / (sigma1__2 + sigma2__2)
            sigma = 1 / (sigma1__2 + sigma2__2)
            return (mu, sigma)

        # assume input Tensors are (BATCH_SIZE, dime)
        mu1, sigma1 = musigma1
        mu2, sigma2 = musigma2
        size_1 = mu1.get_shape().as_list()[1]
        size_2 = mu2.get_shape().as_list()[1]

        if size_1 > size_2:
            print('convert 1d to 1d:', size_2, '->', size_1)
            with tf.variable_scope("precision_weighted"):
                mu2 = linear(mu2, size_1)
                sigma2 = linear(sigma2, size_1)
                musigma2 = (mu2, sigma2)
        elif size_1 < size_2:
            raise ValueError("musigma1 must be equal or bigger than musigma2.")
        else:
            # not need to convert
            pass

        mu, sigma = precision_weighted(musigma1, musigma2)
        eps_p = tf.random_normal((tf.shape(mu)[0], n_z), 0.0, 1.0, dtype=tf.float32)
        z_p = tf.add(mu, tf.multiply(sigma, eps_p))

        return z_p, mu, sigma

    def __call__(self, input, state, scope="clvrnn", inherit_upper_post=False):

        def cvae_layer(input, condition, n_h, n_z, is_train):
            with tf.variable_scope("cond_input"):
                ic = tf.concat(values=(input, linear(condition, n_h)), axis=1)
            with tf.variable_scope("hidden"):
                vae_hidden = tf.nn.relu(tf.layers.batch_normalization(linear(ic, n_h), training=is_train))
            with tf.variable_scope("mu"):
                vae_mu = linear(vae_hidden, n_z)
            with tf.variable_scope("sigma"):
                vae_sigma = tf.nn.softplus(linear(vae_hidden, n_z))
            return vae_mu, vae_sigma, vae_hidden

        with tf.variable_scope(scope or type(self).__name__):
            c, m = state

            [x, y_o, y_r, y_a, train_flag_ph] = tf.split(input, [self.n_x, self.n_y_list[2], self.n_y_list[1],
                                                                 self.n_y_list[0], 1], axis=1)
            train_flag = tf.cast(tf.squeeze(train_flag_ph), tf.bool)

            bn_train_flag = tf.cast(tf.reduce_mean(tf.squeeze(train_flag_ph)), tf.bool)
            # bn_train_flag = True
            # bottom up ng bottom-up information
            with tf.variable_scope("Encoder_cond_r"):
                enc_mu_r, enc_sigma_r, h_r = cvae_layer(input=x, condition=y_r,
                                                        n_h=self.n_h, n_z=self.n_z[0],
                                                        is_train=bn_train_flag)
                self.e_mus[0] = enc_mu_r
                self.e_sigmas[0] = enc_sigma_r

            with tf.variable_scope("Encoder_cond_a"):
                enc_mu_a, enc_sigma_a, h_a = cvae_layer(input=h_r, condition=y_a,
                                                        n_h=self.n_h, n_z=self.n_z[1],
                                                        is_train=bn_train_flag)
                self.e_mus[1] = enc_mu_a
                self.e_sigmas[1] = enc_sigma_a

            with tf.variable_scope("Encoder_cond_s"):
                y_s = tf.concat(values=[linear(y_o, self.n_h), m], axis=1)
                enc_mu_s, enc_sigma_s, h_s = cvae_layer(input=h_a, condition=y_s,
                                                        n_h=self.n_h, n_z=self.n_z[2],
                                                        is_train=bn_train_flag)
                self.e_mus[2] = enc_mu_s
                self.e_sigmas[2] = enc_sigma_s

            # top-down prior information
            with tf.variable_scope("Inference_cond_s"):
                with tf.variable_scope("prior"):
                    y_s = tf.concat(values=(linear(y_o, self.n_h, scope='Linear_ys'), m), axis=1)
                    prior_s_phi = tf.nn.relu(tf.layers.batch_normalization(
                        linear(y_s, self.n_h, scope='Linear_prior_s'), training=bn_train_flag))
                    self.p_mus[2] = linear(prior_s_phi, self.n_z[2], scope='Linear_mu')
                    self.p_sigmas[2] = tf.nn.softplus(linear(prior_s_phi, self.n_z[2], scope='Linear_sigma'))
                    eps_ps = tf.random_normal((tf.shape(x)[0], self.n_z[2]), 0.0, 1.0, dtype=tf.float32)
                    z_prior_s = tf.add(self.p_mus[2], tf.multiply(self.p_sigmas[2], eps_ps))
                with tf.variable_scope("posterior"):
                    dec_mu_a, dec_sigma_a = self.e_mus[2], self.e_sigmas[2]
                    self.d_mus[2], self.d_sigmas[2] = dec_mu_a, dec_sigma_a
                    eps_qs = tf.random_normal((tf.shape(x)[0], self.n_z[2]), 0.0, 1.0, dtype=tf.float32)
                    z_q_s = tf.add(self.d_mus[2], tf.multiply(self.d_sigmas[2], eps_qs))

            with tf.variable_scope("Inference_cond_a"):
                with tf.variable_scope("prior"):

                    y_za = tf.concat(values=(linear(y_a, self.n_h, scope='Linear_ya'),
                                             linear(z_prior_s, self.n_h, scope='Linear_za')), axis=1)
                    prior_a_phi = tf.nn.relu(
                        tf.layers.batch_normalization(linear(y_za, self.n_h, scope='Linear_prior_za'),
                                                      training=bn_train_flag))
                    self.p_mus[1] = linear(prior_a_phi, self.n_z[1], scope='mu')
                    self.p_sigmas[1] = tf.nn.softplus(linear(prior_a_phi, self.n_z[1], scope='sigma'))
                    eps_qa = tf.random_normal((tf.shape(x)[0], self.n_z[1]), 0.0, 1.0, dtype=tf.float32)
                    z_prior_a = tf.add(self.p_mus[1], tf.multiply(self.p_sigmas[1], eps_qa))
                with tf.variable_scope("posterior"):
                    if inherit_upper_post:
                        z_q_a, self.d_mus[1], self.d_sigmas[1] = self.precision_weighted_sampler(
                            (tf.concat([self.e_mus[1], self.d_mus[2]], axis=1),
                             tf.concat([self.e_sigmas[1], self.d_sigmas[2]], axis=1)),
                            (self.p_mus[1], self.p_sigmas[1]), n_z=self.n_z[1]
                        )
                    else:
                        z_q_a, self.d_mus[1], self.d_sigmas[1] = self.precision_weighted_sampler(
                            (self.e_mus[1], self.e_sigmas[1]),
                            (self.p_mus[1], self.p_sigmas[1]), n_z=self.n_z[1]
                        )

            with tf.variable_scope("Inference_cond_r"):
                with tf.variable_scope("prior"):

                    y_zr = tf.concat(values=(linear(y_r, self.n_h, scope='Linear_yr'),
                                             linear(z_prior_a, self.n_h, scope='Linear_zr')), axis=1)
                    prior_r_phi = tf.nn.relu(
                        tf.layers.batch_normalization(linear(y_zr, self.n_h, scope='Linear_prior_zr'),
                                                      training=bn_train_flag))
                    self.p_mus[0] = linear(prior_r_phi, self.n_z[0], scope='mu')
                    self.p_sigmas[0] = tf.nn.softplus(linear(prior_r_phi, self.n_z[0], scope='sigma'))
                    eps_qr = tf.random_normal((tf.shape(x)[0], self.n_z[0]), 0.0, 1.0, dtype=tf.float32)
                    z_prior_r = tf.add(self.p_mus[0], tf.multiply(self.p_sigmas[0], eps_qr))
                with tf.variable_scope("posterior"):
                    if inherit_upper_post:
                        z_q_r, self.d_mus[0], self.d_sigmas[0] = self.precision_weighted_sampler(
                            (tf.concat([self.e_mus[0], self.d_mus[1]], axis=1),
                             tf.concat([self.e_sigmas[0], self.d_sigmas[1]], axis=1)),
                            (self.p_mus[0], self.p_sigmas[0]), n_z=self.n_z[0]
                        )
                    else:
                        z_q_r, self.d_mus[0], self.d_sigmas[0] = self.precision_weighted_sampler(
                            (self.e_mus[0], self.e_sigmas[0]),
                            (self.p_mus[0], self.p_sigmas[0]), n_z=self.n_z[0]
                        )

            with tf.variable_scope("cond_z"):
                z = tf.where(train_flag, x=z_q_r, y=z_prior_r)

            with tf.variable_scope("decoder_output"):

                dec_hidden_enc = tf.nn.relu(linear(z, self.n_h, scope='Linear_z_phi'))
                recon_output = linear(dec_hidden_enc, self.n_x, scope='Linear_output')

            with tf.variable_scope("hidden_state"):
                zyx_phi = tf.nn.relu(linear(tf.concat(values=(z, y_r, y_a, y_o, x), axis=1), self.n_h))

                output, state2 = self.lstm(zyx_phi, state)
                # prediction_t = tf.nn.softmax(dec_x)
        player_encoding = z_q_r
        player_prior = z_prior_r

        cell_output = tf.concat(
            values=(self.d_mus[2], self.d_sigmas[2], self.d_mus[1], self.d_sigmas[1], self.d_mus[0], self.d_sigmas[0],
                    self.d_mus[2], self.d_sigmas[2], self.p_mus[1], self.p_sigmas[1], self.p_mus[0], self.p_sigmas[0],
                    recon_output, player_encoding, player_prior), axis=1)
        return cell_output, state2


class VaRLEA():
    def __init__(self, config, train_flag, extra_prediction_flag=False, deterministic_decoder=False):
        self.extra_prediction_flag = extra_prediction_flag
        self.win_score_diff = True
        self.predict_action_goal = True
        self.config = config
        self.train_flag = train_flag
        self.target_data_ph = tf.placeholder(dtype=tf.float32,
                                             shape=[None, self.config.Learn.max_seq_length,
                                                    self.config.Arch.CLVRNN.x_dim], name='target_data')
        self.input_data_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.config.Learn.max_seq_length,
                                                   self.config.Arch.CLVRNN.x_dim + self.config.Arch.CLVRNN.y_s_dim +
                                                   self.config.Arch.CLVRNN.y_a_dim + self.config.Arch.CLVRNN.y_r_dim + 1],
                                            name='input_data')

        self.selection_matrix_ph = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, self.config.Learn.max_seq_length],
                                                  name='selection_matrix')
        self.sarsa_target_ph = tf.placeholder(dtype=tf.float32,
                                              shape=[None, 3], name='sarsa_target')

        self.trace_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='trace_length')

        self.score_diff_target_ph = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, 3], name='win_target')

        self.action_pred_target_ph = tf.placeholder(dtype=tf.float32,
                                                    shape=[None, self.config.Arch.Predict.output_size],
                                                    name='action_predict')

        self.kld_beta = tf.placeholder(dtype=tf.float32, shape=[], name='kld_beta')

        self.deterministic_decoder = deterministic_decoder

        embed_dim = self.config.Arch.CLVRNN.latent_r_dim

        self.cell_output_dim_list = [self.config.Arch.CLVRNN.latent_s_dim, self.config.Arch.CLVRNN.latent_s_dim,
                                     self.config.Arch.CLVRNN.latent_a_dim, self.config.Arch.CLVRNN.latent_a_dim,
                                     self.config.Arch.CLVRNN.latent_r_dim, self.config.Arch.CLVRNN.latent_r_dim,
                                     self.config.Arch.CLVRNN.latent_s_dim, self.config.Arch.CLVRNN.latent_s_dim,
                                     self.config.Arch.CLVRNN.latent_a_dim, self.config.Arch.CLVRNN.latent_a_dim,
                                     self.config.Arch.CLVRNN.latent_r_dim, self.config.Arch.CLVRNN.latent_r_dim,
                                     self.config.Arch.CLVRNN.x_dim, embed_dim, embed_dim
                                     ]
        self.cell_output_names = ["dec_mu_2", "dec_sigma_2", "dec_mu_1", "dec_sigma_1", "dec_mu_0", "dec_sigma_0",
                                  "p_mu_2", "p_sigma_2", "p_mu_1", "p_sigma_1", "p_mu_0", "p_sigma_0", "x_recon",
                                  "z_posterior", "z_prior"]

        self.score_diff_lstm_cell = []
        self.action_lstm_cell = []
        self.sarsa_lstm_cell = []
        self.build_lstm_models()

    def build_lstm_models(self):

        with tf.name_scope("win"):
            with tf.name_scope("LSTM-layer"):
                for i in range(self.config.Arch.WIN.lstm_layer_num):
                    self.score_diff_lstm_cell.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.WIN.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))

        with tf.name_scope("prediction"):
            with tf.name_scope("LSTM-layer"):
                for i in range(self.config.Arch.Predict.lstm_layer_num):
                    self.action_lstm_cell.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.Predict.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))
        if self.config.Learn.add_sarsa:
            with tf.name_scope("sarsa"):
                with tf.name_scope("LSTM-layer"):
                    for i in range(self.config.Arch.SARSA.lstm_layer_num):
                        self.sarsa_lstm_cell.append(
                            tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.SARSA.h_size, state_is_tuple=True,
                                                    initializer=tf.random_uniform_initializer(-0.05, 0.05)))

    # @property
    def __call__(self):
        def tf_cross_entropy(ce_output, ce_target, condition, if_last_output):
            with tf.variable_scope('win_cross_entropy'):
                ce_loss_all = tf.losses.softmax_cross_entropy(onehot_labels=ce_target,
                                                              logits=ce_output, reduction=tf.losses.Reduction.NONE)
                zero_loss_all = tf.zeros(shape=[tf.shape(ce_loss_all)[0]])
                if if_last_output:
                    return ce_loss_all
                else:
                    return tf.where(condition=condition, x=ce_loss_all, y=zero_loss_all)

        def tf_score_diff(win_output, target_diff, condition, if_last_output):
            with tf.variable_scope('mean_difference'):
                square_diff_loss_all = tf.square(target_diff - win_output)
                abs_diff_loss_all = tf.abs(target_diff - win_output)
                zero_loss_all = tf.zeros(shape=[tf.shape(square_diff_loss_all)[0]])
                if if_last_output:
                    return square_diff_loss_all, abs_diff_loss_all
                else:
                    return tf.where(condition=condition, x=square_diff_loss_all, y=zero_loss_all), \
                           tf.where(condition=condition, x=abs_diff_loss_all, y=zero_loss_all)

        def tf_clvrnn_cross_entropy(target_x, dec_x, condition):
            with tf.variable_scope('cross_entropy'):
                ce_loss_all = tf.losses.softmax_cross_entropy(onehot_labels=target_x,
                                                              logits=dec_x, reduction=tf.losses.Reduction.NONE)

                zero_loss_all = tf.zeros(shape=[tf.shape(ce_loss_all)[0]])
                return tf.where(condition=condition, x=ce_loss_all, y=zero_loss_all)

        def tf_kl_gaussian(mu_1, sigma_1, mu_2, sigma_2, condition):
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            with tf.variable_scope("kl_gaussian"):
                kl_loss_all = tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                    - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                    + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1
                ), 1)
                zero_loss_all = tf.zeros(shape=[tf.shape(kl_loss_all)[0]])

                return tf.where(condition=condition, x=kl_loss_all, y=zero_loss_all)

        def get_clvrnn_lossfunc(dec_mu_2, dec_sigma_2, dec_mu_1, dec_sigma_1, dec_mu_0, dec_sigma_0,
                                p_mu_2, p_sigma_2, p_mu_1, p_sigma_1, p_mu_0, p_sigma_0,
                                x_recon, prior_recon, target_x, condition):
            likelihood_loss = tf_clvrnn_cross_entropy(dec_x=x_recon, target_x=target_x, condition=condition)
            if prior_recon is not None:
                likelihood_loss_prior = tf_clvrnn_cross_entropy(dec_x=prior_recon, target_x=target_x,
                                                                condition=condition)
                likelihood_loss = likelihood_loss + likelihood_loss_prior
            kl_loss_l2 = tf_kl_gaussian(dec_mu_2, dec_sigma_2, p_mu_2, p_sigma_2, condition)
            kl_loss_l1 = tf_kl_gaussian(dec_mu_1, dec_sigma_1, p_mu_1, p_sigma_1, condition)
            kl_loss_l0 = tf_kl_gaussian(dec_mu_0, dec_sigma_0, p_mu_0, p_sigma_0, condition)

            # kl_loss = tf.zeros(shape=[tf.shape(kl_loss)[0]])  # TODO: why if we only optimize likelihood_loss
            return kl_loss_l0, kl_loss_l1, kl_loss_l2, likelihood_loss

        def get_diff_lossfunc(diff_output, diff_target_ph, condition, if_last_output):
            square_diff_loss, abs_diff_loss = tf_score_diff(diff_output, diff_target_ph, condition, if_last_output)
            return square_diff_loss, abs_diff_loss

        def get_action_pred_lossfunc(action_pred_output, action_pred_target_ph, condition, if_last_output):
            action_pred_loss = tf_cross_entropy(action_pred_output, action_pred_target_ph, condition, if_last_output)
            return action_pred_loss

        def tf_td_loss(sarsa_output, sarsa_target_ph, condition, if_last_output):
            with tf.variable_scope('n2_loss'):
                td_loss_all = tf.reduce_mean(tf.square(sarsa_output - sarsa_target_ph), axis=-1)
                zero_loss_all = tf.zeros(shape=[tf.shape(td_loss_all)[0]])
                if if_last_output:
                    return td_loss_all
                else:
                    return tf.where(condition=condition, x=td_loss_all, y=zero_loss_all)

        def tf_td_diff(sarsa_output, sarsa_target_ph, condition, if_last_output):
            with tf.variable_scope('n1_loss'):
                td_loss_all = tf.reduce_mean(tf.abs(sarsa_output - sarsa_target_ph), axis=-1)
                zero_loss_all = tf.zeros(shape=[tf.shape(td_loss_all)[0]])
                if if_last_output:
                    return td_loss_all
                else:
                    return tf.where(condition=condition, x=td_loss_all, y=zero_loss_all)

        def get_td_lossfunc(sarsa_output, sarsa_target_ph, condition, if_last_output):
            td_loss = tf_td_loss(sarsa_output, sarsa_target_ph, condition, if_last_output=if_last_output)
            td_diff = tf_td_diff(sarsa_output, sarsa_target_ph, condition, if_last_output=if_last_output)
            return td_loss, td_diff

        # self.args = args
        # if sample:
        #     args.batch_size = 1
        #     args.seq_length = 1
        batch_size = tf.shape(self.input_data_ph)[0]
        with tf.variable_scope('clvrnn'):

            self.cell = LadderVariationalRNNCell(config=self.config, train_flag=self.train_flag)

            self.initial_state_c, self.initial_state_h = self.cell.zero_state(
                batch_size=tf.shape(self.input_data_ph)[0],
                dtype=tf.float32)

            # if not self.config.Learn.rnn_skip_player:
            #     self.initial_state_c = tf.concat([tf.ones([tf.shape(self.initial_state_c)[0],
            #                                                self.config.Arch.CVRNN.x_dim]),
            #                                       self.initial_state_c], axis=1)

            flat_target_data = tf.reshape(self.target_data_ph, [-1, self.config.Arch.CLVRNN.x_dim])
            clvrnn_outputs, last_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.input_data_ph,
                                                           sequence_length=self.trace_length_ph,
                                                           initial_state=tf.contrib.rnn.LSTMStateTuple(
                                                               self.initial_state_c,
                                                               self.initial_state_h))

        # print outputs
        # outputs = map(tf.pack,zip(*outputs))
        clvrnn_outputs = tf.split(value=tf.transpose(a=clvrnn_outputs, perm=[1, 0, 2]),
                                  num_or_size_splits=[1] * self.config.Learn.max_seq_length, axis=0)
        outputs_reshape = []
        outputs_all = []
        for output in clvrnn_outputs:
            output = tf.squeeze(output, axis=0)
            output = tf.split(value=output, num_or_size_splits=self.cell.output_dim_list, axis=1)
            outputs_all.append(output)

        for n, name in enumerate(self.cell_output_names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs_all])
                x = tf.transpose(x, [1, 0, 2])
                x = tf.reshape(x, [batch_size * self.config.Learn.max_seq_length, self.cell_output_dim_list[n]])
                outputs_reshape.append(x)

        prior_recon = None
        self.prior_recon_output = None
        (self.dec_mu_2, self.dec_sigma_2, self.dec_mu_1, self.dec_sigma_1, self.dec_mu_0, self.dec_sigma_0,
         self.p_mu_2, self.p_sigma_2, self.p_mu_1, self.p_sigma_1, self.p_mu_0, self.p_sigma_0, self.x_recon,
         z_posterior, z_prior) = outputs_reshape

        # self.player_embedding = self.enc_mu

        if self.config.Learn.embed_mode == 'random':
            self.player_embedding = z_posterior
            print('embedding mode is random')
            embed_shape = [batch_size, self.config.Learn.max_seq_length, self.config.Arch.CLVRNN.latent_r_dim]
        elif self.config.Learn.embed_mode == 'mean':
            self.player_embedding = self.dec_mu_0
            print('embedding mode is mean')
            embed_shape = [batch_size, self.config.Learn.max_seq_length, self.config.Arch.CLVRNN.latent_r_dim]
        elif self.config.Learn.embed_mode == 'mean_var':
            self.player_embedding = tf.concat([self.dec_mu_0, self.dec_sigma_0], axis=1)
            print('embedding mode is mean_var')
            embed_shape = [batch_size, self.config.Learn.max_seq_length, self.config.Arch.CLVRNN.latent_r_dim * 2]

        self.select_index = tf.range(0, batch_size) * self.config.Learn.max_seq_length + (self.trace_length_ph - 1)
        # self.z_encoder_output = tf.gather(tf.concat([self.dec_mu_0, self.dec_mu_1], axis=1), self.select_index)
        self.z_encoder_output = tf.gather(z_posterior, self.select_index)
        self.z_prior_output = tf.gather(z_prior, self.select_index)
        # self.z_encoder_output = tf.reshape(
        #     tf.reshape(self.z_encoder, shape=[batch_size, self.config.Learn.max_seq_length,
        #                                       self.config.Arch.CVRNN.latent_dim]), shape=[batch_size, -1])

        self.final_state_c, self.final_state_h = last_state

        condition = tf.cast(tf.reshape(self.selection_matrix_ph,
                                       shape=[tf.shape(self.selection_matrix_ph)[0] *
                                              tf.shape(self.selection_matrix_ph)[1]]), tf.bool)

        self.output = tf.reshape(tf.nn.softmax(self.x_recon),
                                 shape=[batch_size, tf.shape(self.input_data_ph)[1], -1])
        if prior_recon is not None:
            self.prior_recon_output = tf.reshape(tf.nn.softmax(prior_recon),
                                                 shape=[batch_size, tf.shape(self.input_data_ph)[1], -1])

        kl_loss_l0, kl_loss_l1, kl_loss_l2, likelihood_loss = get_clvrnn_lossfunc(self.dec_mu_2, self.dec_sigma_2,
                                                                                  self.dec_mu_1, self.dec_sigma_1,
                                                                                  self.dec_mu_0, self.dec_sigma_0,
                                                                                  self.p_mu_2, self.p_sigma_2,
                                                                                  self.p_mu_1, self.p_sigma_1,
                                                                                  self.p_mu_0, self.p_sigma_0,
                                                                                  self.x_recon, prior_recon,
                                                                                  flat_target_data, condition)
        if self.config.Learn.skip_first_kl:
            kl_loss = kl_loss_l0
        else:
            kl_loss = kl_loss_l0 + kl_loss_l1 + kl_loss_l2

        with tf.variable_scope('clvrnn_cost'):
            self.kl_loss = tf.reshape(kl_loss, shape=[batch_size, self.config.Learn.max_seq_length, -1])
            self.likelihood_loss = tf.reshape(likelihood_loss, shape=[batch_size, self.config.Learn.max_seq_length, -1])

        tvars_clvrnn = tf.trainable_variables(scope='clvrnn')
        for t in tvars_clvrnn:
            print ('clvrnn_var: ' + str(t.name))
        clvrnn_grads = tf.gradients(tf.reduce_mean(self.likelihood_loss + self.kld_beta * self.kl_loss), tvars_clvrnn)
        ll_grads = tf.gradients(tf.reduce_mean(self.likelihood_loss), tvars_clvrnn)
        # grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.config.Learn.learning_rate)
        self.train_ll_op = optimizer.apply_gradients(zip(ll_grads, tvars_clvrnn))
        self.train_general_op = optimizer.apply_gradients(zip(clvrnn_grads, tvars_clvrnn))
        # self.saver = tf.train.Saver(tf.all_variables())


        if self.win_score_diff:
            with tf.variable_scope('score_diff'):
                data_input_action_pred = self.input_data_ph[:, :,
                                         self.config.Arch.CLVRNN.x_dim:self.config.Arch.CLVRNN.y_s_dim +
                                                                       self.config.Arch.CLVRNN.y_r_dim +
                                                                       self.config.Arch.CLVRNN.y_a_dim +
                                                                       self.config.Arch.CLVRNN.x_dim]

                z_encoder_score_diff = tf.reshape(self.player_embedding, shape=embed_shape)
                for i in range(self.config.Arch.WIN.lstm_layer_num):
                    rnn_output = None
                    for i in range(self.config.Arch.WIN.lstm_layer_num):
                        rnn_input = tf.concat([data_input_action_pred, z_encoder_score_diff],
                                              axis=2) if i == 0 else rnn_output
                        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                            inputs=rnn_input, cell=self.score_diff_lstm_cell[i],
                            sequence_length=self.trace_length_ph, dtype=tf.float32,
                            scope='score_diff_rnn_{0}'.format(str(i)))
                    action_pred_rnn_outputs = tf.stack(rnn_output)
                    # Indexing
                    score_diff_rnn_last = tf.gather(tf.reshape(action_pred_rnn_outputs,
                                                               [-1, self.config.Arch.SARSA.h_size]), self.select_index)

                for j in range(self.config.Arch.WIN.dense_layer_number - 1):
                    score_diff_input = score_diff_rnn_last if j == 0 else score_diff_output
                    score_diff_output = tf.nn.relu(
                        linear(score_diff_input, output_size=self.config.Arch.WIN.dense_layer_size,
                               scope='win_dense_Linear'))
                score_diff_input = score_diff_rnn_last if self.config.Arch.WIN.dense_layer_number == 1 else score_diff_output
                score_diff_output = linear(score_diff_input, output_size=3, scope='score_diff')
                # self.diff_output = tf.nn.softmax(diff_output)
                self.diff_output = score_diff_output

                with tf.variable_scope('score_diff_cost'):
                    square_diff_loss, abs_diff_loss = get_diff_lossfunc(self.diff_output, self.score_diff_target_ph,
                                                                        condition,
                                                                        if_last_output=True)
                    self.diff_loss = square_diff_loss
                    self.diff = abs_diff_loss

                    # self.win_acc, win_acc_op = tf.metrics.accuracy(labels=tf.argmax(self.win_target_ph, 1),
                    #                                                predictions=tf.argmax(self.win_output, 1))
            if self.config.Learn.integral_update_flag:
                tvars_score_diff = tf.trainable_variables()
            else:
                tvars_score_diff = tf.trainable_variables(scope='score_diff')
            for t in tvars_score_diff:
                print ('tvars_score_diff: ' + str(t.name))
            score_diff_grads = tf.gradients(tf.reduce_mean(self.diff_loss), tvars_score_diff)
            self.train_diff_op = optimizer.apply_gradients(zip(score_diff_grads, tvars_score_diff))

        if self.extra_prediction_flag:
            with tf.variable_scope('prediction'):
                data_input_action_pred = self.input_data_ph[
                                         :, :,
                                         self.config.Arch.CLVRNN.x_dim:self.config.Arch.CLVRNN.y_a_dim +
                                                                       self.config.Arch.CLVRNN.y_s_dim +
                                                                       self.config.Arch.CLVRNN.x_dim]

                z_encoder_action_pred = tf.reshape(self.player_embedding, shape=embed_shape)
                for i in range(self.config.Arch.Predict.lstm_layer_num):
                    rnn_output = None
                    for i in range(self.config.Arch.Predict.lstm_layer_num):
                        rnn_input = tf.concat([data_input_action_pred, z_encoder_action_pred],
                                              axis=2) if i == 0 else rnn_output
                        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                            inputs=rnn_input, cell=self.action_lstm_cell[i],
                            sequence_length=self.trace_length_ph, dtype=tf.float32,
                            scope='action_pred_rnn_{0}'.format(str(i)))
                    action_pred_rnn_outputs = tf.stack(rnn_output)
                    # Indexing
                    action_pred_rnn_last = tf.gather(tf.reshape(action_pred_rnn_outputs,
                                                                [-1, self.config.Arch.SARSA.h_size]), self.select_index)

                for j in range(self.config.Arch.Predict.dense_layer_number - 1):
                    action_pred_input = action_pred_rnn_last if j == 0 else action_pred_output
                    action_pred_output = tf.nn.relu(linear(action_pred_input,
                                                           output_size=self.config.Arch.Predict.dense_layer_size,
                                                           scope='action_dense_Linear'))
                action_pred_input = action_pred_rnn_last if self.config.Arch.Predict.dense_layer_number == 1 else action_pred_output
                action_pred_output = linear(action_pred_input, output_size=self.config.Arch.Predict.output_size,
                                            scope='action_next')
                # self.diff_output = tf.nn.softmax(diff_output)
                self.action_pred_output = tf.nn.softmax(action_pred_output)

                with tf.variable_scope('action_pred_cost'):
                    action_pred_loss = get_action_pred_lossfunc(self.action_pred_output,
                                                                self.action_pred_target_ph,
                                                                condition,
                                                                if_last_output=True)
                    self.action_pred_loss = action_pred_loss

                    # self.win_acc, win_acc_op = tf.metrics.accuracy(labels=tf.argmax(self.win_target_ph, 1),
                    #                                                predictions=tf.argmax(self.win_output, 1))
            if self.config.Learn.integral_update_flag:
                tvars_action_pred = tf.trainable_variables()

            else:
                tvars_action_pred = tf.trainable_variables(scope='prediction')
            for t in tvars_action_pred:
                print ('tvars_action_pred: ' + str(t.name))
            action_grads = tf.gradients(tf.reduce_mean(self.action_pred_loss), tvars_action_pred)
            self.train_action_pred_op = optimizer.apply_gradients(zip(action_grads, tvars_action_pred))

            # if
            # def sample(self, sess, args, num=4410, start=None):
            #
            #     def sample_gaussian(mu, sigma):
            #         return mu + (sigma * np.random.randn(*sigma.shape))
            #
            #     if start is None:
            #         prev_x = np.random.randn(1, 1, 2 * args.chunk_samples)
            #     elif len(start.shape) == 1:
            #         prev_x = start[np.newaxis, np.newaxis, :]
            #     elif len(start.shape) == 2:
            #         for i in range(start.shape[0] - 1):
            #             prev_x = start[i, :]
            #             prev_x = prev_x[np.newaxis, np.newaxis, :]
            #             feed = {self.input_data_ph: prev_x,
            #                     self.initial_state_c: prev_state[0],
            #                     self.initial_state_h: prev_state[1]}
            #
            #             [o_mu, o_sigma, o_rho, prev_state_c, prev_state_h] = sess.run(
            #                 [self.dec_mu, self.dec_sigma, self.dec_x,
            #                  self.final_state_c, self.final_state_h], feed)
            #
            #         prev_x = start[-1, :]
            #         prev_x = prev_x[np.newaxis, np.newaxis, :]
            #
            #     prev_state = sess.run(self.cell.zero_state(1, tf.float32))
            #     chunks = np.zeros((num, 2 * args.chunk_samples), dtype=np.float32)
            #     mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
            #     sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)
            #
            #     for i in xrange(num):
            #         feed = {self.input_data_ph: prev_x,
            #                 self.initial_state_c: prev_state[0],
            #                 self.initial_state_h: prev_state[1]}
            #         [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.dec_mu, self.dec_sigma,
            #                                                                        self.dec_x, self.final_state_c,
            #                                                                        self.final_state_h], feed)
            #
            #         next_x = np.hstack((sample_gaussian(o_mu, o_sigma),
            #                             2. * (o_rho > np.random.random(o_rho.shape[:2])) - 1.))
            #         chunks[i] = next_x
            #         mus[i] = o_mu
            #         sigmas[i] = o_sigma
            #
            #         prev_x = np.zeros((1, 1, 2 * args.chunk_samples), dtype=np.float32)
            #         prev_x[0][0] = next_x
            #         prev_state = next_state_c, next_state_h
            #
            #     return chunks, mus, sigmas

        if self.config.Learn.add_sarsa:
            with tf.variable_scope('sarsa'):
                data_input_sarsa = self.input_data_ph[
                                   :, :,
                                   self.config.Arch.CLVRNN.x_dim:self.config.Arch.CLVRNN.y_a_dim +
                                                                 self.config.Arch.CLVRNN.y_s_dim +
                                                                 self.config.Arch.CLVRNN.x_dim]

                z_encoder_sarsa = tf.reshape(self.player_embedding, shape=embed_shape)

                # z_encoder_last = tf.gather(z_encoder, self.select_index)
                # self.z_encoder_last = z_encoder_last
                # sarsa_y_last = tf.gather(data_input_sarsa, self.select_index)
                # self.sarsa_y_last = sarsa_y_last

                for i in range(self.config.Arch.SARSA.lstm_layer_num):
                    rnn_output = None
                    for i in range(self.config.Arch.SARSA.lstm_layer_num):
                        rnn_input = tf.concat([data_input_sarsa, z_encoder_sarsa], axis=2) if i == 0 else rnn_output
                        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                            inputs=rnn_input, cell=self.sarsa_lstm_cell[i],
                            sequence_length=self.trace_length_ph, dtype=tf.float32,
                            scope='sarsa_rnn_{0}'.format(str(i)))
                    cvrnn_outputs = tf.stack(rnn_output)
                    # Indexing
                    rnn_last = tf.gather(tf.reshape(cvrnn_outputs,
                                                    [-1, self.config.Arch.SARSA.h_size]), self.select_index)

                for j in range(self.config.Arch.SARSA.dense_layer_number - 1):
                    sarsa_input = rnn_last if j == 0 else sarsa_output
                    sarsa_output = tf.nn.relu(linear(sarsa_input, output_size=self.config.Arch.SARSA.dense_layer_size,
                                                     scope='dense_Linear'))
                sarsa_input = rnn_last if self.config.Arch.SARSA.dense_layer_number == 1 else sarsa_output
                sarsa_output = linear(sarsa_input, output_size=3, scope='output_Linear')
                self.sarsa_output = tf.nn.softmax(sarsa_output)

            with tf.variable_scope('td_cost'):
                td_loss, td_diff = get_td_lossfunc(self.sarsa_output, self.sarsa_target_ph, condition,
                                                   if_last_output=True)
                # self.td_loss = tf.reshape(td_loss, shape=[tf.shape(self.input_data_ph)[0],
                #                                           self.config.Learn.max_seq_length, -1])
                self.td_loss = td_loss
                self.td_avg_diff = tf.reduce_mean(td_diff)
            if self.config.Learn.integral_update_flag:
                tvars_td = tf.trainable_variables()
            else:
                tvars_td = tf.trainable_variables(scope='sarsa')
            for t in tvars_td:
                print ('td_var: ' + str(t.name))
            td_grads = tf.gradients(tf.reduce_mean(self.td_loss), tvars_td)
            self.train_td_op = optimizer.apply_gradients(zip(td_grads, tvars_td))


