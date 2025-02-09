import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_action,
            n_feature,
            hidden_layers=None,
            batch_size=512,
            learning_rate=0.01,
            reward_decay=1.0,
            output_graph=False,
            reuse=False
    ):
        self.n_action = n_action
        self.n_feature = n_feature
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.ep_obs, self.ep_algo_action, self.ep_real_action, self.ep_rs = [], [], [], []
        # with tf.device('/device:GPU:0'):
        self._build_net()
        self.losses = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_feature], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value")
            self.tf_lbs = tf.placeholder(
                tf.float32, [None, self.n_action], name="labels_value")

        with tf.variable_scope("policy_net"):
            layer_1 = tf.keras.layers.Dense(
                units=512,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='fc1'
            )(self.tf_obs)
            layer_2 = tf.keras.layers.Dense(
                units=64,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='fc2'
            )(layer_1)
            all_act = tf.keras.layers.Dense(
                units=self.n_action,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, 1),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc3'
            )(layer_1)
            
            # use softmax to convert to probability
            self.all_act_prob_soft = tf.nn.softmax(all_act, name='act_prob')
            self.all_act_prob = tf.clip_by_value(
                self.all_act_prob_soft, 1e-4, 1)

            self.entropy = - \
                tf.reduce_sum(self.all_act_prob *
                              tf.log(self.all_act_prob), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(
                self.entropy, name="entropy_mean")

            self.prob_avg = tf.reduce_mean(self.all_act_prob_soft, axis=0)
            self.loss_avg = tf.abs(self.prob_avg[0] - 0.5)

        with tf.name_scope('loss'):
            self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act, labels=self.tf_acts)  # 所选 action 的概率 -log 值
            self.loss = tf.reduce_mean(
                self.neg_log_prob * self.tf_vt) + self.entropy_mean
            self.train_op = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)

    def choose_action(self, observation, mode='random', greedy=None):
        prob_weights = self.sess.run(self.all_act_prob_soft, feed_dict={
                                     self.tf_obs: observation})
        actions = []
        probs = []
        for i, p in enumerate(prob_weights):
            p /= np.sum(p)
            if mode == 'random':
                action = np.random.choice(self.n_action, p=p)
            elif mode == 'max':
                action = np.argmax(p)
            else:
                print('choose_action mode is Wrong')
                exit()
            actions.append(action)  # select action w.r.t the actions prob
            probs.append(p)
        return actions, probs

    def store_transition(self, state, algo_action, real_action, reward):
        self.ep_obs.append(state)
        self.ep_algo_action.append(algo_action)
        self.ep_rs.append(reward)
        self.ep_real_action.append(real_action)

    def learn(self, tran=None):
        if tran is not None:
            self.ep_obs, self.ep_rs, self.ep_algo_action, self.ep_real_action = tran.all_data()
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        labels = np.ones((len(discounted_ep_rs_norm), self.n_action))
        _, loss, prob_avg, loss_avg = self.sess.run([self.train_op, self.loss, self.prob_avg, self.loss_avg], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_algo_action),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            self.tf_lbs: labels
        })

        self.ep_obs, self.ep_algo_action, self.ep_real_action, self.ep_rs = [], [], [], []  
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self, terminal=None):
        discounted_ep_rs = np.ones_like(self.ep_rs)
        for t in reversed(range(0, len(self.ep_rs))):
            discounted_ep_rs[t] = self.ep_rs[t]
        return discounted_ep_rs
