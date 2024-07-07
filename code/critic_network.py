import tensorflow as tf
import numpy as np

class CriticNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, num_quantiles):
        super(CriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(embedding_dim, 3 * embedding_dim))
        self.fc1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tf.keras.layers.Dense(num_quantiles, activation='linear')

    def call(self, inputs):
        state, action = inputs
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        quantiles = self.out(x)
        return quantiles

class Critic(object):
    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau, num_quantiles):
        self.embedding_dim = embedding_dim
        self.num_quantiles = num_quantiles
        self.quantile_tau = self.initialize_quantile_tau(num_quantiles)

        self.network = CriticNetwork(embedding_dim, hidden_dim, num_quantiles)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim, num_quantiles)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
        self.loss = self.quantile_huber_loss
        self.tau = tau

    @staticmethod
    #여기서 quantile_tau는 quantile midpoint
    def initialize_quantile_tau(num_quantiles):
        quantile_tau = (tf.range(num_quantiles, dtype=tf.float32) + 0.5) / num_quantiles
        return tf.expand_dims(quantile_tau, axis=-1)

    @staticmethod
    #quantile huber loss 계산
    def quantile_huber_loss(y_true, y_pred, quantile_tau, delta=1.0):
        err = y_true - y_pred
        quantile_tau = tf.expand_dims(quantile_tau, axis=-1)
        huber_loss = tf.where(tf.abs(err) <= delta,
                              0.5 * tf.square(err),
                              delta * (tf.abs(err) - 0.5 * delta))
        quantile_loss = tf.abs(quantile_tau - tf.cast(err < 0, dtype=tf.float32)) * huber_loss
        loss = tf.reduce_mean(quantile_loss, axis=-1)
        return tf.reduce_mean(loss)

    def build_networks(self):
        self.network([np.zeros((1, self.embedding_dim)), np.zeros((1, 3 * self.embedding_dim))])
        self.target_network([np.zeros((1, self.embedding_dim)), np.zeros((1, 3 * self.embedding_dim))])

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.tau * c_omega[i] + (1 - self.tau) * t_omega[i]
        self.target_network.set_weights(t_omega)

    def dq_da(self, inputs):
        actions, states = inputs
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        states = tf.convert_to_tensor(states, dtype=tf.float32)

        with tf.GradientTape() as g:
            g.watch(actions)
            quantiles = self.network([states, actions], training=True)
            #quantiles: critic network의 출력으로, 정확히는 각 quantile midpoint에 해당하는 support의 위치
            q_values = tf.reduce_mean(quantiles, axis=-1)
            #각 support의 평균값의 gradient
        g_grads = g.gradient(q_values, actions)
        return g_grads

    def train(self, inputs, td_targets, weight_batch, epochs=1):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self.network(inputs, training=True)

                loss = self.quantile_huber_loss(y_true=td_targets, y_pred=y_pred, quantile_tau=self.quantile_tau)
                weighted_loss = tf.reduce_mean(loss * weight_batch)
            gradients = tape.gradient(weighted_loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return weighted_loss

    def save_weights(self, path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)
