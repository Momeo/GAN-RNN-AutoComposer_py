import tensorflow as tf
import numpy as np
import os

from modules.midiconnector import MidiConnector

class Composer:

    def __init__(self):
        self.songs = None
        self.note_range = None
        self.n_timesteps = 128

        self.n_input = None
        self.n_hidden = 64
        self.X = None
        self.W = None
        self.bh = None
        self.bv = None

    def main(self, args=None):
        print('Composer start')
        self.songs = MidiConnector.get_songs('midi')
        self.note_range = MidiConnector.span
        self.n_input = 2 * self.note_range * self.n_timesteps
        self.X = tf.placeholder(tf.float32, [None, self.n_input])

        self.train_neural_network()

    def sample(self, probs):
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


    def gibbs_sample(self, k):
        def body(count, k, xk):
            hk = self.sample(tf.sigmoid(tf.matmul(xk, self.W) + self.bh))
            xk = self.sample(tf.sigmoid(tf.matmul(hk, tf.transpose(self.W)) + self.bv))
            return count + 1, k, xk

        count = tf.constant(0)

        def condition(count, k, xk):
            return count < k

        [_, _, x_sample] = tf.while_loop(condition, body, [count, tf.constant(k), self.X])

        x_sample = tf.stop_gradient(x_sample)
        return x_sample


    # 定义神经网络
    def neural_network(self):
        self.W = tf.Variable(tf.random_normal([self.n_input, self.n_hidden], 0.01))
        self.bh = tf.Variable(tf.zeros([1, self.n_hidden], tf.float32))
        self.bv = tf.Variable(tf.zeros([1, self.n_input], tf.float32))
        x_sample = self.gibbs_sample(1)
        h = self.sample(tf.sigmoid(tf.matmul(self.X, self.W) + self.bh))
        h_sample = self.sample(tf.sigmoid(tf.matmul(x_sample, self.W) + self.bh))

        learning_rate = tf.constant(0.005, tf.float32)
        size_bt = tf.cast(tf.shape(self.X)[0], tf.float32)
        W_adder = tf.multiply(learning_rate / size_bt,
                              tf.subtract(tf.matmul(tf.transpose(self.X), h), tf.matmul(tf.transpose(x_sample), h_sample)))
        bv_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(self.X, x_sample), 0, True))
        bh_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
        update = [self.W.assign_add(W_adder), self.bv.assign_add(bv_adder), self.bh.assign_add(bh_adder)]
        return update


    # 训练神经网络
    def train_neural_network(self):
        update = self.neural_network()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.all_variables())

            epochs = 256
            batch_size = 64
            for epoch in range(epochs):
                for song in self.songs:
                    song = np.array(song)
                    song = song[:int(np.floor(song.shape[0] / self.n_timesteps) * self.n_timesteps)]
                    song = np.reshape(song, [song.shape[0] // self.n_timesteps, song.shape[1] * self.n_timesteps])

                for i in range(1, len(song), batch_size):
                    train_x = song[i:i + batch_size]
                    sess.run(update, feed_dict={self.X: train_x})
                print(epoch)
                # 保存模型
                if epoch == epochs - 1:
                    saver.save(sess, os.path.join(os.getcwd(), 'midi.module'))

            sample = self.gibbs_sample(1).eval(session=sess, feed_dict={self.X: np.zeros((1, self.n_input))})

            S = np.reshape(sample[0, :], (self.n_timesteps, 2 * self.note_range))
            MidiConnector.noteStateMatrixToMidi(S, "auto_gen_music")
            print('生成auto_gen_music.mid文件')




