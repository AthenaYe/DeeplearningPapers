import tensorflow as tf
import word_reader
import config
import sys
import numpy as np
import random
from tensorflow.python.ops import rnn, rnn_cell

class OutputModel:
    def init(self, wr):
        self.word_reader = wr
        # initializer = tf.constant_initializer(value=wr.word_vectors, dtype=tf.float32)
        # self.word_dict = tf.get_variable('word_dict', shape=[len(wr.word_vectors), config.embedding_size],
        #                                  initializer=initializer, trainable=config.trainable)
        self.word_dict = tf.Variable(initializer=tf.constant(wr.word_vectors),
                                     shape = None, trainable=config.trainable)
        # paraphrase sentences
        self.x1_index = tf.placeholder(tf.int32, [None, config.max_sentence_len])
        self.x1 = tf.nn.embedding_lookup(self.word_dict, self.x1_index)
        # Permuting batch_size and n_steps
        self.x1 = tf.transpose(self.x1, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        self.x1 = tf.reshape(self.x1, [-1, config.embedding_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        self.x1 = tf.split(0, config.max_sentence_len, self.x1)

        self.x2_index = tf.placeholder(tf.int32, [None, config.max_sentence_len])
        self.x2 = tf.nn.embedding_lookup(self.word_dict, self.x2_index)
        # Permuting batch_size and n_steps
        self.x2 = tf.transpose(self.x2, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        self.x2 = tf.reshape(self.x2, [-1, config.embedding_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        self.x2 = tf.split(0, config.max_sentence_len, self.x2)
        # output label
        self.y = tf.placeholder("float", [None, config.classes])

        # Forward direction cell
        lstm_fw_cell_x1 = rnn_cell.LSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell_x1 = rnn_cell.LSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # Forward direction cell
        lstm_fw_cell_x2 = rnn_cell.LSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell_x2 = rnn_cell.LSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # Get lstm cell output
        outputs_x1, _, _ = rnn.bidirectional_rnn(lstm_fw_cell_x1, lstm_bw_cell_x1,
                                                     self.x1, dtype=tf.float32, scope="RNN1")
        outputs_x2, _, _ = rnn.bidirectional_rnn(lstm_fw_cell_x2, lstm_bw_cell_x2,
                                                     self.x2, dtype=tf.float32, scope="RNN2")
        outputs = tf.concat(1, [outputs_x1[-1], outputs_x2[-1]])
        self.weights = tf.Variable(tf.random_uniform([config.hidden_size * 4, config.classes],
                                                dtype=tf.float32))
        mid = tf.matmul(outputs, self.weights)
        self.results = tf.nn.softmax(mid)
        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(self.results, self.y)
        vars = tf.trainable_variables()
        optimizer=tf.train.AdamOptimizer()
        self.optimizer = optimizer.minimize(self.cost, var_list=vars)
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        return

    def train(self, q, c, label):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x1_index: q, self.x2_index: c, self.y : label})
        return cost

    def cal_cost(self, q, c, label):
        cost = self.sess.run((self.cost, self.optimizer),
                             feed_dict={self.x1_index: q, self.x2_index: c, self.y : label})
        return cost

    def predict(self, q, c):
        pred = self.sess.run(self.results,
                             feed_dict={self.x1_index: q, self.x2_index: c})
        x_max = tf.reduce_max(pred, reduction_indices=[0])
        return x_max

    def train_and_test(self, fold_cut, batch_size):
        train = self.word_reader.corpus_set[:fold_cut]
        test = self.word_reader.corpus_set[fold_cut:]
        x1 = []
        x2 = []
        y = []
        batch_count = 0
        cost = 0
        for i in range(0, fold_cut):
            if batch_count == batch_size:
                cost += self.train(x1, x2, y)
                x1 = []
                x2 = []
                y = []
                continue
            batch_count += 1
            x1.append(train[i].q_list)
            x2.append(train[i].c_list)
            y.append(train[i].label)
        right_count = 0
        for i in range(0, len(test)):
            x1 = []
            x1.append(test[i].q_list)
            x2 = []
            x2.append(test[i].c_list)
            ans = self.predict(x1, x2)
            if abs(ans-test[i].label) < 1e-6:
                right_count += 1
        return right_count * 1.0 / len(test), cost


if __name__ == '__main__':
    wr = word_reader.WordReader()
    # read paraphrase file
    wr.read_file(sys.argv[1])
    # read pretraining file
    if len(sys.argv) >= 3 and config.pre_train == True:
        wr.read_file(sys.argv[2])
    wr.corpus_set = random.shuffle(wr.corpus_set)
    model = OutputModel()
    model.init(wr)
    for i in range(0, config.epoch):
        acc, cost = model.train_and_test(config.fold_cut, config.batch_size)
        print "round: " + str(i)
        print "accuracy: " + str(acc)