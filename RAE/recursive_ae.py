#!/usr/bin/env python2
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class RecursiveAutoencoder(object):

    def __init__(self, embed_size, vocab_size, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        '''\
        Initialization

        Args:
            n_input: wordvector size * 2
            n_hidden: hidden unit size
            transer_function: ?
        '''
        self.n_input = embed_size * 2
        self.n_hidden = embed_size
        self.transfer = transfer_function
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.word_embedding_dict = tf.Variable(tf.random_uniform([vocab_size+10, embed_size], -1.0, 1.0) ,name="dictionary")
        self.x1_index = tf.placeholder(tf.int32, [None])
        self.x2_index = tf.placeholder(tf.int32, [None])
        self.x3_index = tf.placeholder(tf.int32, [None])
        self.x1 = tf.nn.embedding_lookup(self.word_embedding_dict, self.x1_index)
        self.x2 = tf.nn.embedding_lookup(self.word_embedding_dict, self.x2_index)
        self.x3 = tf.nn.embedding_lookup(self.word_embedding_dict, self.x3_index)
        tf.concat(1, [self.x2, self.x3])
        self.y1 = tf.add(tf.matmul(tf.concat(1, [self.x2, self.x3]), self.weights['w1']), self.weights['b1'])
        self.y2 = tf.add(tf.matmul(tf.concat(1, [self.x1, self.y1]), self.weights['w1']), self.weights['b1'])
        self.x1prime, self.y1prime = tf.split(1, 2,
                                              tf.add(tf.matmul(self.y2, self.weights['w2']), self.weights['b2']))
        self.x2prime, self.x3prime = tf.split(1, 2,
                                              tf.add(tf.matmul(self.y1prime, self.weights['w2']), self.weights['b2']))

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(tf.concat(1, [self.x1, self.x2, self.x3]),
                                                      tf.concat(1, [self.x1prime, self.x2prime, self.x3prime])), 2.0))
        vars = tf.trainable_variables()
        self.optimizer = optimizer.minimize(self.cost, var_list=vars)
        # grads = tf.gradients(self.cost, vars)
        # self.optimizer = optimizer.apply_gradients(zip(grads,vars))

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.random_uniform([self.n_input, self.n_hidden], dtype=tf.float32))
        all_weights['b1'] = tf.Variable(tf.random_uniform([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.random_uniform([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.random_uniform([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, x1, x2, x3):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x1_index: x1, self.x2_index: x2, self.x3_index: x3})
        return cost

    def calc_total_cost(self, x1, x2, x3):
        return self.sess.run(self.cost, feed_dict={self.x1_index: x1, self.x2_index: x2, self.x3_index: x3})

    def transform(self, x1, x2, x3):
        return self.sess.run(self.y2, feed_dict={self.x1_index: x1, self.x2_index: x2, self.x3_index: x3})

    def ret_dict(self):
        return self.sess.run((self.word_embedding_dict, self.weights['w1']))

#    def generate(self, hidden=None):
#        if hidden is None:
#            hidden = np.random.normal(size=self.weights["b1"])
#        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
#
#    def reconstruct(self, X):
#        return self.sess.run(self.reconstruction, feed_dict={self.x: X})
#
#    def getWeights(self):
#        return self.sess.run(self.weights['w1'])
#
#    def getBiases(self):
#        return self.sess.run(self.weights['b1'])

if __name__ == '__main__':
    rae = RecursiveAutoencoder(100)

# vim: ts=4 sw=4 sts=4 expandtab
