#!/usr/bin/env python2
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class RecursiveAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
		'''\
        Initialization

        Args:
            n_input: wordvector size * 2
            n_hidden: hidden unit size
            transer_function: ?
		'''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x1 = tf.placeholder(tf.float32, [None, self.n_input / 2])
        self.x2 = tf.placeholder(tf.float32, [None, self.n_input / 2])
        self.x3 = tf.placeholder(tf.float32, [None, self.n_input / 2])
        self.y1 = self.transfer(tf.add(tf.matmul(tf.concat(1, [self.x2, self.x3]),
                                                 self.weights['w1']), self.weights['b1']))
        self.y2 = self.transfer(tf.add(tf.matmul(tf.concat(1, [self.x1, self.y1]),
                                                 self.weights['w1']), self.weights['b1']))
        self.x1prime, self.y1prime = tf.split(1, self.n_input/2,
                                              tf.add(tf.matmul(self.y2, self.weights['w2']), self.weights['b2'])
        self.x2prime, self.x3prime = tf.split(1, self.n_input/2, self.y1prime)

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(tf.concat(1, [self.x1, self.x2, self.x3]),
                                                      tf.concat(1, [self.x1prime, self.x2prime, self.x3prime]), 2.0)))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, x1, x2, x3):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x1: x1, self.x2: x2, self.x3: x3})
        return cost

    def calc_total_cost(self, x1, x2, x3):
        return self.sess.run(self.cost, feed_dict={self.x1: x1, self.x2: x2, self.x3: x3})

    def transform(self, x1, x2, x3):
        return self.sess.run(self.y2, feed_dict={self.x1: x1, self.x2: x2, self.x3: x3})

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

# vim: ts=4 sw=4 sts=4 expandtab
