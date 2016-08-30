#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

import operator
import sys
import tensorflow as tf
from array import array
import struct
import re


reload(sys)
sys.setdefaultencoding('utf-8')

def read_file(file_name):
    word_count_map = {}
    f = open(sys.argv[1], 'r')
    for lines in f:
        token = lines.strip().decode('utf-8').encode('utf-8').split(' ')
        for tmp in token:
            if re.match(u'[^A-Za-z0-9\u4e00-\uf9a5]', tmp.decode('utf-8')) is not None:
                continue
            if word_count_map.has_key(tmp):
                word_count_map[tmp] += 1
            else:
                word_count_map[tmp] = 1
    return word_count_map

def map_truncate(my_map, my_size):
    my_map = sorted(my_map.items(), key=operator.itemgetter(1), reverse=True)
    word_index_map = {}
    # index 'UNK' = 1, totalling vocab_size words including UNK
    word_index_map['UNK'] = 1
    count = 1
    for word, _ in my_map:
        count += 1
        word_index_map[word] = count
        if count == my_size:
            break
    return word_index_map

def map_lookup(my_map, word1, word2, word3):

    def __lookup(my_map, word):
        if my_map.has_key(word):
            return my_map[word]
        return my_map['UNK']
    index1 = __lookup(my_map, word1)
    index2 = __lookup(my_map, word2)
    index3 = __lookup(my_map, word3)
    return index1, index2, index3

def save_model_parameter(model):
    saver = tf.train.Saver()
    save_path = saver.save(model.sess, "model.ckpt")
    return save_path

def save_dictionary(word_dict, word_index_map, embed_size, file_name="wordvector", binary=True):
    if binary:
        file_name += '.bin'
    f = open(file_name, 'wb')
    f.write(str(len(word_index_map)))
    f.write(' ')
    f.write(str(embed_size))
    f.write('\n')
    for word in word_index_map:
        l = word_dict[word_index_map[word]]
        print word
        f.write(word)
        f.write(' ')
        packed = struct.pack('<{}f'.format(embed_size), *l)
        f.write(packed)
        f.write('\n')
    f.close()

# vim: ts=4 sw=4 sts=4 expandtab
