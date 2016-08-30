#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-
import sys

import tensorflow as tf
import config
import utils
import recursive_ae

reload(sys)
sys.setdefaultencoding('utf-8')

word_count_map = utils.read_file(sys.argv[1])
word_index_map = utils.map_truncate(word_count_map, config.vocab_size)

model = recursive_ae.RecursiveAutoencoder(config.embedding_size, config.vocab_size)

f = open(sys.argv[1], 'r')
batch_count = 0
x1 = []
x2 = []
x3 = []

see_dict, _ = model.ret_dict()
line_count = 0
for lines in f:
    line_count += 1
    if line_count % 5000 == 0:
        print line_count
    token = lines.decode('utf-8').encode('utf-8').split(' ')
    if len(token) <= 2:
        continue
    else:
        for i in range(2, len(token)):
            # token[i] = token[i].decode('utf-8').encode('utf-8')
            # token[i-1] = token[i-1].decode('utf-8').encode('utf-8')
            # token[i-2] = token[i-2].decode('utf-8').encode('utf-8')
            x1tmp, x2tmp, x3tmp = utils.map_lookup(word_index_map, token[i-2], token[i-1], token[i])
            x1.append(x1tmp)
            x2.append(x2tmp)
            x3.append(x3tmp)
            batch_count += 1
            if batch_count != 0 and batch_count % config.batch_size == 0:
                batch_count = 0
                model.partial_fit(x1, x2, x3)
                x1 = []
                x2 = []
                x3 = []

see_dict, _ = model.ret_dict()

utils.save_dictionary(see_dict, word_index_map, config.embedding_size)
utils.save_model_parameter(model)

# vim: ts=4 sw=4 sts=4 expandtab
