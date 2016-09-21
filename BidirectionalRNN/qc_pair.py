import config
import numpy as np

class QCPair:
    def __init__(self, query, question, label):
        self.query = query
        self.question = question
        self.label = label
        self.q_list = np.zeros(config.max_sentence_len, dtype='float32')
        self.c_list = np.zeros(config.max_sentence_len, dtype='float32')

