import sys
import re
import numpy as np
import config
import jieba
import qc_pair
import requests
import requests.exceptions
import time

class WordReader:
    word_dict = {}
    word_index = {}
    word_vectors = []
    corpus_set = []
    def read_file(self, file_name):
        reload(sys)
        sys.setdefaultencoding('utf-8')
        self.file_name = file_name
        f = open(file_name, 'r')
        index_count = 3
        self.word_dict['UNK'] = np.array(np.random.uniform(-0.01, 0.01, config.embedding_size),
                                         dtype='float32')
        self.word_dict['EOS'] = np.array(np.random.uniform(-0.01, 0.01, config.embedding_size),
                                         dtype='float32')
        self.word_dict['PAD'] = np.zeros(config.embedding_size, dtype='float32')
        self.word_index['PAD'] = 0
        self.word_index['EOS'] = 1
        self.word_index['UNK'] = 2
        self.word_vectors.append(self.word_dict['PAD'])
        self.word_vectors.append(self.word_dict['EOS'])
        self.word_vectors.append(self.word_dict['UNK'])
        for lines in f:
            if config.chinese == False:
                token = lines.strip().split('\t')
                new_entry = qc_pair.QCPair(token[0].split(' '),
                                           token[1].split(' '), float(token[2]))
                token = token[0].split(' ') + token[1].split(' ')
            else:
                token = lines.strip().split('\t')
            #    print " ".join(jieba.cut(token[0]))
                one = " ".join(jieba.cut(token[0])).split(' ')
                two = " ".join(jieba.cut(token[1])).split(' ')
                new_entry = qc_pair.QCPair(one, two, int(token[2]))
                token = one + two

            for tmp in token:
                # if re.match(u'[^A-Za-z0-9\u4e00-\uf9a5]', tmp.decode('utf-8')) is not None:
                #     continue
                if self.word_index.has_key(tmp):
                    continue
                else:
                    vec, _ = self.read_word_vec(tmp)
                    if _ == -1:
                        self.word_index[tmp] = self.word_index['UNK']
                        self.word_dict[tmp] = self.word_dict['UNK']
                    else:
                        self.word_index[tmp] = index_count
                        self.word_vectors.append(vec)
                        self.word_dict[tmp] = vec
                        index_count += 1
            for i in range(0, len(new_entry.query)):
                new_entry.q_list[i] = self.word_index[new_entry.query[i]]
            new_entry.q_list[len(new_entry.query)] = self.word_index['EOS']
            for i in range(0, len(new_entry.question)):
                new_entry.c_list[i] = self.word_index[new_entry.question[i]]
            new_entry.c_list[len(new_entry.question)] = self.word_index['EOS']
            self.corpus_set.append(new_entry)
        self.word_vectors = np.array(self.word_vectors)
        f.close()
        return

    def read_word_vec(self, word):
    #   return np.zeros(config.embedding_size, dtype='float32'), 1
        vec, res = self.getter(config.link + word)
        if res == 1:
            vec = map(float, vec.split(" "))
            vec = np.array(vec, dtype='float32')
        return vec, res

    def getter(self, link):
        def _getter(link):
            link = requests.get(link, timeout=5)
            if link.status_code != 200:
                return None
            return link.text.encode('utf-8')
    #    logger.info('in getter:')
        for _ in range(5):
        #    print 'getter'+link
            try:
                ret = _getter(link)
            except requests.ConnectionError:
                print 'requests.ConnectiontError'
                time.sleep(10)
                continue
            except requests.exceptions.Timeout:
                print 'requests.exceptions.Timeout'
                time.sleep(10)
                continue
            else:
                if ret is None:
                    return None, -1
                else:
                    return ret, 1
        raise requests.ConnectionError("Connection Error")