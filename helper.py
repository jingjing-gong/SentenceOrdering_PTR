import numpy as np
import operator
from collections import defaultdict
import logging
import copy
import tensorflow as tf

import TfUtils
import time
from copy import deepcopy

class Vocab(object):
    unk = u'<unk>'
    sos = u'<sos>'
    eos = u'<eos>'
    def __init__(self, unk=unk):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)
        self.add_word(self.sos, count=0)
        self.add_word(self.eos, count=0)

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
 

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'wb') as fd:
            for (word, freq) in sorted_tup:
                fd.write(('%s\t%d\n'%(word, freq)).encode('utf-8'))
            

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'rb') as fd:
            for line in fd:
                line_uni = line.decode('utf-8')
                word, freq = line_uni.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = int(freq)
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

def load_data(fileName):
    with open(fileName,'r') as fd:
        step=0
        data_list = []
        tmp_data = []
        for line in fd:
            line_uni = line.decode('utf-8')
            if step < 2:
                tmp_data = []
                step+=1
                continue
            step+=1
            if line_uni.isspace():
                step=0
                if len(tmp_data) != 0:
                    data_list.append(tmp_data)
                continue
            tmp_data.append(line_uni.strip().split())
    return data_list

def shuffleData(data, noize, noize_num=1):

    def get_rand_sents():
        if noize_num > 0:
            ids = [np.random.randint(len(noize)) for _ in range(noize_num)]
            return [noize[i] for i in ids]
        
        elif noize_num < 0:
            num = np.random.choice(range(-noize_num+1))
            ids = [np.random.randint(len(noize)) for _ in range(num)]
            return [noize[i] for i in ids]
        else:
            return []
        
    def shuffleList(li, rand_sents):
        true_len = len(li)
        li = copy.deepcopy(li)
        li = li + rand_sents
        index = range(len(li))
        np.random.shuffle(index)
        tmp_list = [li[i] for i in index]
        index = np.argsort(index)
        index = index[:true_len]
        return tmp_list, index.tolist()
    
    ret_data=[]
    ret_label = []
    for item in data:
        rand_sents = get_rand_sents()
        shuffled, label = shuffleList(item, rand_sents=rand_sents)
        ret_data.append(shuffled)
        ret_label.append(label)
    return ret_data, ret_label

def batch_encodeNpad(data, label, vocab):
    sent_num_enc = [len(i) for i in data]
    sent_num_dec = [len(i) for i in label]
    max_sent_num = max(sent_num_enc)
    sent_len = [[len(i[j]) if j<len(i) else 0 for j in range(max_sent_num)]for i in data]
    max_sent_len = max(flatten(sent_len))
    ret_label = [[i[j] if j<len(i) else -1 for j in range(max_sent_num)] for i in label]
    ret_batch = np.zeros([len(data), max_sent_num, max_sent_len], dtype=np.int32)
    for (i, item) in enumerate(data):
        for (j, sent) in enumerate(item):
            for (k, word) in enumerate(sent):
                ret_batch[i, j, k] = vocab.encode(word)
    return ret_batch, np.array(ret_label), sent_num_enc, sent_num_dec, sent_len #(b_sz, max_snum, max_slen), (b_sz, max_snum), (b_sz,), (max_slen)

"""Prediction """
def calculate_accuracy_seq(pred_matrix, label_matrix, eos_id=0):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, pred_seqLen), type of int
        label_matrix: true label matrix, shape of (data_num, true_seqLen), type of int

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if len(pred_matrix) != len(label_matrix):
        raise TypeError('first argument and second argument have different length')
    
    def seq_equal(seq_a, seq_b):
        length = min(len(seq_a), len(seq_b))
        for i in range(length):
            if seq_a[i] == eos_id and seq_b[i] == eos_id:
                return True
            if seq_a[i] != seq_b[i]:
                return False
        return False
    
    match = [seq_equal(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]
    return np.mean(match)

def print_pred_seq(pred_matrix, label_matrix):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, pred_seqLen), type of int

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    eos_id=0
    def seq_equal(seq_a):
        length = len(seq_a)
        sentence = []
        for i in range(length):
            sentence+= [seq_a[i]]
            if seq_a[i] == eos_id:
                return sentence
        return sentence
    for i in range(len(pred_matrix)):
        print(' '.join([str(j) for j in label_matrix[i]]) + '\t' + ' '.join([str(j) for j in pred_matrix[i]]))

def flatten(li):
    ret = []
    for item in li:
        if isinstance(item, list) or isinstance(item, tuple):
            ret += flatten(item)
        else:
            ret.append(item)
    return ret

"""Read and make embedding matrix"""
def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.strip()
            line_uni = line.decode('utf-8')
            values = line_uni.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    embedding_matrix = np.zeros((len(vocab_dic) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
 
"""Data iterating"""
def data_iter(data, batch_size, vocab, noize_list, noize_num=1):
    
    data_len = len(data)
    epoch_size = data_len // batch_size
    
    idx = np.arange(data_len)
    np.random.shuffle(idx)
      
    for i in xrange(epoch_size):
        indices = range(i*batch_size, (i+1)*batch_size)
        indices = idx[indices]
        
        batch_data = [data[i] for i in indices]
        b_data_ret, b_label_ret = shuffleData(batch_data, noize_list, noize_num=noize_num)
        yield batch_encodeNpad(b_data_ret, b_label_ret, vocab)
    

def average_sentence_as_vector(fetch_output, lengths):
    """
    fetch_output: shape=(batch_size, num_sentence, len_sentence, embed_size)
    lengths: shape=(batch_size, num_sentence)
    maxLen: scalar
    """
    mask = TfUtils.mkMask(lengths, tf.shape(fetch_output)[-2]) #(batch_size, num_sentence, len_sentence)
    avg = TfUtils.reduce_avg(fetch_output, tf.expand_dims(mask, -1), tf.expand_dims(lengths, -1), -2) #(batch_size, num_sentence, embed_size)
    return avg