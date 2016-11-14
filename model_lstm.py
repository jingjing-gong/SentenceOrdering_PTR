from Config import Config
from helper import Vocab
from myRNN import raw_rnn

import helper

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
import numpy as np
import sys
import os
import argparse
import logging

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
    
    args = parser.parse_args()

class PtrNet(object):
  
    def __init__(self, args=args, test=False):
        self.vocab = Vocab()
        self.config=Config()
        
        self.weight_Path = args.weight_path
        if not os.path.exists(self.weight_Path):
            os.makedirs(self.weight_Path)
            
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_Path+'/config')
        
        self.step_p_epoch = self.load_data(test)
        
        self.add_placeholders()
        self.add_embedding()
        self.fetch_input()
        logits, self.prediction = self.add_model()
        train_loss, valid_loss, reg_loss = self.add_loss_op(logits)
        self.train_op = self.add_train_op(train_loss)
        self.loss = valid_loss
        self.reg_loss = reg_loss
        
        MyVars = [v for v in tf.trainable_variables()]
        MyVars_name = [v.name for v in MyVars]
        self.MyVars = MyVars
        print MyVars_name
        
    def decoder(self, cell, initial_state, encoder_outputs, encoder_inputs ,scope=None):
        batch_size = tf.shape(encoder_outputs)[0]
        tstps_en = tf.shape(encoder_outputs)[1]
        hidden_size = tf.shape(encoder_outputs)[2]
        emb_sz = tf.shape(encoder_inputs)[2]
        
        def loop_fn(time, cell_output, cell_state, loop_state, emit_ta):
            """cell_output shape (batch_size, hidden_size)
                encoder_outputs shape (b_sz, tstps_en, h_sz)
                encoder_inputs shape (b_sz, tstps_en, embed_sz)
            """
            
            if cell_output is None:  # time == 0
                next_cell_state = initial_state
                emit_output= tf.ones(tf.shape(initial_state[1])[:1], dtype=tf.int32) * tf.constant(-1) #(batch_size)
                next_input = tf.squeeze(self.sos, [1])
                elements_finished = tf.logical_and(tf.cast(emit_output, dtype=tf.bool), False)
                
            else:
                
                next_cell_state = cell_state
                decoder_outputs = tf.expand_dims(cell_output, 1) #(batch_size, 1, hidden_size)
                encoder_outputs_reshape = tf.reshape(encoder_outputs, shape=(-1, self.config.hidden_size)) #(batch_size*time_steps, hidden_size)
                decoder_outputs_reshape = tf.reshape(decoder_outputs, shape=(-1, self.config.hidden_size)) #(batch_size*1, hidden_size)
                encoder_outputs_linear_reshape = tf.nn.rnn_cell._linear(encoder_outputs_reshape, output_size=self.config.hidden_size, 
                                         bias=False, scope='Ptr_W1')    #(b_sz*tstps_en, h_sz)
                decoder_outputs_linear_reshape = tf.nn.rnn_cell._linear(decoder_outputs_reshape, output_size=self.config.hidden_size, 
                                         bias=False, scope='Ptr_W2')    #(b_sz*1, h_sz)
                encoder_outputs_linear = tf.reshape(encoder_outputs_linear_reshape, tf.shape(encoder_outputs))
                decoder_outputs_linear = tf.reshape(decoder_outputs_linear_reshape, tf.shape(decoder_outputs))
                
                encoder_outputs_linear_expand = tf.expand_dims(encoder_outputs_linear, 1) #(b_sz, 1, tstp_en, h_sz)
                decoder_outputs_linear_expand = tf.expand_dims(decoder_outputs_linear, 2) #(b_sz, 1, 1, h_sz)
                
                after_add = tf.tanh(encoder_outputs_linear_expand + decoder_outputs_linear_expand)  #(b_sz, 1, tstp_en, h_sz)
                
                after_add_reshape = tf.reshape(after_add, shape=(-1, self.config.hidden_size))
                
                after_add_linear_reshape = tf.nn.rnn_cell._linear(after_add_reshape, output_size=1, #(b_sz*1*tstp_en, 1)
                                         bias=False, scope='Ptr_v')
                after_add_linear = tf.reshape(after_add_linear_reshape, shape=(-1, 1, tstps_en)) #(b_sz, 1, tstp_en)
                en_length_mask = tf.sequence_mask(self.encoder_tstps,                #(b_sz, tstp_en)
                                    maxlen=tstps_en, dtype=tf.float32)
                en_length_mask = tf.expand_dims(en_length_mask, 1)    #(b_sz, 1, tstp_en)

                """mask out already hitted ids""" 
                hit_ids = tf.cond(emit_ta.size() > 0, lambda: emit_ta.pack(), lambda: tf.ones(shape=[1, batch_size], dtype=tf.int32)*-1) #(to_cur_tstp, b_sz)
                masks = tf.one_hot(hit_ids, depth=tstps_en, on_value=True, off_value=False) #(to_cur_tstp, b_sz, tstp_en)
                masks = tf.reduce_any(masks, reduction_indices=[0]) #(b_sz, tstp_en)
                hit_masks = tf.logical_not(masks)
                
                logits = tf.reshape(after_add_linear * en_length_mask, 
                                shape=(-1, tstps_en))  #(b_sz, tstp_en)
                dummy_1 = tf.nn.softmax(logits) * tf.cast(hit_masks, dtype=tf.float32) #(b_sz, tstp_en)
                
                emit_output = tf.arg_max(dummy_1, dimension=1)          #(batch_size)
                emit_output = tf.cast(emit_output, dtype=tf.int32)
                
                bool_mask = tf.one_hot(emit_output, depth=tstps_en, on_value=True, off_value=False) #(b_sz, tstps_en)
                bool_mask = tf.reshape(bool_mask, shape=(batch_size, tstps_en))
                next_input = tf.boolean_mask(encoder_inputs, bool_mask) #(b_sz, emb_sz)
                
                elements_finished = tf.equal(emit_output, 0) #(batch_size)
                elements_finished = tf.reshape(elements_finished, (-1,))
                
            elements_finished = tf.logical_or(elements_finished, (time >= self.config.num_steps))
            next_loop_state = loop_state
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        return raw_rnn(cell, loop_fn, scope=scope)
        
    def add_placeholders(self):
        self.ph_encoder_input = tf.placeholder(tf.int32, (None, None, None), name='ph_encoder_input') #(batch_size, tstps_en, max_len_sentence)
        self.ph_decoder_label = tf.placeholder(tf.int32, (None, None), name='ph_decoder_label') #(b_sz, tstps_de)
        self.ph_input_encoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_encoder_len') #(batch_size)
        self.ph_input_decoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_decoder_len') #(batch_size)
        self.ph_input_encoder_sentence_len = tf.placeholder(tf.int32, (None, None), name='ph_input_encoder_sentence_len') #(batch_size, tstps_en)
        
        self.ph_dropout = tf.placeholder(tf.float32, name='ph_dropout')
    
    def add_embedding(self):
        self.embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
    
    def fetch_input(self):
        b_sz = tf.shape(self.ph_encoder_input)[0]
        tstps_en = tf.shape(self.ph_encoder_input)[1]
        tstps_de = tf.shape(self.ph_decoder_label)[1]
        emb_sz = self.config.embed_size
        
        def lstm_sentence_rep(input):
            with tf.variable_scope('lstm_sentence_rep_scope') as scope:
                input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
                length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
                
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
                """#(batch_size, num_sentence, hidden_size)"""
                _, state = tf.nn.dynamic_rnn(lstm_cell, input, length,        #tup((b_sz*tstps_en, h_sz), (b_sz*tstps_en, h_sz)
                                    dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')
                
                state = tf.concat(1, state) #(b_sz*tstps_en, h_sz*2)
                output = tf.reshape(state, shape=[b_sz, tstps_en, -1]) #(b_sz, tstps_en, h_sz*2)
                ###
                scope.reuse_variables()
                
                eos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
                _, state = tf.nn.dynamic_rnn(lstm_cell, eos, tf.ones([b_sz]),        #tup((b_sz, h_sz), (b_sz, h_sz)
                                             dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')
                state = tf.concat(1, state) #(b_sz, h_sz*2)
                state = tf.expand_dims(state, 1) #(b_sz, 1, h_sz*2)
                eos = state
                ###
                sos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
                _, state = tf.nn.dynamic_rnn(lstm_cell, sos, tf.ones([b_sz]),        #tup((b_sz, h_sz), (b_sz, h_sz)
                                             dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')
                state = tf.concat(1, state) #(b_sz, h_sz*2)
                state = tf.expand_dims(state, 1) #(b_sz, 1, h_sz*2)
                sos = state
                
            return output, eos, sos
            
        def cnn_sentence_rep(input):
            # input (batch_size, tstps_en, len_sentence, embed_size)
            input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
            length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
            
            filter_sizes = self.config.filter_sizes
            
            in_channel = self.config.embed_size
            out_channel = self.config.num_filters
            
            def convolution(input, tstps_en, length):
                len_sen = tf.shape(input)[1]
                conv_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.variable_scope("conv-%s" % filter_size):
                        filter_shape = [filter_size, in_channel, out_channel]
                        W = tf.get_variable(name='W', shape=filter_shape)
                        b = tf.get_variable(name='b', shape=[out_channel])
                        conv = tf.nn.conv1d(                # size (b_sz* tstps_en, len_sen, out_channel)
                          input,
                          W,
                          stride=1,
                          padding="SAME",
                          name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        conv_outputs.append(h)
                input = tf.concat(2, conv_outputs) #(b_sz*tstps_en, len_sen, out_channel * len(filter_sizes))
                
                mask = tf.sequence_mask(length, len_sen, dtype=tf.float32) #(b_sz*tstps_en, len_sen)
                
                pooled = tf.reduce_max(input*tf.expand_dims(mask, 2), [1]) #(b_sz*tstps_en, out_channel*len(filter_sizes))
                
                #size (b_sz, tstps_en, out_channel*len(filter_sizes))
                pooled = tf.reshape(pooled, shape=[b_sz, tstps_en, out_channel*len(filter_sizes)])
    
                return pooled
            
            with tf.variable_scope('cnn_sentence_rep_scope') as scope:
                output = convolution(input, tstps_en, length) #size (b_sz, tstps_en, out_channel*len(filter_sizes))
                
                eos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
                scope.reuse_variables()
                eos = convolution(eos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                sos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
                sos = convolution(sos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                
            return output, eos, sos
            
        def cbow_sentence_rep(input):
            output = helper.average_sentence_as_vector(input, 
                                              self.ph_input_encoder_sentence_len) #(b_sz, tstp_en, emb_sz)
            eos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
            sos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
            return output, eos, sos
        
        encoder_input = tf.nn.embedding_lookup(self.embedding, self.ph_encoder_input) #(batch_size, tstps_en, len_sentence, embed_size)
        encoder_input, eos, sos = lstm_sentence_rep(encoder_input) #(b_sz, tstp_en, emb_sz)
        emb_sz = tf.shape(encoder_input)[2]
        self.sos = sos
        
        dummy_1 = tf.expand_dims(encoder_input, 1) #(b_sz, 1, tstps_en, emb_sz)
        
        encoder_input_tile = tf.tile(dummy_1, [1, tstps_de, 1, 1])       #(b_sz, tstps_de, tstps_en, emb_sz)
        
        dummy_decoder_label = tf.select(self.ph_decoder_label >= 0, self.ph_decoder_label, tf.zeros_like(self.ph_decoder_label))
        mask = tf.one_hot(dummy_decoder_label, depth=tstps_en, on_value=True, off_value=False) #(b_sz, tstps_de, tstps_en)
        decoder_input = tf.boolean_mask(encoder_input_tile, mask) #(b_sz*tstps_de, emb_sz)
        decoder_input = tf.reshape(decoder_input, shape=(b_sz, tstps_de, emb_sz), name='fetch_input_reshape_0') #(b_sz, tstps_de, emb_sz)
        
        encoder_input = tf.concat(concat_dim=1, values=[eos, encoder_input]) #(b_sz, tstps_en+1, emb_sz)
        decoder_input = tf.concat(concat_dim=1, values=[sos, decoder_input]) #(b_sz, tstps_de+1, emb_sz)
    
        self.encoder_tstps = self.ph_input_encoder_len + 1
        self.decoder_tstps = self.ph_input_decoder_len + 1
        dummy_1 = tf.reshape(tf.ones(shape=(b_sz, 1), dtype=tf.int32)* tf.constant(-1), shape=(b_sz, 1), name='fetch_input_reshape_1') #b_sz, 1
        decoder_label = tf.concat(concat_dim=1, values=[self.ph_decoder_label, dummy_1]) + 1               #b_sz, tstps_de+1
        self.decoder_label = tf.sequence_mask(self.decoder_tstps, 
                                              maxlen=tstps_de+1, dtype=tf.int32) * decoder_label #b_sz, tstps_de+1
    
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
    
    def add_model(self):
        """
            input_tensor #(batch_size, num_sentence, embed_size)
            input_len    #(batch_size)
        """
        encoder_dropout_input = tf.nn.dropout(self.encoder_input, self.ph_dropout, name='encoder_Dropout')
        decoder_dropout_input = tf.nn.dropout(self.decoder_input, self.ph_dropout, name='decoder_Dropout')
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        """#(batch_size, num_sentence, hidden_size)"""
        encoder_outputs, state = tf.nn.dynamic_rnn(lstm_cell, encoder_dropout_input, self.encoder_tstps, 
                            dtype=tf.float32, swap_memory=True, time_major=False, scope = 'rnn_encode')
        self.state=state
        with tf.variable_scope('decoder') as vscope:
            decoder_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, decoder_dropout_input, self.decoder_tstps,   #(batch_size, time_steps, hidden_size)
                initial_state=state, dtype=tf.float32, swap_memory=True, time_major=False, scope='rnn_decode')
            
            with tf.variable_scope('rnn_decode'):
                #tf.reshape(self.ph_decoder_label, shape=(-1, 1)) #(batch_size*time_steps, 1)
                encoder_outputs_reshape = tf.reshape(encoder_outputs, shape=(-1, self.config.hidden_size), name='add_model_reshape_0') #(batch_size*time_steps, hidden_size)
                decoder_outputs_reshape = tf.reshape(decoder_outputs, shape=(-1, self.config.hidden_size), name='add_model_reshape_1') #(batch_size*time_steps_1, hidden_size)
                encoder_outputs_linear_reshape = tf.nn.rnn_cell._linear(encoder_outputs_reshape, output_size=self.config.hidden_size, #(#(batch_size*time_steps, hidden_size))
                                         bias=False, scope='Ptr_W1')
                decoder_outputs_linear_reshape = tf.nn.rnn_cell._linear(decoder_outputs_reshape, output_size=self.config.hidden_size, #(#(batch_size*time_steps, hidden_size))
                                         bias=False, scope='Ptr_W2')
                encoder_outputs_linear = tf.reshape(encoder_outputs_linear_reshape, tf.shape(encoder_outputs), name='add_model_reshape_2')
                decoder_outputs_linear = tf.reshape(decoder_outputs_linear_reshape, tf.shape(decoder_outputs), name='add_model_reshape_3')
                
                encoder_outputs_linear_expand = tf.expand_dims(encoder_outputs_linear, 1) #(b_sz, 1, tstp_en, h_sz)
                decoder_outputs_linear_expand = tf.expand_dims(decoder_outputs_linear, 2) #(b_sz, tstp_de, 1, h_sz)
                
                after_add = tf.tanh(encoder_outputs_linear_expand + decoder_outputs_linear_expand)  #(b_sz, tstp_de, tstp_en, h_sz)
                
                after_add_reshape = tf.reshape(after_add, shape=(-1, self.config.hidden_size), name='add_model_reshape_4')
                
                after_add_linear_reshape = tf.nn.rnn_cell._linear(after_add_reshape, output_size=1, #(b_sz*tstp_de*tstp_en, 1)
                                         bias=False, scope='Ptr_v')
                after_add_linear = tf.reshape(after_add_linear_reshape, shape=tf.shape(after_add)[:3], name='add_model_reshape_5') #(b_sz, tstp_de, tstp_en)
                en_length_mask = tf.sequence_mask(self.encoder_tstps,                #(b_sz, tstp_en)
                                    maxlen=tf.shape(after_add_linear)[-1], dtype=tf.float32)
                en_length_mask = tf.expand_dims(en_length_mask, 1)    #(b_sz, 1, tstp_en)
                
                logits = tf.reshape(after_add_linear * en_length_mask, 
                                    shape=(-1, tf.shape(after_add_linear)[-1]), name='add_model_reshape_6') #(b_sz * tstp_de, tstp_en)
                
            vscope.reuse_variables()
            outputs_ta, _, _ = self.decoder(lstm_cell, state, encoder_outputs, encoder_dropout_input, scope='rnn_decode')
            
            outputs = outputs_ta.pack() #(time_steps, batch_size)
            outputs = tf.transpose(outputs, [1, 0]) #(batch_size, time_steps)
            
#         self.logic_hid_state = logic_hid_state
        self.logits = logits
        self.encoder_outputs = encoder_outputs
        return logits, outputs
    
    def add_loss_op(self, logits):
        def seq_loss(logits_tensor, label_tensor, length_tensor):
            """
            Args
                logits_tensor: shape (batch_size*time_steps_de, time_steps_en)
                label_tensor: shape (batch_size, time_steps_de), label id 1D tensor
                length_tensor: shape(batch_size)
            Return
                loss: A scalar tensor, mean error
            """
    
            labels = tf.reshape(label_tensor, shape=(-1,))
            loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_tensor, labels, name='sparse_softmax')
            losses = tf.reshape(loss_flat, shape=tf.shape(label_tensor)) #(batch_size, tstp_de)
            length_mask = tf.sequence_mask(length_tensor, tf.shape(losses)[1], dtype=tf.float32, name='length_mask')
            losses_sum = tf.reduce_sum(losses*length_mask, reduction_indices=[1]) #(batch_size)
            losses_mean = losses_sum / (tf.to_float(length_tensor)+1e-20) #(batch_size)
            loss = tf.reduce_mean(losses_mean) #scalar
            return loss 
        
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding]) *self.config.reg
        valid_loss = seq_loss(logits, self.decoder_label, self.decoder_tstps)
        train_loss = reg_loss + valid_loss
        return train_loss, valid_loss, reg_loss
    
    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   int(self.config.decay_epoch * self.step_p_epoch), self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    """data related"""
    def load_data(self, test):
        self.vocab.load_vocab_from_file(self.config.vocab_path)
        if not test:
            self.train_data = helper.load_data(self.config.train_data)
            self.val_data = helper.load_data(self.config.val_data)
            step_p_epoch = len(self.train_data) // self.config.batch_size
        else:
            self.test_data = helper.load_data(self.config.test_data)
            step_p_epoch = 0
        return step_p_epoch

    def create_feed_dict(self, input_batch, sent_len, encoder_len, label_batch=None, decoder_len=None, mode='train'):
        """
        note that the order of value in input_batch tuple matters 
        Args
            input_batch, tuple (encoder_input, decoder_input, decoder_label)
            encoder_len, a length list shape of (batch_size)
            decoder_len, a length list shape of (batch_size+1) with one more word <sos> or <eos>
        Returns
            feed_dict: a dictionary that have elements
        """
        if mode == 'train':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, self.ph_decoder_label, 
                            self.ph_input_encoder_len, self.ph_input_decoder_len, self.ph_dropout)
            data_batch = (input_batch, sent_len, label_batch, encoder_len, decoder_len, self.config.dropout)
        elif mode == 'predict':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, encoder_len, self.ph_dropout)
            data_batch = (input_batch, sent_len, encoder_len, self.config.dropout)
        
        feed_dict = dict(zip(placeholders, data_batch))
        
        return feed_dict

    def run_epoch(self, sess, input_data, verbose=None):
        """
        Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(input_data)
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, (ret_batch, ret_label, sent_num, sent_len) in enumerate(helper.data_iter(input_data, self.config.batch_size, self.vocab)):
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num, ret_label, sent_num)
           
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        sys.stdout.write('\n')
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def fit(self, sess, input_data, verbose=None):
        """
        Runs an epoch of validation or test. return test error

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(input_data[0])
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, (ret_batch, ret_label, sent_num, sent_len) in enumerate(helper.data_iter(input_data, self.config.batch_size, self.vocab)):
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num, ret_label, sent_num)
            loss = sess.run(self.loss, feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:])))
                sys.stdout.flush()
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def predict(self, sess, input_data, verbose=None):
        preds = []
        true_label = []
        for _, (ret_batch, ret_label, sent_num, sent_len) in enumerate(helper.data_iter(input_data, self.config.batch_size, self.vocab)):
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num, ret_label, sent_num)
            pred = sess.run(self.prediction, feed_dict=feed_dict)
            preds+=pred.tolist()
            true_label+=ret_label.tolist()
        return preds, true_label

def test_case(sess, model, data, onset='VALIDATION'):
    """pred must be list"""
    def pad_list(lst, pad=-1):
        inner_max_len = max(map(len, lst))
        map(lambda x: x.extend([pad]*(inner_max_len-len(x))), lst)
        return np.array(lst)

    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = model.fit(sess, data)
    pred, true_label = model.predict(sess, data)
    
    true_label = pad_list(true_label)
    true_label = np.array(true_label)
    pred = pad_list(pred, pad=0)
    pred = np.array(pred)
    
    true_label += 1
    true_label = np.concatenate([true_label, np.zeros([np.shape(true_label)[0], 1], dtype=np.int32)], axis=1)
    true_label = true_label.tolist()
    pred = pred.tolist()
    accuracy = helper.calculate_accuracy_seq(pred, true_label, eos_id=0)
#     helper.print_pred_seq(pred[:10], true_label[:10])
    
    print 'Overall '+onset+' loss is: {}'.format(loss)
    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    logging.info('Overall '+onset+' loss is: {}'.format(loss))
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    
    return loss, pred

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = PtrNet()
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_loss = np.Inf
            best_val_epoch = 0
            sess.run(tf.initialize_all_variables())
            
            if os.path.exists(model.weight_Path+'/parameter.weight'):
                saver.restore(sess, model.weight_Path+'/parameter.weight')
            
            for epoch in range(model.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = model.run_epoch(sess, model.train_data, verbose=1)
                
                print "Mean loss in this epoch is: ", loss
                logging.info('%s %d%s' % ('='*20+'Epoch', epoch, '='*20))
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss))
                
                val_loss, _ = test_case(sess, model, model.val_data, onset='VALIDATION')
                
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists(model.weight_Path):
                        os.makedirs(model.weight_Path)

                    saver.save(sess, model.weight_Path+'/parameter.weight')
                if epoch - best_val_epoch > model.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")

def test_run():
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = PtrNet(test='test')
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model.weight_Path+'/parameter.weight')
            test_case(sess, model, model.test_data, onset='TEST')
            
def main(_):
    logFile = args.weight_path+'/run.log'
    
    if args.train_test == "train":
        
        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()


    
