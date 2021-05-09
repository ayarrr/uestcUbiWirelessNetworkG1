# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
        return concat_outputs, None

        ''' 
         多路注意力机制
        '''
        # concat
        # with tf.variable_scope('bidaf'):
        #     _s11 = tc.layers.fully_connected(passage_encodes, num_outputs=self.hidden_size, activation_fn=None)
        #     _s1 = tf.expand_dims(_s11, 1)
        #     _s22 = tc.layers.fully_connected(question_encodes, num_outputs=self.hidden_size, activation_fn=None)
        #     _s2 = tf.expand_dims(_s22, 2)
        #     sjt1 = tc.layers.fully_connected(tf.nn.tanh(_s1 + _s2), num_outputs=1, activation_fn=None)
        #     sjt = tf.squeeze(sjt1)
        #     ait = tf.nn.softmax(sjt, -1)
        #     qtc = tf.matmul(ait, passage_encodes)
        #
        #     # bilinear
        #     _s11 = tc.layers.fully_connected(passage_encodes, num_outputs=2 * self.hidden_size, activation_fn=None)
        #     _s1 = tf.transpose(_s11, perm=[0, 2, 1])
        #     sjt = tf.matmul(question_encodes, _s1)
        #     ait = tf.nn.softmax(sjt, -1)
        #     qtb = tf.matmul(ait, passage_encodes)
        #
        #     # dot
        #     _s1 = tf.expand_dims(passage_encodes, 1)
        #     _s2 = tf.expand_dims(question_encodes, 2)
        #     _wd = tc.layers.fully_connected(_s1 * _s2, num_outputs=self.hidden_size, activation_fn=None)
        #     _wd1 = tf.nn.tanh(_wd)
        #     sjt1 = tc.layers.fully_connected(_wd1, num_outputs=1, activation_fn=None)
        #     sjt = tf.squeeze(sjt1)
        #     ait = tf.nn.softmax(sjt, -1)
        #     qtd = tf.matmul(ait, passage_encodes)
        #
        #     # minus
        #     _sm = tc.layers.fully_connected((_s1 - _s2), num_outputs=self.hidden_size, activation_fn=None)
        #     _sm1 = tf.nn.tanh(_sm)
        #     sjt1 = tc.layers.fully_connected(_sm1, num_outputs=1, activation_fn=None)
        #     sjt = tf.squeeze(sjt1)
        #     ait = tf.nn.softmax(sjt, -1)
        #     qtm = tf.matmul(ait, passage_encodes)
        #     # ？？？？？？？
        #     # dot,self-attention
        #     _s1 = tf.expand_dims(question_encodes, 1)  # [batch, 1, query_len, encoder_size]
        #     _s2 = tf.expand_dims(question_encodes, 2)  # [batch, query_len, 1, encoder_size]
        #     _ws = tc.layers.fully_connected(_s1 * _s2, num_outputs=self.hidden_size, activation_fn=None)
        #     _ws1 = tf.nn.tanh(_ws)
        #     sjt1 = tc.layers.fully_connected(_ws1, num_outputs=1, activation_fn=None)
        #     sjt = tf.squeeze(sjt1)
        #     ait = tf.nn.softmax(sjt, -1)
        #     qts = tf.matmul(ait, question_encodes)
        #
        #     concat_outputs = tf.concat([question_encodes, qts, qtc, qtd, qtb, qtm], -1)

