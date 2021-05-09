import tensorflow as tf
import tensorflow.contrib as tc
class SelfAttentionLayer(object):
        def __init__(self, hidden_size):
           self.hidden_size = hidden_size
        def self_attn(self,fuse_p_encodes):
            """
           计算单文档自注意力
            """
            with tf.variable_scope('single_selfattn'):

                _s11=tf.layers.dense(fuse_p_encodes,units=self.hidden_size,use_bias=False,activation=None)
                # _s11 = tc.layers.fully_connected(fuse_p_encodes, num_outputs=self.hidden_size, activation_fn=None)
                # # print("全连接输出的是什么：",_s11)
                # _s1 = tf.expand_dims(_s11,1)
                _s22=tf.layers.dense(fuse_p_encodes,units=self.hidden_size,use_bias=False,activation=None)
                # _s22 = tc.layers.fully_connected(fuse_p_encodes, num_outputs=self.hidden_size, activation_fn=None)
                # _s2 = tf.expand_dims(_s22,2)
                sjt11=tf.nn.tanh((_s11 + _s22))
                sjt1=tf.layers.dense(sjt11,units=1,use_bias=False,activation=None)
                # sjt1 = tc.layers.fully_connected(sjt11, num_outputs=1, activation_fn=None)
                # sjt = tf.squeeze(sjt1)
                # print("sjt1.shape:", sjt1.shape)
                ait = tf.nn.softmax(sjt1, -1)
                # print("ait.shape:", ait.shape)

                ct = tf.multiply(ait, fuse_p_encodes)
                # print("ct.shape:", ct.shape)
                concat_outputs = tf.concat([fuse_p_encodes, ct], -1)
                return concat_outputs
        def multi_attn(self,concat_passage_encodes):
            """
           计算多文档注意力
            """
            with tf.variable_scope('multi_attn'):

                _s11=tf.layers.dense(concat_passage_encodes,units=self.hidden_size,use_bias=False,activation=None)
                _s22=tf.layers.dense(concat_passage_encodes,units=self.hidden_size,use_bias=False,activation=None)
                sjt11=tf.nn.tanh((_s11 + _s22))
                sjt1=tf.layers.dense(sjt11,units=1,use_bias=False,activation=None)
                # print("sjt1.shape:", sjt1.shape)
                ait = tf.nn.softmax(sjt1, -1)
                # print("ait.shape:", ait.shape)
                ct = tf.multiply(ait, concat_passage_encodes)
                # print("ct.shape:", ct.shape)
                concat_outputs = tf.concat([concat_passage_encodes, ct], -1)
                return concat_outputs