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
This module implements the Vocab class for converting string to id and back
"""

import numpy as np


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    实现一个词汇表，用于存储数据中的词及其相应的嵌入向量。
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}

        # char
        self.id2char = {}
        self.char2id = {}
        self.char_cnt = {}

        self.lower = lower

        self.embed_dim = None
        self.embeddings = None
        self.char_embed_dim = None
        self.char_embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []#如果不是none，那么initial_tokens就有值，不然initial_tokens=[]
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)
            self.add_char(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def char_size(self):
        return len(self.id2char)

    def load_from_file(self, file_path):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line 一个一行只有一个词的文件
        """
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)
            self.add_char(token)

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        如果词不在词汇表，那么返回unk的编号 如果在则返回词的编号
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_char_id(self, token):
        token = token.lower() if self.lower else token
        return self.char2id[token] if token in self.char2id else self.char2id[self.unk_token]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        """
        adds the token to vocab  把词加入到词汇表中
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
            cnt表示要添加的单词的数量 默认为1
            token_cnt应该是记录一个单词出现的数量
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            #如果这个词有了编号 那么就获取他的编号
            idx = self.token2id[token]
        else:
            #如果没有，那么先获取编号——词的长度 也就是获取编号编到哪儿了
            idx = len(self.id2token)
            #分别建立 编号——词和词——编号对应表
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx
    def add_char(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.char2id:
            idx = self.char2id[token]
        else:
            idx = len(self.id2char)
            self.id2char[idx] = token
            self.char2id[token] = idx
        if cnt > 0:
            if token in self.char_cnt:
                self.char_cnt[token] += cnt
            else:
                self.char_cnt[token] = cnt
        return idx
    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count 通过他们的数量过滤词汇表中的词
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered 如果词出现的频率小于mincnt，则该词被过滤掉
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map 重新建立映射关系
        self.token2id = {}
        self.id2token = {}
        # print(self.initial_tokens)
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def filter_chars_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.char2id if self.char_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.char2id = {}
        self.id2char = {}
        for token in self.initial_tokens:
            self.add_char(token, cnt=0)
        for token in filtered_tokens:
            self.add_char(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token 随机初始化每个词的嵌入
        Args:
            embed_dim: the size of the embedding for each token =300
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros(embed_dim) #生成包含embed_dim个元素的零矩阵

    def randomly_init_char_embeddings(self, embed_dim):
        self.char_embed_dim = embed_dim
        self.char_embeddings = np.random.rand(self.char_size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.char_embeddings[self.get_char_id(token)] = np.zeros([self.char_embed_dim])

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0].decode('utf8')
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec
    def convert_char_to_ids(self, tokens):
        vec = []
        for token in tokens:
            char_vec = []
            for char in token:
                char_vec.append(self.get_char_id(char))
            vec.append(char_vec)
        return vec
    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
