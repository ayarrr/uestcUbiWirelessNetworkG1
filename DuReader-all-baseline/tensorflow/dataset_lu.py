import os
import json
import logging
import numpy as np
import re
from collections import Counter
import jieba.posseg as pseg

class BRCDataset(object):

    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,max_pos_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_pos_len = max_pos_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))


    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
            return stopwords
        with open(data_path,encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']
                words = pseg.cut(str(sample['question']))
                question_flag=[]
                for word, flag in words:
                    question_flag.append(flag)
                    # print('%s %s' % (word, flag))
                # print(question_flag)
                sample['question_pos'] = question_flag

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        # print( doc['paragraphs'][most_related_para])
                        word1 = pseg.cut(doc['paragraphs'][most_related_para])
                        passage_pos=[]
                        for word, flag in word1:
                            passage_pos.append(flag)
                            # print('%s %s' % (word, flag))
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected'],
                             'passage_pos':passage_pos}
                        )
                    else:
                        para_total = []
                        para_token = []
                        # for para_tokens in doc['segmented_paragraphs']:
                        stopwords = stopwordslist('../data/stop_words.txt')
                        for para_tokens in doc['segmented_paragraphs']:
                            for token in para_tokens:
                                token=re.sub(r"[百度经验jingyan.combaidu<p><imgsrc/></p>\u3000]+|[\\x0a]+","",token)
                                # if token not in stopwords:
                                #     if token != '\t':
                                para_token.append(token)
                            para_total.extend(para_token)
                            # para_total += para_token+["。"]
                            # print(para_total)
                        word2 = pseg.cut(str(para_total))
                        passage_pos = []
                        for word, flag in word2:
                            passage_pos.append(flag)
                        sample['passages'].append({'passage_tokens': para_total,
                                                   'passage_pos':passage_pos})
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id,pad_pos_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_pos_ids':[],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_pos_ids':[],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_pos_ids'].append(sample['question_pos_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    passage_pos_ids = sample['passages'][pidx]['passage_pos_ids']
                    batch_data['passage_pos_ids'].append(passage_pos_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_pos_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_pos_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id,pad_pos_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        # print(batch_data)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id,pad_pos_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_pos_len = self.max_pos_len
        pad_p_len = self.max_p_len
            # min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = self.max_q_len
            # min(self.max_q_len, max(batch_data['question_length']))

        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['passage_pos_ids'] = [(ids + [pad_id] * (pad_pos_len - len(ids)))[: pad_pos_len]
                                           for ids in batch_data['passage_pos_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        batch_data['question_pos_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_pos_ids']]


        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token

                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token
    def pos_iter(self, set_name=None):
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_pos']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_pos']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                sample["question_pos_ids"] = vocab.convert_pos_to_ids(sample['question_pos'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])
                    passage['passage_pos_ids'] = vocab.convert_pos_to_ids(passage['passage_pos'])
    def gen_mini_batches(self, set_name, batch_size, pad_id, pad_pos_id,shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        # print("data_size",data_size)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id,pad_pos_id)
