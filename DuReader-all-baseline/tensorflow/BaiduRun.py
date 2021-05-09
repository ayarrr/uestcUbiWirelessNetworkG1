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
This module prepares and runs the whole system.
"""
import os
import pickle
import argparse
import logging
from dataloader.BaiduDataLoader import BRCDataset
from VocabBuild.BaiduVocab import Vocab
from model.BaiduModel import RCModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"## written by Fangyueran

'''Which dataset do you want to use, just choose between search and zhidao'''
dataName = 'search'  

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='2',## written by Fangyueran
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--char_embed_size', type=int, default=32,
                                help='size of the char embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')#一个样本中最大的文章数目
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')#文章最大的长度
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')#问题最大的长度
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')#答案最大的长度

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/demo/'+dataName+'.train.json'],
                               help='list of files that contain the preprocessed train data') 
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/demo/'+dataName+'.dev.json'],
                               help='list of files that contain the preprocessed dev data')  
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/demo/'+dataName+'.test.json'],  
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--save_dir', default='./data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/'+dataName+'/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/Baidu/'+dataName+'/',  
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/Baidu/'+dataName+'/', 
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/Baidu/'+dataName+'/', 
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='./data/summary/Baidu/'+dataName+'/log.txt', 
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_word_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_char_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    print('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    print('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    # print(vocab.size()) brc_data.word_iter('train') is a generator
    for word in brc_data.word_iter('train'):
        vocab.add(word)
        [vocab.add_char(ch) for ch in word]

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    unfiltered_vocab_char_size = vocab.char_size()
    vocab.filter_chars_by_cnt(min_cnt=2)
    filtered_char_num = unfiltered_vocab_char_size - vocab.char_size()
    logger.info('After filter {} chars, the final char vocab size is {}'.format(filtered_char_num,
                                                                                vocab.char_size()))
    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size) #这里只是调用了randomly_init_embeddings这个函数 并没有对vocab进行操作
    vocab.randomly_init_char_embeddings(args.char_embed_size)

    logger.info('Saving vocab...')
    print('Saving vocab...')
    with open(os.path.join(args.vocab_dir, dataName+'BaiduVocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    print('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, dataName+'BaiduVocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('Converting text into ids...')
    print('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    print('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    print('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, dataName+'BaiduVocab.data'), 'rb') as fin: 
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    print('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    print('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    print('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    print('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, dataName+'BaiduVocab.data'), 'rb') as fin: 
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    print('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    print('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    print('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # if args.log_path:
    #     file_handler = logging.FileHandler(args.log_path)
    #     file_handler.setLevel(logging.INFO)
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)
    # else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
      prepare(args)
    if args.train:
      train(args)
    if args.evaluate:
      evaluate(args)
    if args.predict:
      predict(args)

if __name__ == '__main__':
    run()
