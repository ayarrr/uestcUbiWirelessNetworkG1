import os
import re


import json

from gensim.models.word2vec import LineSentence, Word2Vec

import jieba

def func(path, fout,train=False):
   print(path)
   with open(path, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if train:
                if len(sample['answer_spans']) == 0:
                    continue
                if sample['answer_spans'][0][1] >= 600:
                    continue
            if (sample['question_type'] == 'YES_NO'):
                text = sample['answers'][0]
                # print(text)
                text = re.sub(r' ', '', text)
                text = re.sub(r'[0-9]+', '', text)
                text = re.sub(r'[’!"#$%&\'()*+,-./:;<=>?@，。★、…【】《》？“”‘！^_`{|}~]+', '', text)
                text1 = list(jieba.cut(text))
                fout.write(' '.join(text1) + '\n')


def make_corpus():

    #print("-------------haha")

    with open('corpus.txt', 'wt', encoding='utf-8') as fout:
        train_path = '../../data/demo/trainset/search.train.json'
        func(train_path, fout,train=True)
        dev_path = '../../data/demo/devset/search.dev.json'
        func(dev_path, fout,train=True)
        test_path = 'C:/Users/咕噜咕噜噜/Desktop/test.predicted-5-13.json'
        func(test_path, fout)
        # train_data = json.load(open('./sentiment_data/train_data.json'))
        #
        # func(train_data, fout)
        #
        # dev_data = json.load(open('./sentiment_data/dev_data.json'))
        #
        # func(dev_data, fout)
        #
        # test_data = json.load(open('./sentiment_data/test_data.json'))
        #
        # func(test_data, fout)



if __name__ == "__main__":

    if not os.path.exists('corpus.txt'):

        make_corpus()



    sentences = LineSentence('corpus.txt')

    model = Word2Vec(sentences, sg=1, size=128, workers=4, iter=8, negative=8, min_count=2)

    word_vectors = model.wv

    word_vectors.save_word2vec_format('word2vec.txt', fvocab='vocab.txt')