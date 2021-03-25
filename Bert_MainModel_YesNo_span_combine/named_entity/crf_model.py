from sklearn_crfsuite import CRF
import pickle


# ******** CRF 工具函数*************
def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


#模型
class CRFModel(object):
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="ResumeNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test', "renmin3"]

    word_lists = []
    tag_lists = []
    #with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
    if split == "renmin3":
        with open(join(data_dir,"renmin3.txt"), 'r', encoding='utf-8') as f:
            word_list = []
            tag_list = []
            data = f.read()
            data = data.replace("\n", "")
            data = data.split(" ")
            for line in data:
                if line != '\n' and line != "" :
                    if len(line.split("/")) != 2:
                        print ("line:", line)
                    word, tag = line.split("/")
                    word_list.append(word)
                    tag_list.append(tag)
                else:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []
    else:
        with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
            word_list = []
            tag_list = []
            for line in f:
                if line != '\n':
                    word, tag = line.strip('\n').split()
                    word_list.append(word)
                    tag_list.append(tag)
                else:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


import re
def train_main():
    """训练模型，评估结果"""

    # 读取数据
    
    
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    
    # 训练CRF模型    
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")
    
    
    #crf_model = load_model("./ckpts/crf.pkl")
    test_word_lists = "11月4日消息，3日下午，2名中国公民和1名新加坡籍人员在印尼万丹省SANGIANG岛附近水域结伴潜水时失踪。昨晚，中国驻印尼大使馆派员连夜驱车赶到事发地，约见万丹省搜救中心负责人，敦促印尼方全力搜救。目前，使馆领侨处和武官处组成的工作组正在现场协助搜救。"
    test_word_lists = re.split(r"[。，]", test_word_lists)
    pred_tag_lists = crf_model.test(test_word_lists)

    def list_divide(a):
        return (len(a)/10)
    pred_tag_lists = [list(map(list_divide, pred_tag_lists[i])) for i in range(len(pred_tag_lists))]
    for i in range(len(test_word_lists)):
        print ("test_word_lists:", test_word_lists[i])
        print ("pred_tag_lists:", pred_tag_lists[i])
        print ()

    #pred_tag_lists = 

if __name__ == "__main__":
    train_main()