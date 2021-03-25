# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:29:34 2019

@author: 10097
"""

import os
import random
import logging
import numpy as np
import collections
import torch
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers import AdamW as BertAdam
from pytorch_transformers import BertTokenizer

from prepare import write_predictions, write_predictions_test
from CailModel import CailModel
from evaluate import CJRCEvaluator
from read_data import load_dev_features, load_train_features, load_test_features

from named_entity.crf_model import CRFModel

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "unk_logits", "yes_logits", "no_logits", "attention_value"])


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("save model")
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def save_code(path):
    import shutil
    if not os.path.exists(path):
        os.mkdir(path)
    code_path = os.path.join(path+'/code')
    if not os.path.exists(code_path):
        os.mkdir(code_path)
    f_list = os.listdir('./')
    for fileName in f_list:
        if os.path.splitext(fileName)[1] == '.py' or os.path.splitext(fileName)[1] == '.sh':
            shutil.copy(fileName, code_path)


def _test(args, device, n_gpu):
    model = CailModel.from_pretrained(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    test_dataloader, test_examples, test_features = load_test_features(args, tokenizer)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    logger.info("Start evaluating")
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in test_dataloader:
        if len(all_results) % 5000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, \
            batch_unk_logits, batch_yes_logits, batch_no_logits, _ = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            unk_logits = batch_unk_logits[i].detach().cpu().tolist()
            yes_logits = batch_yes_logits[i].detach().cpu().tolist()
            no_logits = batch_no_logits[i].detach().cpu().tolist()
            test_feature = test_features[example_index.item()]
            unique_id = int(test_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions_test.json")
    write_predictions_test(test_examples, test_features, all_results, args.n_best_size, args.max_answer_length,
                                        args.do_lower_case, output_prediction_file, args.verbose_logging,
                           args.version_2_with_negative, args.null_score_diff_threshold)

def _class_test(args, device, n_gpu):
    model = CailModel.from_pretrained(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    
    #总共9类问题，每类问题100个
    label_amount = 0
    label_list = ["时间","谁","地点","实体","原因","什么","方法","数量","是非"]
    for label in range(label_amount):
        args.dev_file = "data/class_dev/class_dev"+str(label)+".json"
        eval_dataloader, eval_examples, eval_features = load_dev_features(args, tokenizer, is_class_test = True)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.eval()
        logger.info("Start evaluating")
        
        
        #从eval_dataloader中读取数据
        all_results = []
        for input_ids, input_mask, segment_ids, word_feature_list2, named_entity_list2, example_indices in eval_dataloader:
            #每1000个样本打印一次信息
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            
            #取出Bert模型需要的三个特征
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            word_feature_list2 = word_feature_list2.to(device)
            named_entity_list2 = named_entity_list2.to(device)
            
            #使用模型进行预测，得到当前批数据的五个概率
            with torch.no_grad():
                batch_start_logits, batch_end_logits, \
                batch_unk_logits, batch_yes_logits, batch_no_logits, attention_value = model(input_ids, segment_ids, input_mask, word_feature_list = word_feature_list2, named_entity_list = named_entity_list2)
            
            #处理当前批预测
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                
                
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                
                #打包预测结果
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             unk_logits=unk_logits,
                                             yes_logits=yes_logits,
                                             no_logits=no_logits,
                                             attention_value = attention_value))
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        
        #写入预测
        all_predictions = write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)
        
        #加载评价器
        evaluator = CJRCEvaluator(args.dev_file)
        #使用评价器进行评价
        res = evaluator.model_performance(all_predictions)
        #返回评价的F1分数
        metrics = {'f1': res['overall']['f1'], "Rouge_L":res['overall']['rouge_l'], 'exact_match':res['overall']['em']}
        
        logger.info("类别 {} 的分数, F1：{:.4f}, Rouge_L： {:.4f}, Exact match： {:.4f}.".
                            format(label_list[label], metrics['f1'], metrics['Rouge_L'], metrics['exact_match']))


def _dev(args, device, model, eval_dataloader, eval_examples, eval_features):
    
    #执行模型的评价
    model.eval()
    logger.info("Start evaluating")
    model.eval()
    
    #从eval_dataloader中读取数据
    all_results = []
    for input_ids, input_mask, segment_ids, word_feature_list2, named_entity_list2, example_indices in eval_dataloader:
        #每1000个样本打印一次信息
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        
        #取出Bert模型需要的三个特征
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        word_feature_list2 = word_feature_list2.to(device)
        named_entity_list2 = named_entity_list2.to(device)
        
        #使用模型进行预测，得到当前批数据的五个概率
        with torch.no_grad():
            batch_start_logits, batch_end_logits, \
            batch_unk_logits, batch_yes_logits, batch_no_logits, batch_attention_value = model(input_ids, segment_ids, input_mask, word_feature_list = word_feature_list2, named_entity_list = named_entity_list2)
        
        #处理当前批预测
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            unk_logits = batch_unk_logits[i].detach().cpu().tolist()
            yes_logits = batch_yes_logits[i].detach().cpu().tolist()
            no_logits = batch_no_logits[i].detach().cpu().tolist()
            attention_value = batch_attention_value[i].detach().cpu().tolist()
            
            
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            
            #打包预测结果
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits,
                                         attention_value = attention_value))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    
    #写入预测
    all_predictions = write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold)
    
    #加载评价器
    evaluator = CJRCEvaluator(args.dev_file)
    #使用评价器进行评价
    res = evaluator.model_performance(all_predictions)
    #返回评价的F1分数
    result = {'f1': res['overall']['f1'], "Rouge_L":res['overall']['rouge_l'], 'exact_match':res['overall']['em']}
    return result


def _train(args, device, n_gpu):
    #实例化一个Bert的tokenizer
    print ("args.bert_model:",args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    #加载训练集数据
    train_dataloader = load_train_features(args, tokenizer)
    #加载验证集数据
    eval_dataloader, eval_examples, eval_features = load_dev_features(args, tokenizer)
    num_train_optimization_steps = int(
        len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs)

    # logger.info('num_train_optimization_steps:', num_train_optimization_steps)
    # logger.info('train_dataloader:', len(train_dataloader))
    print ("num_train_optimization_steps:",num_train_optimization_steps)
    print ("len(train_dataloader):",len(train_dataloader))
    # Prepare model
    #初始化模型
    model = CailModel.from_pretrained(args.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #加载优化器的参数
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    #初始化优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate)
                         #warmup=args.warmup_proportion,
                         #t_total=num_train_optimization_steps)

    global_step = 0
    
    #开始训练
    model.train()
    f1 = 0
    #训练 num_train_epochs轮
    for epoch in range(int(args.num_train_epochs)):
        print ("epoch:",epoch)
        #训练len(data)/batch_size步
        for step, batch in enumerate(train_dataloader):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
            #从batch中重新读取各个特征
            input_ids, input_mask, segment_ids, start_positions, end_positions, \
            unk_mask, yes_mask, no_mask, word_feature_list, named_entity_list = batch
            #使用模型计算损失
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,
                         unk_mask, yes_mask, no_mask, word_feature_list, named_entity_list)
            if n_gpu > 1:
                loss = loss.mean()
                # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            #将损失进行后传
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            #每100步打印一次当前损失
            if step % 100 == 0:
                logger.info('step is {} and the loss is {:.2f}...'.format(step, loss))
            
            #每1000步进行一次评价
            if step % 1000 == 0 and step != 0 and epoch != 0:#
                # if step % 1000 == 0:
                # ignore the first epoch
                #进行模型评价
                metrics = _dev(args, device, model, eval_dataloader, eval_examples, eval_features)
                #如果当前的F1分数大于之前的F1分数，则保存模型
                if metrics['f1'] > f1:
                    f1 = metrics['f1']
                    save_model(args, model, tokenizer)
                logger.info("epoch is {} ,step is {}, f1 is {:.4f}, Rouge_L is {:.4f}, Exact match is {:.4f}, current_best is {:.4f}...".
                            format(epoch, step, metrics['f1'], metrics['Rouge_L'], metrics['exact_match'], f1))
                model.train()
                
        #训练结束进行一次评价
        metrics = _dev(args, device, model, eval_dataloader, eval_examples, eval_features)
        if metrics['f1'] > f1:
            f1 = metrics['f1']
            save_model(args, model, tokenizer)
        logger.info("epoch is {} ,step is {}, f1 is {:.4f}, Rouge_L is {:.4f}, Exact match is {:.4f}, current_best is {:.4f}...".
                            format(epoch, step, metrics['f1'], metrics['Rouge_L'], metrics['exact_match'], f1))
        model.train()

#设置配置项
import argparse

def config():
    parser = argparse.ArgumentParser()

    ## Required parameters
    #使用选项之后，模型会自动下载模型
    parser.add_argument("--bert_model", default="bert-base-chinese", type=str, #required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default="./output/", type=str,# required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default="data/big_train_data.json", type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file", default="data/data.json", type=str,
                        help="SQuAD json for dev. E.g., dev-v1.1.json or dev-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_class_test", action='store_true', help="Whether to run eval on the class test set.")
    parser.add_argument("--train_batch_size", default=6, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=6, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=40, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=60, type=int,#384
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        type=bool, default=True,
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)
    return args


def main():
    #加载配置
    args = config()
    
    #判断GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    #设置日志格式
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    #打印GPU信息
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    #如果梯度累计步数小于1
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    #根据梯度累计步数更新batch_size
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    #为GPU设置随机种子生成随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
    #训练
    if args.do_train:
        save_code(args.output_dir)
        _train(args, device, n_gpu)
        
    #测试
    if args.do_test:
        _test(args, device, n_gpu)
    
    #分类别的测试
    if args.do_class_test:
        _class_test(args, device, n_gpu)


if __name__ == "__main__":
    main()
