# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:30:37 2019

@author: 10097
"""
import os
import pickle
import torch
import logging
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from prepare import read_squad_examples, convert_examples_to_features

logger = logging.getLogger(__name__)

def load_dev_features(args, tokenizer, is_class_test = False):
    #合成缓存数据文件的文件名
    cached_dev_features_file = os.path.join(args.output_dir, 'dev_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length)))
    #从验证集中读取数据
    eval_examples = read_squad_examples(
        input_file=args.dev_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
    # eval_examples = eval_examples[:20]
    
    #尝试读取缓存的数据文件
    eval_features = None
    
    if not is_class_test:
        try:
            with open(cached_dev_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            #将所有样例转化为SQUAD类型的特征
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            #保存为缓存文件
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving dev features into cached file %s", cached_dev_features_file)
                with open(cached_dev_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)
    else:
        #将所有样例转化为SQUAD类型的特征
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
    
    #生成验证集特征
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_word_feature_list2 = torch.tensor([f.word_feature_list for f in eval_features], dtype=torch.float)
    all_named_entity_list2 = torch.tensor([f.named_entity_list for f in eval_features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_word_feature_list2, all_named_entity_list2, all_example_index)
    # Run prediction for full data
    #将数据打包为验证集
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    logger.info("***** Eval *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    return eval_dataloader, eval_examples, eval_features


def load_train_features(args, tokenizer):
    #缓存的训练特征文件  output_dir/train____
    cached_train_features_file = os.path.join(args.output_dir, 'train_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length)))
    
    #读取训练样本，从train_file中
    train_examples = read_squad_examples(
        input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)

    # train_examples = train_examples[:20]
    train_features = None
    try:
        #如果已经有缓存文件，则从缓存文件中读取train_features
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        #如果没有缓存文件，则使用convert_examples_to_features函数获得train_features
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        #读取完毕将其加载到缓存文件中
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    #打印训练相关信息
    logger.info("***** Train *****")
    logger.info("  Num orig examples = %d", len(train_examples))  #原始的样本数
    logger.info("  Num split examples = %d", len(train_features))  #划分后的样本数
    logger.info("  Batch size = %d", args.train_batch_size)   #batch_size大小
    
    #加载特征：Bert三输入：词典映射ID、下一句映射ID、位置ID，以及unk和yes_no掩码
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    all_unk_mask = torch.tensor([f.unk_mask for f in train_features], dtype=torch.long)
    all_yes_mask = torch.tensor([f.yes_mask for f in train_features], dtype=torch.long)
    all_no_mask = torch.tensor([f.no_mask for f in train_features], dtype=torch.long)
    all_word_feature_list2 = torch.tensor([f.word_feature_list for f in train_features], dtype=torch.float)
    all_named_entity_list2 = torch.tensor([f.named_entity_list for f in train_features], dtype=torch.float)
    
    #打包成为训练数据
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions,
                               all_unk_mask, all_yes_mask, all_no_mask, all_word_feature_list2, all_named_entity_list2)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    return train_dataloader


def load_test_features(args, tokenizer):
    test_examples = read_squad_examples(
        input_file=args.test_file, is_training=False, version_2_with_negative=args.version_2_with_negative)

    test_features = convert_examples_to_features(
        examples=test_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)    
    all_word_feature_list2 = torch.tensor([f.word_feature_list for f in test_features], dtype=torch.float)
    all_named_entity_list2 = torch.tensor([f.named_entity_list for f in test_features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_word_feature_list2, all_named_entity_list2, all_example_index)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)

    logger.info("***** Test *****")
    logger.info("  Num orig examples = %d", len(test_examples))
    logger.info("  Num split examples = %d", len(test_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    return test_dataloader, test_examples, test_features