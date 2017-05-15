#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Time    : 17-4-29 下午11:40
  @Author  : LinLifang
  @File    : reader.py
"""
import os
import random

import collections

import copy

from config.config import *


def format_data(filename):
    """
    函数说明: 读取数据源
    :param filename: 数据源文件名
    :return:
    """
    convs = []
    with open(filename, 'r', encoding='utf-8') as fp:
        one_conv = []
        for line in fp.readlines():
            line = line.strip().replace('/', '')
            if not line:
                continue
            if line[0] == 'E':
                convs.append(one_conv) if one_conv else None
                one_conv = []
            elif line[0] == 'M':
                one_conv.append(line.split(' ')[1])
    return convs


def ask_response(path):
    """
    函数说明: 对话转化为问答集
    :return: ask, response
    """
    convs = format_data(path)
    ask = []
    response = []
    for conv in convs:
        if len(conv) == 1:
            continue
        if len(conv) % 2 != 0:
            conv.pop()
        for i, s in enumerate(conv):
            ask.append(s) if i % 2 == 0 else response.append(s)
    return ask, response


def convert_seq2seq_files(seqs, filename):
    """
    函数说明: 保存问答集
    :param seqs:
    :param filename:
    :return:
    """
    with open(filename, 'w') as fp:
        seq_str = '\n'.join(seqs)
        fp.write(seq_str + '\n')


def convert_char2id(seqs, char2id_path):
    """
    函数说明: 保存char2id
    :param seqs:
    :param char2id_path:
    :return:
    """
    chars = []
    for seq in seqs:
        seq_list = [kk for kk in seq]
        chars.extend(seq_list)
    count = copy.deepcopy(start_char)
    count.extend(collections.Counter(chars).most_common(5000-4))

    words = [word for (word, _) in count]
    char2id = [w + '\t' + str(i) for (i, w) in enumerate(words)]
    with open(char2id_path, 'w') as fp:
        fp.write('\n'.join(char2id))


def save_seq2seq():
    """
    函数说明: 保存对话集, char_id
    :return:
    """
    ask, res = ask_response(data_path)
    convert_char2id(ask + res, char_id_path)
    convert_seq2seq_files(ask, ask_path)
    convert_seq2seq_files(res, res_path)


def read_data(ratio=0.99):
    """
    函数说明: 获取对话向量集
    :param ratio: 训练集和验证集比例
    :return:
    """
    print('获取数据集...\n')
    ask = []
    res = []
    with open(ask_path, 'r') as fp:
        for line in fp.readlines():
            ask.append(line.strip())
    with open(res_path, 'r') as fp:
        for line in fp.readlines():
            res.append(line.strip())

    vocab = get_char2id()
    ask_vector = convert_vector(ask, vocab)
    res_vector = convert_vector(res, vocab, end=True)

    index = int(len(ask_vector) * ratio)
    train_enc, train_dec = ask_vector[:index], res_vector[:index]
    test_enc, test_dec = ask_vector[index:], res_vector[index:]

    train_set = get_data(zip(train_enc, train_dec))
    test_set = get_data(zip(test_enc, test_dec))
    return train_set, test_set


def get_char2id():
    """
    函数说明: 获取字符-id
    :return:
    """
    char_id = []
    with open(char_id_path) as fp:
        for line in fp.readlines():
            char_id.append(line.strip().split())
    char_id = [kk for kk in char_id if len(kk) == 2]
    vocab = dict(char_id)
    return vocab


def convert_vector(seqs, vocab, end=None):
    """
    函数说明: 对话集转换为矢量
    :param seqs: 对话集
    :param vocab: char-id
    :param end: 是否为对话结束
    :return:
    """
    seqs_vector = []
    for line in seqs:
        line_vec = []
        for word in line:
            line_vec.append(int(vocab.get(word, UNK_ID)))
        if end:
            line_vec.append(EOS_ID)
        seqs_vector.append(line_vec)
    return seqs_vector


def get_data(seqs):
    """
    函数说明: 获取对话向量
    :param seqs: 对话集
    :return:
    """
    data_set = [[] for _ in buckets]
    for source, target in seqs:
        for bucket_id, (source_size, target_size) in enumerate(buckets):
            if len(source) < source_size and len(target) < target_size:
                data_set[bucket_id].append([source, target])
                break
    return data_set


def reverse_dict(data_dict):
    array = [(j, i) for i, j in data_dict.items()]
    result_dict = dict(array)
    return result_dict
