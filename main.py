#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Time    : 17-4-29 下午11:23
  @Author  : LinLifang
  @File    : main.py
"""
from core.reader import *
from config.config import ConfigTrain, ConfigTest
from api.chatbot import ChatBot
from core.reader import save_seq2seq


def train():
    train_set, test_set = read_data()
    config = ConfigTrain()
    print('开始训练...')
    model = ChatBot(config)
    model.train(train_set, test_set)


def test():
    vocab = get_char2id()
    vocab_dec = reverse_dict(vocab)
    config = ConfigTest()
    model = ChatBot(config)
    model.test(vocab, vocab_dec)


def convert_data():
    save_seq2seq()


name_dict = {'训练': 'train()', '测试': 'test()'}


def main(run_name):
    run = name_dict.get(run_name, run_name)
    exec(run)


if __name__ == '__main__':
    # main('训练')
    main('测试')
