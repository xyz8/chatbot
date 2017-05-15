#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Time    : 17-4-29 下午11:21
  @Author  : LinLifang
  @File    : config.py
"""
import os

# 参数配置
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
start_char = [('__PAD__', 0), ('__GO__', 0), ('__EOS__', 0), ('__UNK__', 0)]

# 路径
data_path = os.path.join(os.getcwd(), 'data/dgk_shooter_min.conv')
ask_path = os.path.join(os.getcwd(), 'data/ask')
res_path = os.path.join(os.getcwd(), 'data/res')
char_id_path = os.path.join(os.getcwd(), 'data/char2id')


class ConfigTrain(object):
    layer_size = 256
    num_layers = 3
    batch_size = 64
    learn_rate = 0.5
    decay_factor = 0.97
    max_gradient_norm = 5.0
    vocabulary_size = 5000
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    forward_only = False


class ConfigTest(object):
    layer_size = 256
    num_layers = 3
    batch_size = 1
    learn_rate = 0.5
    decay_factor = 0.99
    max_gradient_norm = 5.0
    vocabulary_size = 5000
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    forward_only = True
