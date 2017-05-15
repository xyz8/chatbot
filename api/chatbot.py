#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Time    : 17-4-29 下午11:23
  @Author  : LinLifang
  @File    : chatbot.py
"""
import datetime
import tensorflow as tf
import numpy as np
import math
from config.config import *
from api.seq2seq_model import Seq2SeqModel
# from tensorflow.models.rnn.translate.seq2seq_model import Seq2SeqModel


class ChatBot(object):
    def __init__(self, config):
        self.buckets = config.buckets
        self.model = Seq2SeqModel(source_vocab_size=config.vocabulary_size,
                                  target_vocab_size=config.vocabulary_size,
                                  buckets=config.buckets,
                                  size=config.layer_size,
                                  num_layers=config.num_layers,
                                  max_gradient_norm=config.max_gradient_norm,
                                  batch_size=config.batch_size,
                                  learning_rate=config.learn_rate,
                                  learning_rate_decay_factor=config.decay_factor,
                                  forward_only=config.forward_only)

    def train(self, train_set, test_set):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:
            # 恢复前一次训练
            ckpt = tf.train.get_checkpoint_state('./ckpt')
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            train_bucket_sizes = [len(train_set[b]) for b in range(len(self.buckets))]
            train_total_size = float(sum(train_bucket_sizes))
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in
                                   range(len(train_bucket_sizes))]

            loss = 0.0
            total_step = 0
            previous_losses = []
            # 训练, 每1000步保存一次模型！
            while True:
                random_number_01 = np.random.random_sample()
                bucket_id = min(
                    [i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

                encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(train_set, bucket_id)
                _, step_loss, _ = self.model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                  False)

                loss += step_loss / 1000
                total_step += 1
                if total_step % 100 == 0:
                    print('迭代步数: ', total_step)
                if total_step % 1000 == 0:
                    start = datetime.datetime.now()
                    print('\n----------------------------------\n')
                    print('*保存模型...\n*step: %s' % self.model.global_step.eval())
                    print('*loss:%s\n*time: %s\n' % (loss, start))
                    print('----------------------------------\n')
                    # if model has't not improve，decrese the learning rate
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(self.model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # 保存模型
                    checkpoint_path = "./ckpt/chatbot_seq2seq.ckpt"
                    self.model.saver.save(sess, checkpoint_path, global_step=self.model.global_step)
                    loss = 0.0
                    # evaluation the model by test dataset
                    for bucket_id in range(len(self.buckets)):
                        if len(test_set[bucket_id]) == 0:
                            continue
                        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(test_set, bucket_id)
                        _, eval_loss, _ = self.model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                          bucket_id, True)
                        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        print('验证集: bucket_id: %s\t eval_ppx: %s' % (bucket_id, eval_ppx))

    def test(self, vocab, vocab_dec):
        self.model.batch_size = 1
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./ckpt/')
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("model not found")
                exit()

            while True:
                input_string = input('me > ')
                if input_string == 'q':
                    exit()
                # 转化为矢量 input_string_vec=[23,62,58,1990]
                input_string_vec = []
                for words in input_string.strip():
                    input_string_vec.append(vocab.get(words, UNK_ID))
                # 获取bucked_id -> [(5, 10), (10, 15), (20, 25), (40, 50)]
                bucket_id = min([b for b in range(len(self.buckets)) if self.buckets[b][0] > len(input_string_vec)])
                # 测试
                test_data = {bucket_id: [(input_string_vec, [])]}
                encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(test_data, bucket_id)
                _, _, output_logits = self.model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                      True)

                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                if EOS_ID in outputs:
                    outputs = outputs[:outputs.index(EOS_ID)]

                response = "".join([tf.compat.as_str(vocab_dec[str(output)]) for output in outputs])
                print('AI > ' + response)
