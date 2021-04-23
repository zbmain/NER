#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_to_id, input_feature_size, hidden_size, batch_size, sentence_length,
                 num_layers=1, batch_first=True):
        '''
        模型初始化
        :param vocab_size:           所有不同字符的个数
        :param tag_to_id:            标签2ID
        :param input_feature_size:   字嵌入维度
        :param hidden_size:          隐藏层向量维度
        :param batch_size:           批次训练大小
        :param sentence_length:      句子长度
        :param num_layers:           堆叠LSTM层数
        :param batch_first:          是否batch——size防止到矩阵第一维度
        '''
        super(BiLSTM, self).__init__()
        self.tag_to_id = tag_to_id
        self.tag_size = len(tag_to_id)
        self.embedding_size = input_feature_size
        self.hidden_size = hidden_size // 2  # 双向LSTM
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        # 创建BiLSTM实例
        self.bilstm = nn.LSTM(input_size=input_feature_size,
                              hidden_size=self.hidden_size,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, self.tag_size)

    def forward(self, sentences_sequence):
        #初始化隐层状态
        h0 = torch.randn(self.num_players * 2, self.batch_size, self.hidden_size)
        #初始化单元状态
        c0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        #生成字向量
        input_features = self.embedding(sentences_sequence)
        #字向量与状态初始值传入LSTM模型结构计算：
        output, (hn, vn) = self.bilstm(input_features, (h0, c0))
        #将输出特征进行线性变换
        sequence_features = self.linear(output)
        return sequence_features
