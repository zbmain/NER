#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

from torch import nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # 输出hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出output_size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        '''隐含层初始化功能函数'''
        return torch.zeros(1, self.hidden_size)  # 1维度
