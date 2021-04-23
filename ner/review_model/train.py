#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

from zbmain.utils import time as ztime
from zbmain.utils import path
from zbmain.utils import args
from collections import Counter
from torch import nn
import matplotlib.pyplot as plt
import torch
import os, sys, random
import pandas as pd
from RNN_MODEL import RNN

# ----加载数据----


train_data_path = './train_data.csv'
train_data = pd.read_csv(train_data_path, header=None, sep='\t')
# UT
print(dict(Counter(train_data[0].values)))  # 查看正负标签比例
train_data = train_data.values.tolist()
# 查看head10
print(train_data[:10])


# ----加载预训练模型----
base_path = 'huggingface/pytorch-transformers'
model_name = 'bert-base-chinese'
model = torch.hub.load(base_path, 'model', model_name)
tokenizer = torch.hub.load(base_path, 'tokenizer', model_name)


# ----构建Bert中文编码函数
def get_bert_encode_for_single(text: str):
    indexed_tokens = tokenizer.encode(text)[1:-1]
    token_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        encoded_layers, _ = model(token_tensor)
    # print(encoded_layers.shape)
    encoded_layers = encoded_layers[0]
    return encoded_layers


# UT
# outputs = get_bert_encode_for_single('你好，周杰伦')
# print(outputs)
# print(outputs.shape)


# ----模型训练----
# 1.构建随机获取数据函数
def randomTrainingExample(train_data):
    category, line = random.choice(train_data)
    line_tensor = get_bert_encode_for_single(line)
    category_tensor = torch.tensor([int(category)])
    return category, line, category_tensor, line_tensor


# UT
# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
#     print('category =', category, '/ line =', line)

# 2.构建训练train函数
criterion = nn.NLLLoss()
learning_rate = 5e-3


def train(category_tensor, line_tensor, rnn: RNN):
    # 隐层初始化
    hidden = rnn.initHidden()
    # 梯度归零
    rnn.zero_grad()
    # 遍历每个字的张量
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    loss = criterion(output, category_tensor)
    # 误差反向传播
    loss.backward()
    # 更新参数
    for param in rnn.parameters():
        param.data.add_(-learning_rate, param.grad.data)
    return output, loss.item()


# 4.构建验证函数
def valid(category_tensor, line_tensor, rnn: RNN):
    # 隐层初始化
    hidden = rnn.initHidden()
    # 验证不求梯度
    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
        # 计算损失
        loss = criterion(output, category_tensor)
    # return 结果和损失
    return output, loss.item()


# 5.训练调用函数：训练、验证、查看、保存
def run(rnn: RNN, n_iters: int = 50000, plot_every: int = 1000):
    train_current_loss = 0
    train_current_acc = 0
    valid_current_loss = 0
    valid_current_acc = 0

    global all_train_losses
    global all_valid_losses
    global all_train_acc
    global all_valid_acc
    all_train_losses = []
    all_train_acc = []
    all_valid_losses = []
    all_valid_acc = []

    start = ztime.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
        category_, line_, category_tensor_, line_tensor_ = randomTrainingExample(train_data)
        train_output, train_loss = train(category_tensor, line_tensor, rnn)
        valid_output, valid_loss = valid(category_tensor_, line_tensor_, rnn)
        # 损失&&准确率累加
        train_current_loss += train_loss
        train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
        valid_current_loss += valid_loss
        valid_current_acc += (valid_output.argmax(1) == category_tensor_).sum().item()

        if (iter % plot_every == 0):
            train_average_loss = train_current_loss / plot_every
            train_average_acc = train_current_acc / plot_every
            valid_average_loss = valid_current_loss / plot_every
            valid_average_acc = valid_current_acc / plot_every
            # 打印查看
            print("Iter:", iter, "|", "TimeSince:", ztime.timeSince(start))
            print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
            print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)
            # 保存结果
            all_train_losses.append(train_average_loss)
            all_train_acc.append(train_average_acc)
            all_valid_losses.append(valid_average_loss)
            all_valid_acc.append(valid_average_acc)
            # 本轮清0
            train_current_loss = 0
            train_current_acc = 0
            valid_current_loss = 0
            valid_current_acc = 0

            pltImg(iter)  # 绘制记录
            save_model(rnn, f'./BERT_RNN_n{iter}.pth')  # 保存记录

    pltImg()
    save_model(rnn)  # 保存最终模型


# 6.绘制表：损失和准确率
def pltImg(timestamp:str=''):
    global all_train_losses
    global all_valid_losses
    global all_train_acc
    global all_valid_acc

    plt.figure(0)
    plt.plot(all_train_losses, label="Train Loss")
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"loss {timestamp}.png")

    plt.figure(1)
    plt.plot(all_train_acc, label="Train Acc")
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    plt.legend(loc='upper left')
    plt.savefig(f"acc {timestamp}.png")


# 7.保存模型
def save_model(rnn: RNN, path: str = './BERT_RNN.pth'):
    torch.save(rnn.state_dict(), path)


# ----模型的调用使用----
# 预测功能函数
def _test(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    return output


# 预测调用函数
def predict(input_line):
    # 预测不更新梯度
    with torch.no_grad():
        # 文本bert编码
        output = _test(get_bert_encode_for_single(input_line))
        # 输出列表中的最大值
        _, topi = output.topk(1, 1)
        return topi.item()


# ----模型实际使用----
# 处理数据文件
def batch_predict(in_path, out_path):
    csv_list = os.listdir(in_path)
    for csv in csv_list:
        with open(os.path.join(in_path, csv), "r") as fr:
            with open(os.path.join(out_path, csv), "w") as fw:
                input_line = fr.readline()
                res = predict(input_line)
                # 审核通过的正样本
                if res:
                    # 审核成功,写入csv
                    fw.write(input_line + "\n")
                else:
                    pass
