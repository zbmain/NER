#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import torch
import torch.nn as nn

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')


def get_bert_encode_for_single(text: str):
    # 使用huggingface字符映射器对每个字进行映射
    indexed_tokens = tokenizer.encode(text)[1:-1]  # 切除首尾：101、102
    # 列表转化为tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    print(tokens_tensor)
    # 不求梯度
    with torch.no_grad():
        # 模型隐层输出
        encoded_layers, _ = model(tokens_tensor)
    print(encoded_layers.shape)
    # 3维度张量，降一维
    encoded_layers = encoded_layers[0]
    return encoded_layers

if __name__ == '__main__':
    text = "你好，祖国！"
    outputs = get_bert_encode_for_single(text)
    print(outputs)
    print(outputs.shape)
