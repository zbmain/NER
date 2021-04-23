#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import torch
import torch.nn as nn

torch.manual_seed(1)


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    # 找出最大值
    max_score = vec[0, argmax(vec)]
    # 数转张量
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 防止数字溢出,先减去最大值的张量,运算之后再在原数加上最大值
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, start_tag='<START>', stop_tag='<STOP>'):
        super(BiLSTM_CRF, self).__init__()
        self.START_TAG = start_tag
        self.STOP_TAG = stop_tag
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 实例化双向LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # 全连接线性层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 转移矩阵
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # 定义限制条件，限制标签的合法性，
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def _get_lstm_features(self, sentence):
        # 初始化隐张量
        self.hidden = self.init_hidden()
        # 词嵌入，样本转三维张量
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        # 返回发射矩阵的张量
        return lstm_feats

    def _forward_alg(self, feats):
        # 全部负化张量
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 开口：START
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.STOP_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        # 真实句子分数
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # 合法序列必须从START开始
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 遍历完所有序列，最后加上结束标签的转移分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        # print(self.tag_to_ix,'~',self.START_TAG,start)
        # 检测正确性
        assert start == self.tag_to_ix[self.START_TAG]

        best_path.reverse()
        return path_score, best_path

    # 计算损失loss
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # 预测阶段使用的前向传播函数
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

def sentence_map(sentence_list):
    # 初始化数字映射字典
    word_to_ix = {}
    for sentence, tags in sentence_list:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    return word_to_ix

if __name__ == '__main__':
    # 首先设置若干超参数
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 100
    VOCAB_SIZE = 30
    # 初始化标签字典
    tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}

    # 输入样本数据
    sentence_list = [
        (["今","早","上","车","后","，","发","现","汽","车","爆","胎"],
         ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-sym", "I-sym"]
         )]

    word_to_ix = sentence_map(sentence_list)
    # 创建类对象
    model = BiLSTM_CRF(vocab_size=len(word_to_ix), tag_to_ix=tag_to_ix,
                       embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)

    # 遍历样本数据
    for sentence, tags in sentence_list:
        # 完成自然语言文本到数字化张量的映射
        sentence_in = prepare_sequence(sentence, word_to_ix)
        # 调用类内函数完成发射矩阵张量的计算
        feat_out = model._get_lstm_features(sentence_in)
        print(feat_out)
        print(feat_out.shape)
        print('******')

    # 遍历样本数据
    for sentence, tags in sentence_list:
        # 完成自然语言文本到数字化张量的映射
        sentence_in = prepare_sequence(sentence, word_to_ix)
        # 调用类内函数完成发射矩阵张量的计算
        feat_out = model._get_lstm_features(sentence_in)
        forward_score = model._forward_alg(feat_out)
        print(forward_score)

    # 遍历样本数据
    for sentence, tags in sentence_list:
        # 完成自然语言文本到数字化张量的映射
        sentence_in = prepare_sequence(sentence, word_to_ix)
        # 调用类内函数完成发射矩阵张量的计算
        feat_out = model._get_lstm_features(sentence_in)
        forward_score = model._forward_alg(feat_out)
        # 将tags标签进行数字化映射, 然后封装成torch.long型的Tensor张量
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        # 计算真实句子分数
        gold_score = model._score_sentence(feat_out, targets)
        print(gold_score)
        # 遍历样本数据
        for sentence, tags in sentence_list:
            # 完成自然语言文本到数字化张量的映射
            sentence_in = prepare_sequence(sentence, word_to_ix)
            # 调用类内函数完成发射矩阵张量的计算
            feat_out = model._get_lstm_features(sentence_in)
            forward_score = model._forward_alg(feat_out)
            # 将tags标签进行数字化映射, 然后封装成torch.long型的Tensor张量
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # 计算真实句子分数
            gold_score = model._score_sentence(feat_out, targets)
            # 利用维特比算法解析最佳路径
            score, tag_seq = model._viterbi_decode(feat_out)
            print(tag_seq)
            print(score)
            print('******')
    # 遍历样本数据
    for sentence, tags in sentence_list:
        # 完成自然语言文本到数字化张量的映射
        sentence_in = prepare_sequence(sentence, word_to_ix)
        # 调用类内函数完成发射矩阵张量的计算
        feat_out = model._get_lstm_features(sentence_in)
        # 将tags标签进行数字化映射, 然后封装成torch.long型的Tensor张量
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 直接调用损失值计算函数, 计算loss值
        loss = model.neg_log_likelihood(sentence_in, targets)
        print('loss=', loss)

        # 进入预测阶段, 调用模型默认的forward()函数
        with torch.no_grad():
            # 模型默认的函数就是预测阶段的维特比解码函数
            score, tag_seq = model(sentence_in)
            print(score)
            print(tag_seq)
            print('******')
