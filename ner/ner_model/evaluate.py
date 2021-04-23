#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import torch
import torch.nn as nn

def evaluate(sentence_list, gold_tags, predict_tags, id2char, id2tag):
    gold_entities, gold_entity = [], []
    # 预测实体集合以及每个实体的字与标签列表
    predict_entities, predict_entity = [], []
    # 遍历当前待评估的句子列表
    for line_no, sentence in enumerate(sentence_list):
        # 迭代句子每个字符
        for char_no in range(len(sentence)):
            # 判断:若句子的id值对应的是0(即:<PAD>)则跳过循环
            if sentence[char_no] == 0:
                break
            # 获取当前句子中的每一个文字
            char_text = id2char[sentence[char_no]]
            # 获取当前字真实实体标注类型
            gold_tag_type = id2tag[gold_tags[line_no][char_no]]
            # 获取当前预测实体标注类型
            predict_tag_type = id2tag[predict_tags[line_no][char_no]]
            # 判断 id2tag 第一个字符是否为 B ,表示实体开始
            if gold_tag_type[0] == "B":
                # 将实体文字与类别加入实体列表中
                gold_entity = [char_text + "/" + gold_tag_type]
            # 判断 id2tag 第一个字符是否为 I ,表示实体中部到结尾
            # 判断依据: I 开头; entiry 不为空; 实体类别相同.
            elif gold_tag_type[0] == "I" and len(gold_entity) != 0 \
                    and gold_entity[-1].split("/")[1][1:] == gold_tag_type[1:]:
                # 满足条件则加入实体列表中
                gold_entity.append(char_text + "/" + gold_tag_type)
            # 判断依据: O 开头, 并且entiry 不为空.
            elif gold_tag_type[0] == "O" and len(gold_entity) != 0:
                # 增加唯一标识(人为增加的标志位)
                gold_entity.append(str(line_no) + "_" + str(char_no))
                # 将实体文字与类别加入实体列表中
                gold_entities.append(gold_entity)
                # 重置实体列表
                gold_entity = []
            else:
                # 重置实体列表
                gold_entity = []

            # 判断 id2tag 第一个字符是否为 B ,表示实体开始
            if predict_tag_type[0] == "B":
                # 将实体文字与类别加入实体列表中
                predict_entity = [char_text + "/" + predict_tag_type]
            # 判断 id2tag 第一个字符是否为 I ,表示实体中部到结尾
            # 判断依据: I 开头; entiry 不为空; 实体类别相同.
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 \
                    and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                # 将实体文字与类别加入实体列表中
                predict_entity.append(char_text + "/" + predict_tag_type)
            # 判断依据: O 开头, 并且entiry 不为空.
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                # 增加唯一标识(人为添加的标志位)
                predict_entity.append(str(line_no) + "_" + str(char_no))
                # 将实体加入列表中
                predict_entities.append(predict_entity)
                # 重置实体列表
                predict_entity = []
            else:
                # 重置实体列表
                predict_entity = []

    # 获取预测正确的实体集合
    acc_entities = [entity for entity in predict_entities if entity in gold_entities]
    # 预测正确实体长度(用于计算准确率, 召回, F1值)
    acc_entities_length = len(acc_entities)
    # 预测出实体个数
    predict_entities_length = len(predict_entities)
    # 真实实体列表个数
    gold_entities_length = len(gold_entities)

    # 如果准确实体个数大于0, 则计算准确度, 召回率, f1值
    if acc_entities_length > 0:
        # 准确率
        step_acc = float(acc_entities_length / predict_entities_length)
        # 召回率
        step_recall = float(acc_entities_length / gold_entities_length)
        # f1值
        f1_score = 2 * step_acc * step_recall / (step_acc + step_recall)
        # 返回评估值与各个实体长度(用于整体计算)
        return step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, gold_entities_length
    else:
        # 否则准确率, 召回率, f1都为为0
        return 0, 0, 0, acc_entities_length, predict_entities_length, gold_entities_length


def sentence_map(sentence_list, char_to_id, max_length):
    sentence_list.sort(key=lambda c: len(c), reverse=True)
    sentence_map_list = []
    for sentence in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sentence]
        padding_list = [0] * (max_length - len(sentence))
        sentence_id_list.extend(padding_list)
        sentence_map_list.append(sentence_id_list)
    return torch.tensor(sentence_map_list, dtype=torch.long)


char_to_id = {"<PAD>": 0}

SENTENCE_LENGTH = 20

for sentence in sentence_list:
    for _char in sentence:
        if _char not in char_to_id:
            char_to_id[_char] = len(char_to_id)

sentences_sequence = sentence_map(sentence_list, char_to_id, SENTENCE_LENGTH)

if __name__ == '__main__':
    # 编码与字符对照字典，引入更全的id2char
    id2char = {0: '<PAD>', 1: '爆', 2: '胎'}
    # 编码与标签对照字典
    id2tag = {0: 'O', 1: 'B-dis', 2: 'I-dis', 3: 'B-sym', 4: 'I-sym'}
    accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length = evaluate(
        sentences_sequence.tolist(), tag_list, predict_tag_list, id2char, id2tag)

    print("accuracy:", accuracy,
          "\nrecall:", recall,
          "\nf1_score:", f1_score,
          "\nacc_entities_length:", acc_entities_length,
          "\npredict_entities_length:", predict_entities_length,
          "\ntrue_entities_length:", true_entities_length)
