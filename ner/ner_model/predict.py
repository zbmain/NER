#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import os
import torch
import json
from tqdm import tqdm

# 完成中文文本数字化映射的函数
from ner_model.bilsm_crf import BiLSTM_CRF

def prepare_sequence(seq, char_to_id):
    char_ids = []
    for idx, char in enumerate(seq):
        # 判断若字符不在码表对应字典中, 则取NUK的编码(即unknown), 否则取对应的字符编码
        if char_to_id.get(char):
            char_ids.append(char_to_id[char])
        else:
            char_ids.append(char_to_id["<UNK>"])
    return torch.tensor(char_ids, dtype=torch.long)


# 单样本预测
def singel_predict(model_path,
                   content,
                   char_to_id_json_path,
                   embedding_dim,
                   hidden_dim,
                   target_type_list,
                   tag_to_id):
    char_to_id = json.load(open(char_to_id_json_path, mode="r", encoding="utf8"))
    model = BiLSTM_CRF(len(char_to_id), tag_to_id, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    tag_id_dict = {v: k for k, v in tag_to_id.items() if k[2:] in target_type_list}
    entities = []
    # 预测阶段不求解梯度 不更新参数
    with torch.no_grad():
        sentence_in = prepare_sequence(content, char_to_id)
        score, best_path_list = model(sentence_in)
        entity = None
        for char_idx, tag_id in enumerate(best_path_list):
            if tag_id in tag_id_dict:
                tag_index = tag_id_dict[tag_id][0]
                current_char = content[char_idx]
                if tag_index == "B":
                    entity = current_char
                elif tag_index == "I" and entity:
                    entity += current_char
            if tag_id == tag_to_id["O"] and entity:

                if "、" not in entity and "～" not in entity and "。" not in entity \
                        and "”" not in entity and "：" not in entity and ":" not in entity \
                        and "，" not in entity and "," not in entity and "." not in entity \
                        and ";" not in entity and "；" not in entity and "【" not in entity \
                        and "】" not in entity and "[" not in entity and "]" not in entity:
                    entities.append(entity)
                # 重置为空, 准备下一个实体的识别
                entity = None
    # 返回单条文本中所有识别出来的命名实体, 以集合的形式返回
    return set(entities)


def batch_predict(data_path,
                  model_path,
                  char_to_id_json_path,
                  embedding_dim,
                  hidden_dim,
                  target_type_list,
                  prediction_result_path,
                  tag_to_id):
    # 迭代路径, 读取文件名
    for fn in tqdm(os.listdir(data_path)):
        # 拼装全路径
        fullpath = os.path.join(data_path, fn)
        # 定义输出结果文件
        entities_file = open(os.path.join(prediction_result_path, fn.replace("txt", "csv")),
                             mode="w", encoding="utf-8")
        with open(fullpath, mode="r", encoding="utf-8") as f:
            # 读取文件
            content = f.readline()
            # 调用单个预测模型
            entities = singel_predict(model_path,
                                      content,
                                      char_to_id_json_path,
                                      embedding_dim,
                                      hidden_dim,
                                      target_type_list,
                                      tag_to_id)
            # 结果写入保存文件
            entities_file.write("\n".join(entities))
    print("batch_predict Finished".center(100, "-"))

if __name__ == '__main__':
    # 待识别文本
    content = ""

    # 模型保存路径
    model_path = "model/bilstm_crf_state_dict_saved.pt"

    # 词嵌入维度
    EMBEDDING_DIM = 200

    # 隐藏层维度
    HIDDEN_DIM = 100

    # 标签数字化映射字典
    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

    # 字符数字化映射字典的存储文件
    char_to_id_json_path = "./data/char_to_id.json"

    # 预测结果存储路径
    prediction_result_path = "prediction_result"

    # 待匹配标签类型
    target_type_list = ["sym"]

    entities = singel_predict(model_path,
                              content,
                              char_to_id_json_path,
                              EMBEDDING_DIM,
                              HIDDEN_DIM,
                              target_type_list,
                              tag_to_id)

    # 打印实体结果
    print("entities:\n", entities)


    #模型文件
    model_path = "model/bilstm_crf_state_dict_final.pt"
    # 模型原始参数
    # 字向量维度
    EMBEDDING_DIM = 200
    # 隐层维度
    HIDDEN_DIM = 100
    # 标签码表对照字典
    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}
    # 字符码表文件
    char_to_id_json_path = "./data/char_to_id.json"
    # 预测结果存储路径
    prediction_result_path = "prediction_result"
    # 待匹配标签类型
    target_type_list = ["sym"]
    # 待预测文件所在目录
    data_path = "./unstructured/norecognized/"

    batch_predict(data_path,
                  model_path,
                  char_to_id_json_path,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  target_type_list,
                  prediction_result_path,
                  tag_to_id)