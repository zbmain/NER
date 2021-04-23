#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

import json
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

from bilsm_crf import BiLSTM_CRF
from evaluate import evaluate


def prepare_sequence(seq, char_to_id):
    char_ids = []
    for idx, char in enumerate(seq):
        if char_to_id.get(char):
            char_ids.append(char_to_id[char])
        else:
            char_ids.append(char_to_id['UNK'])
    return torch.tensor(char_ids, dtype=torch.long)


def get_data():
    train_data_file_path = './data/train_data.txt'
    validate_file_path = './data/validate_data.txt'
    train_data_list = []
    validate_data_list = []

    # print(open(train_data_file_path).readlines())
    lines = open(train_data_file_path, mode='r', encoding='utf-8').readlines()
    for line in lines:
        data = json.loads(line)
        train_data_list.append(data)
    for line in open(validate_file_path, mode='r', encoding='utf-8').readlines():
        data = json.loads(line)
        validate_data_list.append(data)
    return train_data_list, validate_data_list


def save_train_history_image(train_history_list,
                             validate_history_list,
                             history_image_path,
                             data_type):
    # 存储训练历史图片
    plt.plot(train_history_list, label="Train %s History" % (data_type))
    plt.plot(validate_history_list, label="Validate %s History" % (data_type))
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel(data_type)
    plt.savefig(history_image_path.replace("plot", data_type))
    plt.close()


if __name__ == '__main__':
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 100
    epochs = 10  # 训练批次数

    train_data_list, validate_data_list = get_data()
    # 字符到id的映射提前通过全样本统计完了, 也可以放在单独的函数中构建char_to_id
    char_to_id = json.load(open('./data/char_to_id.json', mode='r', encoding='utf-8'))
    # 标签到id的映射是由特定任务决定的, 任务一旦确定那么这个字典也就确定了, 因此也以超参数的形式提前准备好
    tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

    # 直接构建模型对象
    model = BiLSTM_CRF(len(char_to_id), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    # 直接选定优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.85, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 调转字符标签与id值
    id_to_tag = {v: k for k, v in tag_to_ix.items()}
    # 调转字符编码与id值
    id_to_char = {v: k for k, v in char_to_id.items()}

    # 获取时间戳标签, 用于模型文件和图片, 日志文件的保存命名
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    model_saved_path = "model/bilstm_crf_state_dict_%s.pt" % (time_str)
    train_history_image_path = "log/bilstm_crf_train_plot_%s.png" % (time_str)
    log_file = open("log/train_%s.log" % (time_str), mode="w", encoding="utf8")

    # 初始化
    train_loss_history, train_acc_history, train_recall_history, train_f1_history = [], [], [], []
    validate_loss_history, validate_acc_history, validate_recall_history, validate_f1_history = [], [], [], []

    # 按epoch进行迭代训练
    for epoch in range(epochs):
        tqdm.write("Epoch {}/{}".format(epoch + 1, epochs))
        total_acc_length, total_prediction_length, total_gold_length, total_loss = 0, 0, 0, 0
        # 每一个epoch先遍历训练集, 通过训练样本更新模型
        for train_data in tqdm(train_data_list):
            model.zero_grad()
            # 取出原始样本数据
            sentence, tags = train_data.get("text"), train_data.get("label")
            # 完成数字化编码的映射
            sentence_in = prepare_sequence(sentence, char_to_id)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # 直接将特征和标签送入模型中计算loss
            # model中真实计算损失值的函数不是forward(), 而是自定义的neg_log_likelihood()
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            # 这一步才真正的调用forward()函数, 目的是通过维特比算法解码出最佳路径, 本质上是一个预测的过程
            score, best_path_list = model(sentence_in)
            # 累加每一个样本的损失值
            step_loss = loss.data.numpy()
            total_loss += step_loss
            sentence_in_unq = sentence_in.unsqueeze(0)
            targets_unq = targets.unsqueeze(0)
            best_path_list_up = [best_path_list]
            step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, gold_entities_length = \
                evaluate(sentence_in_unq.tolist(), targets_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)
            # 累加三个重要的统计量，预测正确的实体数，预测的总实体数，真实标签的实体数
            total_acc_length += acc_entities_length
            total_prediction_length += predict_entities_length
            total_gold_length += gold_entities_length

        print("train:", total_acc_length, total_prediction_length, total_gold_length)
        # 如果for循环结束, 真实预测的实体数大于0, 则计算准确率, 召回率, F1等统计量值
        if total_prediction_length > 0:
            train_mean_loss = total_loss / len(train_data_list)
            train_epoch_acc = total_acc_length / total_prediction_length
            train_epoch_recall = total_acc_length / total_gold_length
            train_epoch_f1 = 2 * train_epoch_acc * train_epoch_recall / (train_epoch_acc + train_epoch_recall)
        else:  # 否则0
            log_file.write("train_total_prediction_length is 0" + "\n")

        # 每一个epoch, 训练结束后直接在验证集上验证
        total_acc_length, total_prediction_length, total_gold_length, total_loss = 0, 0, 0, 0
        # 验证阶段，不反向传播和更新参数
        with torch.no_grad():
            for validate_data in tqdm(validate_data_list):
                # 直接提取验证集的特征和标签, 并进行数字化映射
                sentence, tags = validate_data.get("text"), validate_data.get("label")
                sentence_in = prepare_sequence(sentence, char_to_id)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                # 验证阶段的损失值 neg_log_likelihood函数计算
                loss = model.neg_log_likelihood(sentence_in, targets)
                # 验证阶段的解码 forward函数
                score, best_path_list = model(sentence_in)
                # 累加每个样本的损失值
                step_loss = loss.data.numpy()
                total_loss += step_loss
                sentence_in_unq = sentence_in.unsqueeze(0)
                targets_unq = targets.unsqueeze(0)
                best_path_list_up = [best_path_list]
                step_acc, step_recall, f1_score, acc_entities_length, predict_entities_length, gold_entities_length = evaluate(
                    sentence_in_unq.tolist(), targets_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)
                # 累加三个重要的统计量, 预测正确的实体数, 预测的总实体数, 真实标签的实体数
                total_acc_length += acc_entities_length
                total_prediction_length += predict_entities_length
                total_gold_length += gold_entities_length

        print("validate:", total_acc_length, total_prediction_length, total_gold_length)
        # 当准确预测的数量大于0, 并且总的预测标签量大于0, 计算验证集上的准确率, 召回率, F1值
        if total_acc_length != 0 and total_prediction_length != 0:
            validate_mean_loss = total_loss / len(validate_data_list)
            validate_epoch_acc = total_acc_length / total_prediction_length
            validate_epoch_recall = total_acc_length / total_gold_length
            validate_epoch_f1 = 2 * validate_epoch_acc * validate_epoch_recall / (
                    validate_epoch_acc + validate_epoch_recall)
            log_text = "Epoch: %s | train loss: %.5f |train acc: %.3f |train recall: %.3f |train f1 score: %.3f" \
                       " | validate loss: %.5f |validate acc: %.3f |validate recall: %.3f |validate f1 score: %.3f" % \
                       (epoch, train_mean_loss, train_epoch_acc, train_epoch_recall, train_epoch_f1,
                        validate_mean_loss, validate_epoch_acc, validate_epoch_recall, validate_epoch_f1)
            log_file.write(log_text + "\n")

            # 保存历史信息到列表，绘图使用
            train_loss_history.append(train_mean_loss)
            train_acc_history.append(train_epoch_acc)
            train_recall_history.append(train_epoch_recall)
            train_f1_history.append(train_epoch_f1)
            validate_loss_history.append(validate_mean_loss)
            validate_acc_history.append(validate_epoch_acc)
            validate_recall_history.append(validate_epoch_recall)
            validate_f1_history.append(validate_epoch_f1)
        else:
            log_file.write("validate_total_prediction_length is 0" + "\n")
        # 临时保存（每个epoch）
        if (epoch != epochs): torch.save(model.state_dict(), model_saved_path)

    # 保存最终模型
    torch.save(model.state_dict(), 'model/bilstm_crf_state_dict_final.pt')
    # loss绘图
    save_train_history_image(train_loss_history, validate_loss_history, train_history_image_path, "Loss")
    # 准确率绘图
    save_train_history_image(train_acc_history, validate_acc_history, train_history_image_path, "Acc")
    # 召回绘图
    save_train_history_image(train_recall_history, validate_recall_history, train_history_image_path, "Recall")
    # F1绘图
    save_train_history_image(train_f1_history, validate_f1_history, train_history_image_path, "F1")
    print("train Finished".center(100, "-"))
    # file关闭
    log_file.close()

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
