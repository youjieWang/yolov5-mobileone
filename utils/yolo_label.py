#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/24 14:51
# @Author  : yj.wang
# @File    : yolo_label.py
# @Describe: 该模块包含json标签转换成yolo需要的txt标签；数据集划分功能。
# -*- coding: UTF-8 -*-

import json
import os
import random, shutil
from tqdm import tqdm

# 类别映射，用以映射成数字， 标签中的类型可以是单词
classify_map = {
    '1': 0,
    '2': 1,
    '3': 2,
}


# 转换成yolo格式的数据
def xyxy2xywh(size, box):
    '''
    size:(w, h)
    '''
    dw = 1. / size[0]
    dh = 1. / size[1]
    # x1 + (x2 - x1)/2 = (x1+x2)/2
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)


def json2yolo(json_path, txt_out_path):
    '''
    path: json标签保存路径
    txt_out_path: 转换之后txt保存路径
    '''
    if not os.path.exists(txt_out_path):
        os.mkdir(txt_out_path)

    for file in os.listdir(json_path):
        if not file.endswith('.json'):
            print(file)
            continue
        with open(os.path.join(json_path, file), 'r') as f:
            data = json.load(f)
            txt_name = file.split(".")[0] + '.txt'
            # h, w用以归一化操作
            h = data["imageHeight"]
            w = data["imageWidth"]
            res = []
            # shapes:[{}]
            for item in data['shapes']:
                classify = classify_map[item['label']]  # 获取类别
                points = item['points']
                xmin = min(points[0][0], points[1][0])
                ymin = min(points[0][1], points[1][1])
                xmax = max(points[0][0], points[1][0])
                ymax = max(points[0][1], points[1][1])
                box = [float(xmin), float(ymin), float(xmax), float(ymax)]
                # 将x1, y1, x2, y2 转换成yolov5中的格式x,y, w, h
                bbox = xyxy2xywh((w, h), box)
                res.append([classify, bbox])
            with open(os.path.join(txt_out_path, txt_name), 'w') as out_file:
                for classify, bbox in res:
                    out_file.write(str(classify) + " " + " ".join(str(x) for x in bbox) + '\n')
            out_file.close()
    print(len(os.listdir(txt_out_path)))

    print('transfer yolo success')


# 数据集划分
def split_img(origin_path, Data, split_list):
    '''
    origin_path: 原始数据集的路径包括images， labels(json, txt)
    split_list: 数据集划分的比例
    Data: 文件夹路径
    '''

    try:

        if not os.path.exists(Data):
            os.makedirs(Data)
        train_img_dir = Data + '/images/train'
        val_img_dir = Data + '/images/val'
        test_img_dir = Data + '/images/test'

        train_label_dir = Data + '/labels/train'
        val_label_dir = Data + '/labels/val'
        test_label_dir = Data + '/labels/test'

        os.makedirs(train_img_dir)
        os.makedirs(val_img_dir)
        os.makedirs(test_img_dir)

        os.makedirs(train_label_dir)
        os.makedirs(val_label_dir)
        os.makedirs(test_label_dir)

    except:
        print('目录已经存在')

    train, val, test = split_list
    img_path = os.path.join(origin_path, 'images')
    label_path = os.path.join(origin_path, 'txt_labels')

    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img]
    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    # train_img_copy = [os.path.join(train_img_dir, img.split('\\')[-1]) for img in train_img]
    train_label = [toLabelPath(img, label_path) for img in train_img]
    # train_label_copy = [os.path.join(train_label_dir, label.split('\\')[-1]) for label in train_label]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=80, unit='img'):
        _copy(train_img[i], train_img_dir)
        _copy(train_label[i], train_label_dir)
        all_img_path.remove(train_img[i])
    val_img = random.sample(all_img_path, int(val / (val + test) * len(all_img_path)))
    val_label = [toLabelPath(img, label_path) for img in val_img]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=80, unit='img'):
        _copy(val_img[i], val_img_dir)
        _copy(val_label[i], val_label_dir)
        all_img_path.remove(val_img[i])
    test_img = all_img_path
    test_label = [toLabelPath(img, label_path) for img in test_img]
    for i in tqdm(range(len(test_img)), desc='test ', ncols=80, unit='img'):
        _copy(test_img[i], test_img_dir)
        _copy(test_label[i], test_label_dir)


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)


def toLabelPath(img_path, label_path):
    img = img_path.split('\\')[-1]
    label = img.split('.')[0] + '.txt'
    print(label)
    return os.path.join(label_path, label)


if __name__ == '__main__':
    # --------------------转换成yolo label的代码
    # json_path = r'E:\sw\highlightTime-DOTA2\AllData\yolo_data\json_labels'
    # txt_out_path = r'E:\sw\highlightTime-DOTA2\AllData\yolo_data\txt_labels'
    # json2yolo(json_path, txt_out_path)

    # --------------------数据集划分的代码
    origin_path = r'E:\sw\highlightTime-DOTA2\AllData\yolo_data'
    Data = '../dataset'  # 划分数据集的路径
    split_list = [0.7, 0.2, 0.1]  # 数据集划分比例[train:val:test]
    split_img(origin_path, Data, split_list)
