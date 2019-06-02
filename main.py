#/usr/bin/python2.7
#coding:utf-8

import tensorflow as tf
import csv
import numpy as np
import random
global feature
global label
def addLayer(inputs, in_size, out_size, activationFunction=None,layerName='layer'):
    # 添加一层神经网络，返回输出值
    with tf.name_scope(layerName):          #可视化图层分级
        with tf.name_scope(layerName+'_Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        tf.histogram_summary(layerName+'_Weights', Weights)
        with tf.name_scope(layerName+'_biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        tf.histogram_summary(layerName+'_biases', biases)
        with tf.name_scope(layerName+'_Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activationFunction is None:
            outputs = Wx_plus_b
        else:
            outputs = activationFunction(Wx_plus_b)
        tf.histogram_summary(layerName + '_outputs', outputs)
        return outputs
def next_batch(feature_list,label_list,size):
    # 随机梯度下降
    feature_batch_temp=[]
    label_batch_temp=[]
    f_list = random.sample(range(len(feature_list)), size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp
def main():
    global feature
    global label
    # load数据
    load_data()
    feature_train=feature[0:490000]
    feature_test=feature[490001:]
    label_train=label[0:490000]
    label_test=label[490001:]
    # 创建placeholder
    with tf.name_scope('inputs'):
        picData = tf.placeholder(tf.float32, [None, 41], name='picData')
        picLabel = tf.placeholder(tf.float32, [None, 2], name='picLabel')
    # 画神经网络图
    picPrediction = addLayer(picData, 41, 2, activationFunction=tf.nn.softmax, layerName='L1')
    # 计算loss/cost
    with tf.name_scope('loss'):
        loss = -tf.reduce_sum(picLabel * tf.log(picPrediction), name='loss')
    tf.scalar_summary('loss', loss)
    # 创建优化器并设置学习效率
    with tf.name_scope('train'):
        trainStep = tf.train.GradientDescentOptimizer(0.5, name='trainStep').minimize(loss)
    # 评估测试
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(picPrediction, 1), tf.argmax(picLabel, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary('accuracy', accuracy)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # 将神经网络结构画出来
        writer = tf.train.SummaryWriter("logs/", sess.graph)
        sess.run(init)
        # 分别分出训练和评估时标记值的变化
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter("logs/train", sess.graph)
        test_writer = tf.train.SummaryWriter("logs/test", sess.graph)
        # 进行训练1001次
        for step in range(1001):
            feature_train_batch,label_train_batch=next_batch(feature_train,label_train, 1000)# 随机梯度下降训练，每次选大小为1000的batch
            # feature_test_batch=next_batch(feature_test, 100)# 随机梯度下降训练，每次选大小为1000的batch
            # label_test_batch=next_batch(label_test, 100)

            sess.run(trainStep, feed_dict={picData: feature_train_batch, picLabel: label_train_batch})
            if step % 50 == 0:
                # train_writer.add_summary(sess.run(merged, feed_dict={picData: feature_train_batch, picLabel: label_train_batch}), step)
                # test_writer.add_summary(
                #     sess.run(merged, feed_dict={picData: feature_test_batch, picLabel: label_test_batch}), step)
                print(step, sess.run(accuracy, feed_dict={picData: feature_test, picLabel: label_test}),
                        sess.run(accuracy, feed_dict={picData: feature_train_batch, picLabel: label_train_batch}))
def load_data():
    global feature
    global label
    feature=[]
    label=[]
    file_path ='/home/peter/Desktop/pycharm/ids-kdd99/kddcup.data_10_percent_corrected_handled2.cvs'
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            # print i
            feature.append(i[:41])
            label_list=[0 for i in range(23)]
            label_list[i[41]]=1
            label.append(label_list)
            # print label
            # print feature
            # return 0
if __name__  == '__main__':
    global feature
    global label
    load_data()
