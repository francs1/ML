# trainingModel.py

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from constantInit import *
import dataLoad as dl


import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

def train(mnist):    
    trainX, trainY = mnist.train.images, mnist.train.labels
    validateX, validateY = mnist.validation.images, mnist.validation.labels
    testX,testY = mnist.test.images,mnist.test.labels

    # 根据对图像编码的格式要求来设置输入层的格式。
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        validateX = validateX.reshape(validateX.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:# 'channels_last'
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        validateX = validateX.reshape(validateX.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    # 使用Keras API定义模型。
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
 
    # 定义损失函数、优化函数和评测方法。
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


    model.fit(trainX, trainY,
          batch_size=BATCH_SIZE,
          epochs=TRAINING_STEPS,
          validation_data=(validateX, validateY))

    # 在验证集数据上计算准确率。
    score = model.evaluate(validateX, validateY)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result = model.predict(testX, batch_size=BATCH_SIZE)
    dl.saveData(result)