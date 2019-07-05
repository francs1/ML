import tensorflow as tf
from constantInit import *
import dataLoad as dl
import tensorflow as tf
from tensorflow.keras import layers

def train(mnist):
    model = tf.keras.Sequential()
    model.add(layers.Dense(units=LAYER1_NODE,activation='relu',use_bias=True,input_shape=(INPUT_NODE,),kernel_regularizer=tf.keras.regularizers.l2(REGULARAZTION_RATE)))
    model.add(layers.Dense(units=OUTPUT_NODE,activation='softmax',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(REGULARAZTION_RATE)))
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    data,labels = mnist.train.images,mnist.train.labels
    val_data,val_labels = mnist.validation.images,mnist.validation.labels
    model.fit(data, labels, epochs=TRAINING_STEPS, batch_size=BATCH_SIZE,validation_data=(val_data, val_labels))
    #model.evaluate(val_data, val_labels, batch_size=32)

    test_data,test_labels = mnist.test.images,mnist.test.labels
    result = model.predict(test_data, batch_size=32)
    #print(type(result))
    dl.saveData(result)

# def main(argv=None):
#     #mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
#     mnist = dl.loadData()
#     train(mnist)

# if __name__=='__main__':
#     main()
   