import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from os import environ
import subprocess

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

rows, cols = 28, 28

x_train = x_train.reshape(x_train.shape[0],rows, cols,1) 
x_test = x_test.reshape(x_test.shape[0],rows, cols,1)

input_shape = (28,28,1)

#normalization
x_train = x_train.astype('float32')
x_train = x_train / 255.0
x_test = x_test.astype('float32')
x_test = x_test / 255.0

#one-hot encoding labels(i.e bool vector labeling)
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

def build_lenet(input_shape):
    #sequentail API
    model = tf.keras.Sequential()
    
    # convo layer #1
    model.add(tf.keras.layers.Conv2D(filters=6,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     activation='tanh',
                                     input_shape=input_shape))
    # sub-sampling #2
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                               strides=(2, 2)))
    # convo layer #3
    model.add(tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     activation='tanh'))
    # sub-sampling #4
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                               strides=(2, 2)))
    # flatten layer #5
    model.add(tf.keras.layers.Flatten())
    
    # 1st fully connected layer #6
    model.add(tf.keras.layers.Dense(units=120,activation='tanh'))
    
    # flatten layer #7
    model.add(tf.keras.layers.Flatten())
    
    # 2nd fully connected layer #8
    model.add(tf.keras.layers.Dense(units=84,activation='tanh'))
    
    # output fully connected layer #9
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, weight_decay=0.0),
                  metrics=['accuracy'])
    
    return model

print('\n')

lenet = build_lenet(input_shape)
lenet.summary()
print('\n')

dirWeights = os.path.dirname(os.path.abspath(__file__))+ "\\checkpoints"
pathWeights = dirWeights + "\\my_checkpoint"


if( os.path.isdir(dirWeights) ):                            # Loading weights or Training then saving weights
    print("Saved weights found and loaded:")
    lenet.load_weights(pathWeights).expect_partial()
    loss, acc = lenet.evaluate(x_test, y_test, verbose=2)
    print("\nLoaded model, accuracy: {:5.2f}%".format(100 * acc))
else:
    epoch = 10
    history = lenet.fit(x_train, y_train,
                        epochs=epoch,
                        batch_size=128,
                        verbose=1,
                        )
    loss, acc = lenet.evaluate(x_test, y_test, verbose=2)
    print("\nTrained model, accuracy: {:5.2f}%".format(100 * acc))
    lenet.save_weights(pathWeights)
    print('Weights saved in: ',pathWeights)

# Dataset info (Training & Testing)
print('\n----------------------------------------')
x_train=x_train.reshape(x_train.shape[0],28,28)
print("Training data", x_train.shape, y_train.shape)

x_test=x_test.reshape(x_test.shape[0],28,28)
print("Testing data", x_test.shape, y_test.shape)
print('----------------------------------------\n')

# Random sample from test data
image_index = random.randint(1,x_test.shape[0]-1)
print("Displaying a random entry from the test dataset with it's prediction( id=",image_index,"):")
plt.imshow(x_test[image_index].reshape(28,28), cmap='Greys')
plt.show()
pred = lenet.predict(x_test[image_index].reshape(1, rows, cols, 1),verbose=0)
print("\t>>>>>>>>>>\tPredicted value: ",pred.argmax(),"\t<<<<<<<<<<\n")

# User inputs a drawing
drawingPathInit = os.path.dirname(os.path.abspath(__file__))+ "\\28by28.jpg"
while(1):
    try:
        drawingPath = input("\nPlease enter the path to a 28 by 28 image with white background and black drawn number (from 0 to 9):")
        if(drawingPath == ""):
            drawingPath=drawingPathInit
            print("Default used: ",drawingPathInit)
        img = tf.keras.preprocessing.image.load_img(drawingPath)
        img = tf.image.rgb_to_grayscale(img)
        image_array = tf.keras.preprocessing.image.img_to_array(img)
        #print(image_array.shape)
        #print(image_array)
        image_array = image_array.astype('float32')
        image_array = 1 - (image_array / 255.0)
        #plt.imshow(image_array.reshape(28,28), cmap='Greys')
        #plt.show()
        pred = lenet.predict(image_array.reshape(1, rows, cols, 1),verbose=0)
        print("\t>>>>>>>>>>\tPredicted value: ",pred.argmax(),"\t<<<<<<<<<<\n")
    except:
        print("Error")


    