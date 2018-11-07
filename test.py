from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import Inceptionv3 as net
import keras
from keras.preprocessing.image import ImageDataGenerator
import sys
data_shape = (299, 299, 3)
num_train = 50000
num_test = 10000
num_classes = 10
train_batch_size = 32
test_batch_size = 32
epochs = 200
l2_rate = 1e-4
keep_prob = 0.7
lr = 0.045

# epochs reduce learning_rate by 0.94
reduce_lr_epoch = [i for i in range(2, epochs ,2)]

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
train_gen = ImageDataGenerator(horizontal_flip=True,
                               width_shift_range=0.1,
                               height_shift_range=0.1
).flow(x_train, y_train, batch_size=train_batch_size)
test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=test_batch_size)


testnet = net.Inceptionv3(data_shape, num_classes, l2_rate, keep_prob, 'channels_last')

for epoch in range(epochs):
    print('-'*20, 'epoch', epoch, '-'*20)
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # reduce learning rate
    if epoch in reduce_lr_epoch:
        lr = lr * 0.94
        print('reduce learning lr =', lr, 'now' )

    # train one epoch
    for iter in range(num_train//train_batch_size):
        images,labels = train_gen.next()
        # train_one_batch also can accept your own session
        loss, acc = testnet.train_one_batch(images,labels,lr)
        train_acc.append(acc)
        train_loss.append(loss)
        sys.stdout.write("\r>> train "+str(iter+1)+'/'+str(num_train//train_batch_size)+' loss '+str(loss)+' acc '+str(acc))
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    sys.stdout.write("\n")
    print('>> epoch',epoch,'train mean loss',mean_train_acc,'train mean acc',mean_train_acc)

    # validate one epoch
    for iter in range(num_test//test_batch_size):
        images,labels = test_gen.next()
        # validate_one_batch also can accept your own session
        loss, acc = testnet.validate_one_batch(images,labels)
        val_acc.append(acc)
        val_loss.append(loss)
        sys.stdout.write("\r>> val "+str(iter+1)+'/'+str(num_test//test_batch_size)+' loss '+str(loss)+' acc '+str(acc))
    mean_val_loss = np.mean(val_loss)
    mean_val_acc = np.mean(val_acc)
    sys.stdout.write("\n")
    print('>> epoch',epoch,'val mean loss',mean_val_acc,'val mean acc',mean_val_acc)

    # logit = testnet.test(images)
    # testnet.save_weight(self, mode, path, sess=None)
    # testnet.load_weight(self, mode, path, sess=None)


