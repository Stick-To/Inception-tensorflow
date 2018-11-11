from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np



from keras.preprocessing.image import ImageDataGenerator
import sys

data_shape = (75, 75, 3)
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
reduce_lr_epoch = [i for i in range(2, epochs , 2)]

train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()

# import Inceptionv3 as net
# testnet = net.Inceptionv3(data_shape, num_classes, l2_rate, keep_prob, 'channels_last')
# import Inceptionv4 as net
# testnet = net.Inceptionv4(data_shape, num_classes, l2_rate, keep_prob, 'channels_last')
import InceptionResnetv2 as net
testnet = net.InceptionResnetv2(data_shape, num_classes, l2_rate, keep_prob, 'channels_last')

for epoch in range(epochs):
    print('-'*20, 'epoch', epoch, '-'*20)
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    # reduce learning rate
    if epoch in reduce_lr_epoch:
        lr = lr * 0.94
        print('reduce learning lr =', lr, 'now' )

    # train one epoch
    for iter in range(num_train//train_batch_size):
        # get and preprocess image
        images,labels = train_gen.next()
        images = images / 127.5 - 1
        # train_one_batch also can accept your own session
        loss, acc = testnet.train_one_batch(images,labels,lr)
        train_acc.append(acc)
        train_loss.append(loss)
        sys.stdout.write("\r>> train "+str(iter+1)+'/'+str(num_train//train_batch_size)+' loss '+str(loss)+' acc '+str(acc))
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    sys.stdout.write("\n")
    print('>> epoch', epoch, 'train mean loss', mean_train_acc, 'train mean acc', mean_train_acc)

    # validate one epoch
    for iter in range(num_test//test_batch_size):
        # get and preprocess image
        images,labels = test_gen.next()
        images = images / 127.5 - 1
        # validate_one_batch also can accept your own session
        loss, acc = testnet.validate_one_batch(images,labels)
        test_acc.append(acc)
        test_loss.append(loss)
        sys.stdout.write("\r>> test "+str(iter+1)+'/'+str(num_test//test_batch_size)+' loss '+str(loss)+' acc '+str(test_acc))
    mean_val_loss = np.mean(test_loss)
    mean_val_acc = np.mean(test_acc)
    sys.stdout.write("\n")
    print('>> epoch', epoch, 'test mean loss',mean_val_acc,' test mean acc', mean_val_acc)

    # logit = testnet.test(images)
    # testnet.save_weight(self, mode, path, sess=None)
    # testnet.load_weight(self, mode, path, sess=None)


