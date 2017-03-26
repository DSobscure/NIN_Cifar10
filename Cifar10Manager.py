import TensorFlowEnvironment
import os as OS
import tensorflow as TensorFlow
import random
import numpy as Numpy

def Unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    if "data" in dict:
        dict["data"] = dict["data"].reshape((-1,3,32,32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3);
    return dict

def SetupCifar10TranningResources():
    images = []
    labels = []
    class CIFAR10Record(object):
        pass
    records = []
    for i in range(1, 6):
        dictionary = Unpickle("../Cifar10/data_batch_" + str(i))
        for j in range(10000):
            record = CIFAR10Record()
            record.label = Numpy.zeros([10])
            record.label[dictionary['labels'][j]] = 1
            record.image = Numpy.divide(Numpy.subtract(Numpy.reshape(dictionary['data'][j], (32,32,3)), (125.3,123.0,113.9)), (63.0,62.1,66.7))
            records.append(record)
    return Numpy.array(records)

def SetupCifar10TestingResources():
    dictionary = Unpickle("../Cifar10/test_batch")
    images = []
    labels = []
    for j in range(10000):
        labelVector = Numpy.zeros([10])
        labelVector[dictionary['labels'][j]] = 1
        labels.append(labelVector)
        images.append(Numpy.divide(Numpy.subtract(Numpy.reshape(dictionary['data'][j], (32,32,3)), (125.3,123.0,113.9)), (63.0,62.1,66.7)))
    return Numpy.array(images), Numpy.array(labels)

def RandomCrop(batch, crop_shape, padding=None):
    oshape = Numpy.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = Numpy.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch

def RandomFlipLeftRight(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = Numpy.fliplr(batch[i])
    return batch

def TakeRandomTranningSampleBatch(tranningSet, batchSize):
    batch = Numpy.random.choice(tranningSet, batchSize, replace=False)
    images = [element.image for element in batch]
    croppedData = RandomCrop(images, (32, 32, 3), 4)
    flippedData = RandomFlipLeftRight(croppedData) 
    return Numpy.array(flippedData), Numpy.array([element.label for element in batch])
