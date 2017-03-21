import TensorFlowEnvironment
import os as OS
import tensorflow as TensorFlow
import random
import numpy as Numpy

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

ImageSize = 32

def Unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def SetupCifar10TranningResources():
    labels = []
    images = []
    for i in range(1, 6):
        dictionary = Unpickle("../Cifar10/data_batch_" + str(i))
        images.extend(Numpy.reshape(Numpy.array(dictionary['data']), (10000, 32, 32, 3)).astype('float32'))
        for j in range(10000):
            labelVector = Numpy.zeros([10])
            labelVector[dictionary['labels'][j]] = 1
            labels.append(labelVector)
    class CIFAR10Record(object):
        pass
    records = []
    for i in range(50000):
        record = CIFAR10Record()
        record.image = images[i]
        record.label = labels[i]
        records.append(record)
    return Numpy.array(records)

def SetupCifar10TestingResources():
    dictionary = Unpickle("../Cifar10/test_batch")
    testingImageSet = Numpy.reshape(Numpy.array(dictionary['data']), (10000, 32, 32, 3)).astype('float32')
    labels = []
    for j in range(10000):
        labelVector = Numpy.zeros([10])
        labelVector[dictionary['labels'][j]] = 1
        labels.append(labelVector)
    testingLabelSet = Numpy.array(labels)
    return testingImageSet, testingLabelSet

def TakeRandomTranningSampleBatch(tranningSet, batchSize):
    batch = Numpy.random.choice(tranningSet, batchSize, replace=False)
    padding = [[4, 4], [4, 4], [0, 0]]
    images = []
    for j in range(batchSize):
        #image = TensorFlow.pad(batch[j].image, padding)
        #image = TensorFlow.random_crop(image, [32, 32, 3])
        #image = TensorFlow.image.random_flip_left_right(image)
        #image = TensorFlow.image.per_image_standardization(image)
        #image = image.eval()
        #images.append(image)
        images.append(batch[j].image)
    return Numpy.array(images), Numpy.array([d.label for d in batch])
