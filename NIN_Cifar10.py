import Cifar10Manager
import TensorFlowEnvironment
import tensorflow as TensorFlow
import numpy as Numpy

def main(argv = None):
    tranningSet = Cifar10Manager.SetupCifar10TranningResources()
    testingImageSet, testingLabelSet = Cifar10Manager.SetupCifar10TestingResources()
    print('Setup Cifar10Manager Successful')
    x = TensorFlow.placeholder(TensorFlow.float32, [None, 32, 32, 3])
    output = TensorFlow.placeholder(TensorFlow.float32, [None, 10])
    keepProbe = TensorFlow.placeholder(TensorFlow.float32)
    imageInput = TensorFlow.reshape(x, [-1, 32, 32, 3])
    
    network = TensorFlowEnvironment.CreateNetwork(imageInput, keepProbe)

    learningRate = TensorFlow.placeholder(TensorFlow.float32)

    crossEntropy = TensorFlow.reduce_mean(TensorFlow.nn.softmax_cross_entropy_with_logits(labels=output, logits=network))

    trainStep = TensorFlow.train.MomentumOptimizer(learningRate, 0.9).minimize(crossEntropy)

    correct_prediction = TensorFlow.equal(TensorFlow.argmax(network,1), TensorFlow.argmax(output,1))
    accuracy = TensorFlow.reduce_mean(TensorFlow.cast(correct_prediction, TensorFlow.float32))

    session = TensorFlow.InteractiveSession()
    session.run(TensorFlow.global_variables_initializer())

    def Testing(sess):
        acc = 0.0
        for i in range(1,11):  
            image = Numpy.array(testingImageSet[0+(i-1)*1000:1000*i])
            label = Numpy.array(testingLabelSet[0+(i-1)*1000:1000*i])
            acc= acc + accuracy.eval(session=sess, feed_dict={x: image, output: label, keepProbe: 1.0})
        return acc/10

    for i in range(37520): #1~80 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every 100 step
        if i%100 == 0:
            print("step %d, Test accuracy %g"%(i,  Testing(session)))
        ## trianing
        # if learningRate is 0.1  first ablot 10 step loss will go large and to nan, and network can't learning anything
        [_, loss] = session.run([trainStep, crossEntropy],feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.001})  #learning rate 0.1
        print(loss)

    for i in range(18760): #81~121 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every 100 step
        if i%100 == 0:
            print("step %d, Test accuracy %g"%(i,  Testing(session)))
        ## trianing
        trainStep.run(feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.01})  #learning rate 0.01
    
    for i in range(18760): #122~164 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every 100 step
        if i%100 == 0:
            print("step %d, Test accuracy %g"%(i,  Testing(session)))
        ## trianing
        trainStep.run(feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.001})  #learning rate 0.001

    ##final testing
    print("step %d, Test accuracy %g"%(i, Testing(session)))

if __name__ == '__main__':
    TensorFlow.app.run()