import tensorflow as TensorFlow

def CreateWeightVariable(shape):
    initial = TensorFlow.random_normal(shape, stddev = 0.05, dtype=TensorFlow.float32)
    return TensorFlow.Variable(initial)

def CreateBiasVariable(shape):
    initial = TensorFlow.constant(0, shape=shape,dtype=TensorFlow.float32)
    return TensorFlow.Variable(initial)

def Convolution2D(x, W, stride, isSamePadding):
    if isSamePadding:
        return TensorFlow.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    else:
        return TensorFlow.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def MaxPool3x3(x):
    return TensorFlow.nn.max_pool(x, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = "SAME")

def AveragePool8x8(x):
    return TensorFlow.nn.max_pool(x, ksize = [1, 8, 8, 1], strides = [1, 1, 1, 1], padding = "VALID")

def CreateNetwork(inputs, keepProbe):
    # network weights
    convolution1Weights = CreateWeightVariable([5, 5, 3, 192])
    convolution1Bias = CreateBiasVariable([192])
    mlp1_1Weights = CreateWeightVariable([1, 1, 192, 160])
    mlp1_1Bias = CreateBiasVariable([160])
    mlp1_2Weights = CreateWeightVariable([1, 1, 160, 96])
    mlp1_2Bias = CreateBiasVariable([96])

    convolution2Weights = CreateWeightVariable([5, 5, 96, 192])
    convolution2Bias = CreateBiasVariable([192])
    mlp2_1Weights = CreateWeightVariable([1, 1, 192, 192])
    mlp2_1Bias = CreateBiasVariable([192])
    mlp2_2Weights = CreateWeightVariable([1, 1, 192, 192])
    mlp2_2Bias = CreateBiasVariable([192])

    convolution3Weights = CreateWeightVariable([3, 3, 192, 192])
    convolution3Bias = CreateBiasVariable([192])
    mlp3_1Weights = CreateWeightVariable([1, 1, 192, 192])
    mlp3_1Bias = CreateBiasVariable([192])
    mlp3_2Weights = CreateWeightVariable([1, 1, 192, 10])
    mlp3_2Bias = CreateBiasVariable([10])

    # hidden layers convolution1
    convolution1 = TensorFlow.nn.relu(Convolution2D(inputs, convolution1Weights, 1, True) + convolution1Bias)
    mlp1_1 = TensorFlow.nn.relu(Convolution2D(convolution1, mlp1_1Weights, 1, False) + mlp1_1Bias)
    mlp1_2 = TensorFlow.nn.relu(Convolution2D(mlp1_1, mlp1_2Weights, 1, False) + mlp1_2Bias)
    convolution1Pool = TensorFlow.nn.dropout(MaxPool3x3(mlp1_2), keepProbe)
    # hidden layers convolution2
    convolution2 = TensorFlow.nn.relu(Convolution2D(convolution1Pool, convolution2Weights, 1, True) + convolution2Bias)
    mlp2_1 = TensorFlow.nn.relu(Convolution2D(convolution2, mlp2_1Weights, 1, False) + mlp2_1Bias)
    mlp2_2 = TensorFlow.nn.relu(Convolution2D(mlp2_1, mlp2_2Weights, 1, False) + mlp2_2Bias)
    convolution2Pool = TensorFlow.nn.dropout(MaxPool3x3(mlp2_2), keepProbe)
    # hidden layers convolution3
    convolution3 = TensorFlow.nn.relu(Convolution2D(convolution2Pool, convolution3Weights, 1, True) + convolution3Bias)
    mlp3_1 = TensorFlow.nn.relu(Convolution2D(convolution3, mlp3_1Weights, 1, False) + mlp3_1Bias)
    mlp3_2 = TensorFlow.nn.relu(Convolution2D(mlp3_1, mlp3_2Weights, 1, False) + mlp3_2Bias)
    convolution3Pool = AveragePool8x8(mlp3_2)

    return TensorFlow.reshape(convolution3Pool, [-1, 10])