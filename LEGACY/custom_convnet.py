# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
from common.util import *

def he_stdev(node_num):
    return np.sqrt(2)/np.sqrt(node_num)

class ConvNet:
    """ConvNet class

    < Model structure >
    conv - relu - pool -
    conv - relu - pool -
    conv - relu - pool -
    affine - softmax

    Parameters
    ----------
    input_size : 784 (like MNIST）
    hidden_size_list : number of hidden neurans（e.g. [100, 100, 100]）
    output_size : classes
    activation : 'relu' or 'sigmoid'
    weight_init_std : specifies the standard deviation of the weight (eg 0.01)
    """
    def __init__(self, input_dim=(3, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 pool_size=2,
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        print("\n***** Network *****")
        print(" Input  : "+ str(input_dim))
        print(" Output : (1, "+ str(output_size)+")\n")

        # extract patameter from conv_param dictionary
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad'] # padding
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # output_height = ((input_size - filter_size + 2*filter_pad) / filter_stride) + 1
        # deep learning from scratch p.234
        # Layer1
        layer1_filter_num = filter_num*1
        conv1_output_size = int(conv_out_size(input_size=input_size, filter_size=filter_size, filter_pad=filter_pad, filter_stride=filter_stride))
        pool1_output_size = int(pool_out_size(conv_output_size=conv1_output_size, filter_pad=filter_pad, pool_size=pool_size))
        # Layer2
        layer2_filter_num = filter_num*2
        conv2_output_size = int(conv_out_size(input_size=pool1_output_size, filter_size=filter_size, filter_pad=filter_pad, filter_stride=filter_stride))
        pool2_output_size = int(pool_out_size(conv_output_size=conv2_output_size, filter_pad=filter_pad, pool_size=pool_size))
        # Layer3
        layer3_filter_num = filter_num*1
        conv3_output_size = int(conv_out_size(input_size=pool2_output_size, filter_size=filter_size, filter_pad=filter_pad, filter_stride=filter_stride))
        pool3_output_size = int(pool_out_size(conv_output_size=conv3_output_size, filter_pad=filter_pad, pool_size=pool_size))
        # Laye4
        layer4_input_size = int(layer3_filter_num * pool3_output_size * pool3_output_size)
        # weight initialize with normal distribution
        self.params = {}

        # Layer1
        # layer1_filter_num: number of weight filter (output tensor)
        # input_dim[0]: channel (input tensor)
        # filter_size: weight width and height
        print(" Layer1: conv - relu - pool")
        print(" filter: %d x %d | input_dim: %d | output_dim: %d" %(filter_size, filter_size, input_dim[0], layer1_filter_num))
        self.params['W1'] = np.random.randn(layer1_filter_num, input_dim[0], filter_size, filter_size) * he_stdev(input_dim[0])
        self.params['b1'] = np.zeros(layer1_filter_num)

        # Layer2
        # layer2_filter_num: number of weight filter (output tensor)
        # layer1_filter_num: input tensor
        # filter_size: weight width and height
        print("\n Layer2: conv - relu - pool")
        print(" filter: %d x %d | input_dim: %d | output_dim: %d" %(filter_size, filter_size, layer1_filter_num, layer2_filter_num))
        self.params['W2'] = np.random.randn(layer2_filter_num, layer1_filter_num, filter_size, filter_size) * he_stdev(layer1_filter_num)
        self.params['b2'] = np.zeros(layer2_filter_num)

        # Layer3
        # layer3_filter_num: number of weight filter (output tensor)
        # layer2_filter_num: input tensor
        # filter_size: weight width and height
        print("\n Layer3: conv - relu - pool")
        print(" filter: %d x %d | input_dim: %d | output_dim: %d" %(filter_size, filter_size, layer2_filter_num, layer3_filter_num))
        self.params['W3'] = np.random.randn(layer3_filter_num, layer2_filter_num, filter_size, filter_size) * he_stdev(layer2_filter_num)
        self.params['b3'] = np.zeros(layer3_filter_num)
        # Layer4
        # layer4_input_size: fully connected size (input matrix)
        # output_size: output (output matrix)
        print("\n Layer4: affine")
        print(" input: %d | output: %d" %(layer4_input_size, output_size))
        self.params['W4'] = weight_init_std * np.random.randn(layer4_input_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # Last layer is softmax
        print("\n Last layer: softmax")

        # define layers
        # Layer1
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=pool_size, pool_w=pool_size, stride=2)
        # Layer2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=pool_size, pool_w=pool_size, stride=2)
        # Layer3
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=pool_size, pool_w=pool_size, stride=2)
        # Layer4
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4'])
        # Last Layer
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x): # It used for predict loss and accuracy.
        for layer in self.layers.values():
            #print("Input size: "+str(x.shape)+" | "+str(x.shape))
            x = layer.forward(x)

        return x

    # Compute loss
    # x : input data
    # t : input label
    def loss(self, x, t):

        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # Compute accuracy
    # x : input data
    # t : input label
    # batch_size : default value is 100
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # Find Weight and Bias with Numerical differentiation method
    # x : input data
    # t : input label
    def numerical_gradient(self, x, t):

        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    # Find Weight and Bias with back propagation method
    # x : input data
    # t : input label
    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        # last_layer: Softmax layer with loss
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values()) # return dictionary keywords
        layers.reverse() # list upside down

        # backward method compute each layer's Weight and Bias - dW, db
        for layer in layers:
            dout = layer.backward(dout)

        # set gradients
        grads = {}
        # Layer1
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        # Layer2
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        # Layer3
        grads['W3'] = self.layers['Conv3'].dW
        grads['b3'] = self.layers['Conv3'].db
        # Layer4
        grads['W4'] = self.layers['Affine1'].dW
        grads['b4'] = self.layers['Affine1'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        # make dictionary
        # params is not self.params
        params = {}

        # extract key and values from self.params and initialize dictionary
        # .values : return key only
        # .items(): return key and value
        for key, val in self.params.items():
            params[key] = val

        # save dictionary to file
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        # load dictionary to file
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # extract key and values from params and set to self.params
        # .values : return key only
        # .items(): return key and value
        for key, val in params.items():
            self.params[key] = val

        # initialize layers with Weight and Bias
        # ['Conv1', 'Conv2', 'Affine1'] are not in pkl file
        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Affine1']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
