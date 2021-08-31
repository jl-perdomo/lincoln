#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:29:28 2021

@author: joanaperdomo
"""

import numpy as np
from numpy import ndarray

# Helpers
def assert_same_shape(output: ndarray, 
                      output_grad: ndarray):
    assert output.shape == output_grad.shape, \
    '''
    Two ndarray should have the same shape; instead, first ndarray's shape is {0}
    and second ndarray's shape is {1}.
    '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None

def assert_dim(t: ndarray,
               dim: ndarray):
    assert len(t.shape) == dim, \
    '''
    Tensor expected to have dimension {0}, instead has dimension {1}
    '''.format(dim, len(t.shape))
    return None

#1D Convolution (1 input 1 output)
## Padding
input_1d = np.array([1,2,3,4,5])
param_1d = np.array([1,1,1])
'''
Padding is the term used in the book to show the values before and after the 
middle term (all are gradients) of the partial derivative of the Loss wrt to 
the element of interest in the input.
'''

def _pad_1d(inp: ndarray,
            num: int) -> ndarray:
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])

_pad_1d(input_1d, 1) # Out[38]: array([0, 1, 2, 3, 4, 5, 0])

## Forward
'''
Now that we've computed the partial (padding being the last part we were missing),
we can now fully implement the forward pass of the 1D convolution.
'''
def conv_1d(inp: ndarray, 
            param: ndarray) -> ndarray:
    
    # assert correct dimensions
    assert_dim(inp, 1)
    assert_dim(param, 1)
    
    # pad the input
    param_len = param.shape[0]
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)
    
    # initialize the output
    out = np.zeros(inp.shape)
    
    # perform the 1d convolution
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * inp_pad[o+p]

    # ensure shapes didn't change            
    assert_same_shape(inp, out)

    return out

def conv_1d_sum(inp: ndarray, 
                param: ndarray) -> ndarray:
    out = conv_1d(inp, param)
    return np.sum(out)

conv_1d_sum(input_1d, param_1d) # Out[39]: 39.0

# Testing gradients
np.random.seed(190220)
print(np.random.randint(0, input_1d.shape[0])) #4
print(np.random.randint(0, param_1d.shape[0])) #0

input_1d_2 = np.array([1,2,3,4,6])
param_1d = np.array([1,1,1])

print(conv_1d_sum(input_1d_2, param_1d) - conv_1d_sum(input_1d_2, param_1d)) #41-41=0.0

input_1d = np.array([1,2,3,4,5])
param_1d_2 = np.array([2,1,1])

print(conv_1d_sum(input_1d, param_1d_2) - conv_1d_sum(input_1d, param_1d)) #49 - 39 = 10

# Gradients
def _param_grad_1d(inp: ndarray, 
                   param: ndarray, 
                   output_grad: ndarray = None) -> ndarray:
    
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)
    
    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            param_grad[p] += input_pad[o+p] * output_grad[o]
        
    assert_same_shape(param_grad, param)
    
    return param_grad

def _input_grad_1d(inp: ndarray, 
                   param: ndarray, 
                   output_grad: ndarray = None) -> ndarray:
    
    param_len = param.shape[0]
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)
    
    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)
    
    output_pad = _pad_1d(output_grad, param_mid)
    
    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for f in range(param.shape[0]):
            input_grad[o] += output_pad[o+param_len-f-1] * param[f]
        
    assert_same_shape(param_grad, param)
    
    return input_grad

_input_grad_1d(input_1d, param_1d) #Out[18]: array([2, 3, 3, 3, 2])

_param_grad_1d(input_1d, param_1d) #Out[19]: array([10, 15, 14])

# Batch Size of 2
## Pad
input_1d_batch = np.array([[0,1,2,3,4,5,6], 
                           [1,2,3,4,5,6,7]])

def _pad_1d(inp: ndarray,
            num: int) -> ndarray:
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])

def _pad_1d_batch(inp: ndarray, 
                  num: int) -> ndarray:
    outs = [_pad_1d(obs, num) for obs in inp]
    return np.stack(outs)

_pad_1d_batch(input_1d_batch, 1)
#Out[20]: 
#array([[0, 0, 1, 2, 3, 4, 5, 6, 0],
#       [0, 1, 2, 3, 4, 5, 6, 7, 0]])

## Forward
def conv_1d_batch(inp: ndarray, 
                  param: ndarray) -> ndarray:

    outs = [conv_1d(obs, param) for obs in inp]
    return np.stack(outs)

conv_1d_batch(input_1d_batch, param_1d) 
#array([[ 1.,  3.,  6.,  9., 12., 15., 11.],
#       [ 3.,  6.,  9., 12., 15., 18., 13.]])

## Gradient
def input_grad_1d_batch(inp: ndarray, 
                        param: ndarray) -> ndarray:

    out = conv_1d_batch(inp, param)
    
    out_grad = np.ones_like(out)
    
    batch_size = out_grad.shape[0]
        
    grads = [_input_grad_1d(inp[i], param, out_grad[i]) for i in range(batch_size)]    

    return np.stack(grads)

def param_grad_1d_batch(inp: ndarray, 
                        param: ndarray) -> ndarray:

    output_grad = np.ones_like(inp)
    
    inp_pad = _pad_1d_batch(inp, 1)
    out_pad = _pad_1d_batch(inp, 1)

    param_grad = np.zeros_like(param)    
    
    for i in range(inp.shape[0]):
        for o in range(inp.shape[1]):
            for p in range(param.shape[0]):
                param_grad[p] += inp_pad[i][o+p] * output_grad[i][o]    

    return param_grad

# Checking gradients for conv_1d_batch
def conv_1d_batch_sum(inp: ndarray, 
                      fil: ndarray) -> ndarray:
    out = conv_1d_batch(inp, fil)
    return np.sum(out)

conv_1d_batch_sum(input_1d_batch, param_1d) #Out[23]: 133.0

print(np.random.randint(0, input_1d_batch.shape[0])) #0
print(np.random.randint(0, input_1d_batch.shape[1])) #2

input_1d_batch_2 = input_1d_batch.copy()
input_1d_batch_2[0][2] += 1
conv_1d_batch_sum(input_1d_batch_2, param_1d) - conv_1d_batch_sum(input_1d_batch, param_1d) #3.0

input_grad_1d_batch(input_1d_batch, param_1d)
#Out[28]: 
#array([[2, 3, 3, 3, 3, 3, 2],
#       [2, 3, 3, 3, 3, 3, 2]])

print(np.random.randint(0, param_1d.shape[0])) #2


param_1d_2 = param_1d.copy()
param_1d_2[2] += 1
conv_1d_batch_sum(input_1d_batch, param_1d_2) - conv_1d_batch_sum(input_1d_batch, param_1d) #48

param_grad_1d_batch(input_1d_batch, param_1d) #Out[31]: array([36, 49, 48])

# 2D Convolutions
imgs_2d_batch = np.random.randn(3, 28, 28)

param_2d = np.random.randn(3, 3)

## Padding
def _pad_2d(inp: ndarray, 
            num: int):
    '''
    Input is a 3 dimensional tensor, first dimension batch size
    '''
    outs = [_pad_2d_obs(obs, num) for obs in inp]

    return np.stack(outs)

def _pad_2d_obs(inp: ndarray, 
                num: int):
    '''
    Input is a 2 dimensional, square, 2D Tensor
    '''
    inp_pad = _pad_1d_batch(inp, num)

    other = np.zeros((num, inp.shape[0] + num * 2))

    return np.concatenate([other, inp_pad, other])

_pad_2d(imgs_2d_batch, 1).shape #Out[33]: (3, 30, 30)

## Compute output
def _compute_output_obs_2d(obs: ndarray, 
                           param: ndarray):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    param_mid = param.shape[0] // 2
    
    obs_pad = _pad_2d_obs(obs, param_mid)
    
    out = np.zeros_like(obs)
    
    for o_w in range(out.shape[0]):
        for o_h in range(out.shape[1]):
            for p_w in range(param.shape[0]):
                for p_h in range(param.shape[1]):
                    out[o_w][o_h] += param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
    return out

def _compute_output_2d(img_batch: ndarray,
                       param: ndarray):
    
    assert_dim(img_batch, 3)
    
    outs = [_compute_output_obs_2d(obs, param) for obs in img_batch]
    
    return np.stack(outs)

_compute_output_2d(imgs_2d_batch, param_2d).shape #Out[34]: (3, 28, 28)

## Param grads
def _compute_grads_obs_2d(input_obs: ndarray,
                          output_grad_obs: ndarray, 
                          param: ndarray) -> ndarray:
    '''
    input_obs: 2D Tensor representing the input observation
    output_grad_obs: 2D Tensor representing the output gradient  
    param: 2D filter
    '''
    
    param_size = param.shape[0]
    output_obs_pad = _pad_2d_obs(output_grad_obs, param_size // 2)
    input_grad = np.zeros_like(input_obs)

    for i_w in range(input_obs.shape[0]):
        for i_h in range(input_obs.shape[1]):
            for p_w in range(param_size):
                for p_h in range(param_size):
                    input_grad[i_w][i_h] += output_obs_pad[i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                    * param[p_w][p_h]

    return input_grad

def _compute_grads_2d(inp: ndarray,
                      output_grad: ndarray, 
                      param: ndarray) -> ndarray:

    grads = [_compute_grads_obs_2d(inp[i], output_grad[i], param) for i in range(output_grad.shape[0])]    

    return np.stack(grads)


def _param_grad_2d(inp: ndarray,
                   output_grad: ndarray, 
                   param: ndarray) -> ndarray:

    param_size = param.shape[0]
    inp_pad = _pad_2d(inp, param_size // 2)

    param_grad = np.zeros_like(param)
    img_shape = output_grad.shape[1:]
    
    for i in range(inp.shape[0]):
        for o_w in range(img_shape[0]):
            for o_h in range(img_shape[1]):
                for p_w in range(param_size):
                    for p_h in range(param_size):
                        param_grad[p_w][p_h] += inp_pad[i][o_w+p_w][o_h+p_h] \
                        * output_grad[i][o_w][o_h]
    return param_grad

img_grads = _compute_grads_2d(imgs_2d_batch, 
                              np.ones_like(imgs_2d_batch),
                              param_2d)

img_grads.shape #Out[36]: (3, 28, 28)

param_grad = _param_grad_2d(imgs_2d_batch, 
                              np.ones_like(imgs_2d_batch),
                              param_2d)
param_grad.shape #Out[36]: (3, 28, 28)

# Testing gradients
## Input
print(np.random.randint(0, imgs_2d_batch.shape[0])) #0
print(np.random.randint(0, imgs_2d_batch.shape[1])) #6
print(np.random.randint(0, imgs_2d_batch.shape[2])) #18

imgs_2d_batch_2 = imgs_2d_batch.copy()
imgs_2d_batch_2[0][6][18] += 1

def _compute_output_2d_sum(img_batch: ndarray,
                           param: ndarray):
    
    out = _compute_output_2d(img_batch, param)
    
    return out.sum()

_compute_output_2d_sum(imgs_2d_batch_2, param_2d) #-38.211192248160906
_compute_output_2d_sum(imgs_2d_batch, param_2d) #-35.02684450830099

img_grads[0][6][18] #-3.184347739859924

## Param
print(np.random.randint(0, param_2d.shape[0])) #2
print(np.random.randint(0, param_2d.shape[1])) #2

param_2d_2 = param_2d.copy()
param_2d_2[0][2] += 1

_compute_output_2d_sum(imgs_2d_batch, param_2d_2) - _compute_output_2d_sum(imgs_2d_batch, param_2d) #5.53349015923007

param_grad[0][2] #5.533490159230001

# With channels + batch size
## Helper
def _pad_2d_channel(inp: ndarray, 
                    num: int):
    '''
    inp has dimension [num_channels, image_width, image_height] 
    '''
    return np.stack([_pad_2d_obs(channel, num) for channel in inp])

def _pad_conv_input(inp: ndarray,
                    num: int):   
    '''
    inp has dimension [batch_size, num_channels, image_width, image_height]
    '''    
    return np.stack([_pad_2d_channel(obs, num) for obs in inp])

## Forward
def _compute_output_obs(obs: ndarray, 
                        param: ndarray):
    '''
    obs: [channels, img_width, img_height]
    param: [in_channels, out_channels, fil_width, fil_height]    
    '''
    assert_dim(obs, 3)
    assert_dim(param, 4)
    
    param_size = param.shape[2]
    param_mid = param_size // 2
    obs_pad = _pad_2d_channel(obs, param_mid)
    
    in_channels = param.shape[0]
    out_channels = param.shape[1]
    img_size = obs.shape[1]
    
    out = np.zeros((out_channels,) + obs.shape[1:])
    for c_in in range(in_channels):
        for c_out in range(out_channels):
            for o_w in range(img_size):
                for o_h in range(img_size):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            out[c_out][o_w][o_h] += \
                            param[c_in][c_out][p_w][p_h] * obs_pad[c_in][o_w+p_w][o_h+p_h]
    return out    

def _output(inp: ndarray,
                    param: ndarray) -> ndarray:
    '''
    obs: [batch_size, channels, img_width, img_height]
    fil: [in_channels, out_channels, fil_width, fil_height]    
    '''
    outs = [_compute_output_obs(obs, param) for obs in inp]    

    return np.stack(outs)

## Backward
def _compute_grads_obs(input_obs: ndarray,
                       output_grad_obs: ndarray,
                       param: ndarray) -> ndarray:
    '''
    input_obs: [in_channels, img_width, img_height]
    output_grad_obs: [out_channels, img_width, img_height]
    param: [in_channels, out_channels, img_width, img_height]    
    '''
    input_grad = np.zeros_like(input_obs)    
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = input_obs.shape[1]
    in_channels = input_obs.shape[0]
    out_channels = param.shape[1]
    output_obs_pad = _pad_2d_channel(output_grad_obs, param_mid)
    
    for c_in in range(in_channels):
        for c_out in range(out_channels):
            for i_w in range(input_obs.shape[1]):
                for i_h in range(input_obs.shape[2]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            input_grad[c_in][i_w][i_h] += \
                            output_obs_pad[c_out][i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                            * param[c_in][c_out][p_w][p_h]
    return input_grad

def _input_grad(inp: ndarray,
                output_grad: ndarray, 
                param: ndarray) -> ndarray:

    grads = [_compute_grads_obs(inp[i], output_grad[i], param) for i in range(output_grad.shape[0])]    

    return np.stack(grads)

def _param_grad(inp: ndarray,
                output_grad: ndarray, 
                param: ndarray) -> ndarray:
    '''
    inp: [in_channels, img_width, img_height]
    output_grad_obs: [out_channels, img_width, img_height]
    param: [in_channels, out_channels, img_width, img_height]    
    '''
    param_grad = np.zeros_like(param)    
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]
    in_channels = inp.shape[1]
    out_channels = output_grad.shape[1]    

    inp_pad = _pad_conv_input(inp, param_mid)
    img_shape = output_grad.shape[2:]

    for i in range(inp.shape[0]):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_shape[0]):
                    for o_h in range(img_shape[1]):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                param_grad[c_in][c_out][p_w][p_h] += \
                                inp_pad[i][c_in][o_w+p_w][o_h+p_h] \
                                * output_grad[i][c_out][o_w][o_h]
    return param_grad

# Testing gradients
cifar_imgs = np.random.randn(10, 3, 32, 32)
cifar_param = np.random.randn(3, 16, 5, 5)

print(np.random.randint(0, cifar_imgs.shape[0])) #8
print(np.random.randint(0, cifar_imgs.shape[1])) #0
print(np.random.randint(0, cifar_imgs.shape[2])) #24 
print(np.random.randint(0, cifar_imgs.shape[3])) #22
print()
print(np.random.randint(0, cifar_param.shape[0])) #2
print(np.random.randint(0, cifar_param.shape[1])) #12
print(np.random.randint(0, cifar_param.shape[2])) #3
print(np.random.randint(0, cifar_param.shape[3])) #1

def _compute_output_sum(imgs: ndarray,
                        param: ndarray):
    return _output(imgs, param).sum()

## Input grad
cifar_imgs_2 = cifar_imgs.copy()
cifar_imgs_2[3][1][2][19] += 1

_compute_output_sum(cifar_imgs_2, cifar_param) - _compute_output_sum(cifar_imgs, cifar_param) #Out[65]: -15.587194634558728

_input_grad(cifar_imgs,
            np.ones((10, 16, 32, 32)),
            cifar_param)[3][1][2][19] #Out[66]: -15.58719463455936

## Param grad
cifar_param_2 = cifar_param.copy()
cifar_param_2[0][8][0][2] += 1

_compute_output_sum(cifar_imgs, cifar_param_2) - _compute_output_sum(cifar_imgs, cifar_param) #Out[67]: 4.016999380278094

_param_grad(cifar_imgs,
            np.ones((10, 16, 32, 32)),
            cifar_param)[0][8][0][2] #Out[68]: 4.016999380277497


##############################################################################
##############################################################################
##############################################################################
########## mnist.py
'''
Credit: https://github.com/hsjeong5
'''

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


###################################################
# Experiments: Load MNIST data
###################################################
'''
Train a small convolutional neural network to more than 90% accuracy on MNIST.
'''
X_train, y_train, X_test, y_test = load()

X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

X_train_conv, X_test_conv = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)

num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1


def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: 
    {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')
    

# CNN from scratch
model = NeuralNetwork(
    layers=[Conv2D(out_channels=16,
                   param_size=5,
                   dropout=0.8,
                   weight_init="glorot",
                   flatten=True,
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190402)

trainer = Trainer(model, SGDMomentum(lr = 0.1, momentum=0.9))
trainer.fit(X_train_conv, train_labels, X_test_conv, test_labels,
            epochs = 1,
            eval_every = 1,
            seed=20190402,
            batch_size=60,
            conv_testing=True);
'''
batch 0 loss 31.191501893742515
batch 10 loss 14.150390490522677
batch 20 loss 8.507022909161908
batch 30 loss 9.816084627772618
batch 40 loss 2.7069323023579726
batch 50 loss 5.039132957323399
batch 60 loss 3.841319313019057
batch 70 loss 8.475119811123957
batch 80 loss 5.378899040203182
batch 90 loss 2.3807974008532304
batch 100 loss 3.99368600358378
Validation accuracy after 100 batches is 86.84%
batch 110 loss 7.413863226371884
batch 120 loss 5.708919960405325
batch 130 loss 6.285799123382726
batch 140 loss 2.836869385769824
batch 150 loss 10.034980302688647
batch 160 loss 4.83542870835372
batch 170 loss 2.6279038468912037
batch 180 loss 3.065694622726854
batch 190 loss 3.3587961578277694
batch 200 loss 4.76623367361618
Validation accuracy after 200 batches is 87.92%
batch 210 loss 4.140642702073635
batch 220 loss 6.217010850836869
batch 230 loss 5.219309829373702
batch 240 loss 2.0723265950087377
batch 250 loss 3.4538781989142224
batch 260 loss 5.009145382594713
batch 270 loss 5.481472573180205
batch 280 loss 4.993055757093043
batch 290 loss 4.466142608937261
batch 300 loss 2.417707264898298
Validation accuracy after 300 batches is 88.97%
batch 310 loss 3.8468980051322035
batch 320 loss 3.2801586218053904
batch 330 loss 3.5442033106316746
batch 340 loss 1.2893697931243862
batch 350 loss 3.7784462280074127
batch 360 loss 3.8687422867316648
batch 370 loss 6.019560250741406
batch 380 loss 5.668543118128109
batch 390 loss 2.0723265950091525
batch 400 loss 3.4575684210200297
Validation accuracy after 400 batches is 87.92%
batch 410 loss 1.4972535351003
batch 420 loss 4.201519047945979
batch 430 loss 2.072326609901927
batch 440 loss 4.144654922572134
batch 450 loss 6.2377846036635685
batch 460 loss 5.526204236689966
batch 470 loss 5.075755630767164
batch 480 loss 5.756027848428946
batch 490 loss 4.164931082994363
batch 500 loss 4.293113944121421
Validation accuracy after 500 batches is 89.86%
batch 510 loss 6.907755293362459
batch 520 loss 1.3815813879223424
batch 530 loss 1.963764139346509
batch 540 loss 5.52620618165159
batch 550 loss 2.0737730702411867
batch 560 loss 4.252779562200006
batch 570 loss 3.1304080105462457
batch 580 loss 4.731670214182596
batch 590 loss 3.1473520338184433
batch 600 loss 5.114071056010041
Validation accuracy after 600 batches is 89.31%
batch 610 loss 2.948582648682471
batch 620 loss 3.4538776522371495
batch 630 loss 5.526204236689967
batch 640 loss 4.608599159579532
batch 650 loss 5.408793888633547
batch 660 loss 6.184699344728917
batch 670 loss 1.558677553823525
batch 680 loss 5.6922151008570285
batch 690 loss 3.8731253267162513
batch 700 loss 2.2707136581046314
Validation accuracy after 700 batches is 84.46%
batch 710 loss 4.835428708371727
batch 720 loss 1.3815526651020649
batch 730 loss 4.843187099890584
batch 740 loss 2.881605645794185
batch 750 loss 2.770323169803122
batch 760 loss 2.5491236579783254
batch 770 loss 2.188667217034544
batch 780 loss 4.370676478074215
batch 790 loss 3.6815365602840746
batch 800 loss 2.7631021233449835
Validation accuracy after 800 batches is 90.70%
batch 810 loss 4.231118139491239
batch 820 loss 1.9515543914856208
batch 830 loss 4.382270800562499
batch 840 loss 4.201462250323644
batch 850 loss 3.7849886441961402
batch 860 loss 0.6907776885168401
batch 870 loss 4.485660217282724
batch 880 loss 2.176841000035447
batch 890 loss 3.665548934538259
batch 900 loss 2.7631142550663257
Validation accuracy after 900 batches is 90.38%
batch 910 loss 3.5038868097974434
batch 920 loss 2.3361595551557515
batch 930 loss 2.5423480360598356
batch 940 loss 2.1107649393150685
batch 950 loss 1.400884795558769
batch 960 loss 3.808601512341384
batch 970 loss 1.6446401110285143
batch 980 loss 4.11239593917401
batch 990 loss 4.528111080424218
Validation loss after 1 epochs is 2.791
'''

calc_accuracy_model(model, X_test_conv)
#The model validation accuracy is: 
#    92.57%

##############################################################################
##############################################################################
##############################################################################

'''
CIFAR (Canadian Institute for Advanced Research) developed datasets, like the 
CIFAR-10 dataset that we can play around with.

https://www.cs.toronto.edu/~kriz/cifar.html

This dataset consists of 60,000 32 x 32 colour images in 10 classes, with 6,000
images per class. There are 50,000 training images and 10,000 test images.

The important points that distinguish this dataset from MNIST are:

Images are colored in CIFAR-10 as compared to the black and white texture of MNIST
- Each image is 32 x 32 pixel
- 50,000 training images and 10,000 testing images
Now, these images are taken in varying lighting conditions and at different angles, 
and since these are colored images, you will see that there are many variations in 
the color itself of similar objects (for example, the color of ocean water). If 
you use the simple CNN architecture that we saw in the MNIST example above, you 
will get a low validation accuracy of around 60%.

TODO: Maybe try recreating the above CNN using Keras? There's even a MATLAB version
you can download!
'''





