#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:01:40 2021

@author: joanaperdomo
"""
'''
In this chapter we will experiment with 
    - Loss functions
    - Learning rate decay
    - Weight initialization
    - Optimizers
    - Dropout
'''

###################################################
# lincoln imports
###################################################
'''
lincoln is not currently a pip installable library.

To install, I cloned the lincoln github library (https://github.com/SethHWeidman/lincoln).

After cloning the library, I navigated to the local repo on my machine
and went to the terminal. I then pip installed the local library using the 
following command:
    python3 -m pip install lincoln


'''
import numpy as np
from numpy import ndarray
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from typing import List
from copy import deepcopy
from urllib import request
import gzip
import pickle



########## utils
def to_2d(a: np.ndarray,
          type: str="col") -> np.ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)


def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]


def permute_data(X: np.ndarray, y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


Batch = Tuple[np.ndarray, np.ndarray]


def generate_batch(X: np.ndarray,
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:

    assert (X.dim() == 2) and (y.dim() == 2), \
        "X and Y must be 2 dimensional"

    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]

    return X_batch, y_batch


def assert_same_shape(output: np.ndarray,
                      output_grad: np.ndarray):
    assert output.shape == output_grad.shape, \
        '''
        Two tensors should have the same shape;
        instead, first Tensor's shape is {0}
        and second Tensor's shape is {1}.
        '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None


def assert_dim(t: np.ndarray,
               dim: int):
    assert t.ndim == dim, \
        '''
        Tensor expected to have dimension {0}, instead has dimension {1}
        '''.format(dim, len(t.shape))
    return None


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

########## base   
class Operation(object):

    def __init__(self):
        pass

    def forward(self,
                input_: ndarray,
                inference: bool=False) -> ndarray:

        self.input_ = input_

        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


########## activation
class Linear(Operation):
    '''
    Linear activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)


class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        mask = self.output >= 0
        return output_grad * mask
    

        

########## conv
class Conv2D_Op(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, inp: ndarray) -> ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self,
                      inp: ndarray) -> ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self,
                    inp: ndarray):
        '''
        Input is a 2 dimensional, square, 2D Tensor
        '''
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])


    # def _pad_2d(self,
    #             inp: ndarray):
    #     '''
    #     Input is a 3 dimensional tensor, first dimension batch size
    #     '''
    #     outs = [self._pad_2d_obs(obs, self.param_pad) for obs in inp]
    #
    #     return np.stack(outs)

    def _pad_2d_channel(self,
                        inp: ndarray):
        '''
        inp has dimension [num_channels, image_width, image_height]
        '''
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self,
                           input_: ndarray):
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height-self.param_size+1):
            for w in range(img_height-self.param_size+1):
                patch = imgs_batch_pad[:, :, h:h+self.param_size, w:w+self.param_size]
                patches.append(patch)
        return np.stack(patches)

    def _output(self,
                inference: bool = False):
        '''
        conv_in: [batch_size, channels, img_width, img_height]
        param: [in_channels, out_channels, fil_width, fil_height]
        '''
    #     assert_dim(obs, 4)
    #     assert_dim(param, 4)
        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)

        patches_reshaped = (patches
                            .transpose(1, 0, 2, 3, 4)
                            .reshape(batch_size, img_size, -1))

        param_reshaped = (self.param
                          .transpose(0, 2, 3, 1)
                          .reshape(patch_size, -1))

        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2))

        return output_reshaped


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (self._get_image_patches(output_grad)
                          .transpose(1, 0, 2, 3, 4)
                          .reshape(batch_size * img_size, -1))

        param_reshaped = (self.param
                          .reshape(self.param.shape[0], -1)
                          .transpose(1, 0))

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )


    def _param_grad(self, output_grad: ndarray) -> ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
            )

        out_grad_reshape = (output_grad
                            .transpose(0, 2, 3, 1)
                            .reshape(batch_size * img_size, -1))

        return (np.matmul(in_patches_reshape,
                          out_grad_reshape)
                .reshape(in_channels, self.param_size, self.param_size, out_channels)
                .transpose(0, 3, 1, 2))


########## dense
class WeightMultiply(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self, inference: bool) -> ndarray:
        return np.matmul(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        return np.matmul(self.input_.transpose(1, 0), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: ndarray):
        super().__init__(B)

    def _output(self, inference: bool) -> ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        output_grad_reshape = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshape


########## dropout
class Dropout(Operation):

    def __init__(self,
                 keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> ndarray:
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask

########## reshape
class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = False) -> ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self.input_.shape)


########## layers
class Layer(object):

    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: ndarray) -> None:
        pass

    def forward(self, input_: ndarray,
                inference=False) -> ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):

    def __init__(self,
                 neurons: int,
                 activation: Operation = Linear(),
                 conv_in: bool = False,
                 dropout: float = 1.0,
                 weight_init: str = "standard") -> None:
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> None:
        np.random.seed(self.seed)
        num_in = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(num_in + self.neurons)
        else:
            scale = 1.0

        # weights
        self.params = []
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(num_in, self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None


class Conv2D(Layer):
    '''
    Once we define all the Operations and the outline of a layer,
    all that remains to implement here is the _setup_layer function!
    '''
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 dropout: int = 1.0,
                 weight_init: str = "normal",
                 activation: Operation = Linear(),
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels

    def _setup_layer(self, input_: ndarray) -> ndarray:

        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc=0,
                                      scale=scale,
                                      size=(input_.shape[1],  # input channels
                                     self.out_channels,
                                     self.param_size,
                                     self.param_size))

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None


########## losses
class Loss(object):

    def __init__(self):
        pass

    def forward(self,
                prediction: ndarray,
                target: ndarray) -> float:

        # batch size x num_classes
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        self.output = self._output()

        return self.output

    def backward(self) -> ndarray:

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self,
                 normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:

        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)

        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:

        # if the network is just outputting probabilities
        # of just belonging to one class:
        if self.target.shape[1] == 0:
            self.single_class = True

        # if "single_class", apply the "normalize" operation defined above:
        if self.single_class:
            self.prediction, self.target = \
            normalize(self.prediction), normalize(self.target)

        # applying the softmax function to each row (observation)
        softmax_preds = softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = (
            -1.0 * self.target * np.log(self.softmax_preds) - \
                (1.0 - self.target) * np.log(1 - self.softmax_preds)
        )

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:

        # if "single_class", "un-normalize" probabilities before returning gradient:
        if self.single_class:
            return unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]


# class SoftmaxCrossEntropyComplex(SoftmaxCrossEntropy):
#     def __init__(self, eta: float=1e-9,
#                  single_output: bool = False) -> None:
#         super().__init__()
#         self.single_output = single_output

#     def _input_grad(self) -> ndarray:

#         prob_grads = []
#         batch_size = self.softmax_preds.shape[0]
#         num_features = self.softmax_preds.shape[1]
#         for n in range(batch_size):
#             exp_ratio = exp_ratios(self.prediction[n] - np.max(self.prediction[n]))
#             jacobian = np.zeros((num_features, num_features))
#             for f1 in range(num_features):  # p index
#                 for f2 in range(num_features):  # SCE index
#                     if f1 == f2:
#                         jacobian[f1][f2] = (
#                             self.softmax_preds[n][f1] - self.target[n][f1])
#                     else:
#                         jacobian[f1][f2] = (
#                             -(self.target[n][f2]-1) * exp_ratio[f1][f2] + self.target[n][f2] + self.softmax_preds[n][f1] - 1)
#             prob_grads.append(jacobian.sum(axis=1))

#         if self.single_class:
#             return unnormalize(np.stack(prob_grads))
#         else:
#             return np.stack(prob_grads)



########## network
class LayerBlock(object):

    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers

    def forward(self,
                X_batch: ndarray,
                inference=False) ->  ndarray:

        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out

    def backward(self, loss_grad: ndarray) -> ndarray:

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss = MeanSquaredError,
                 seed: int = 1):
        super().__init__(layers)
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward_loss(self,
                     X_batch: ndarray,
                     y_batch: ndarray,
                     inference: bool = False) -> float:

        prediction = self.forward(X_batch, inference)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self,
                    X_batch: ndarray,
                    y_batch: ndarray,
                    inference: bool = False) -> float:

        prediction = self.forward(X_batch, inference)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss



########## optimizers
class Optimizer(object):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:

        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.lr,
                                       1.0 / (self.max_epochs - 1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:

        if not self.decay_type:
            return

        if self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch

        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch

    def step(self,
             epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:

        update = self.lr*kwargs['grad']
        kwargs['param'] -= update

class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9) -> None:
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param,
                              grad=param_grad,
                              velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

            # Update velocity
            kwargs['velocity'] *= self.momentum
            kwargs['velocity'] += self.lr * kwargs['grad']

            # Use this to update parameters
            kwargs['param'] -= kwargs['velocity']


class AdaGrad(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr_exp: float = 0,
                 final_lr_linear: float = 0) -> None:
        super().__init__(lr, final_lr_exp, final_lr_linear)
        self.eps = 1e-7

    def step(self) -> None:
        if self.first:
            self.sum_squares = [np.zeros_like(param)
                                for param in self.net.params()]
            self.first = False

        for (param, param_grad, sum_square) in zip(self.net.params(),
                                                   self.net.param_grads(),
                                                   self.sum_squares):
            self._update_rule(param=param,
                              grad=param_grad,
                              sum_square=sum_square)

    def _update_rule(self, **kwargs) -> None:

            # Update running sum of squares
            kwargs['sum_square'] += (self.eps +
                                     np.power(kwargs['grad'], 2))

            # Scale learning rate by running sum of squareds=5
            lr = np.divide(self.lr, np.sqrt(kwargs['sum_square']))

            # Use this to update parameters
            kwargs['param'] -= lr * kwargs['grad']


class RegularizedSGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 alpha: float = 0.1) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

            # Use this to update parameters
            kwargs['param'] -= (
                self.lr * kwargs['grad'] + self.alpha * kwargs['param'])





########## train
class Trainer(object):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True,
            early_stopping: bool = True,
            conv_testing: bool = False)-> None:

        setattr(self.optim, 'max_epochs', epochs)
        self.optim._setup_decay()

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

                if conv_testing:
                    if ii % 10 == 0:
                        test_preds = self.net.forward(X_batch, inference=True)
                        batch_loss = self.net.loss.forward(test_preds, y_batch)
                        print("batch",
                              ii,
                              "loss",
                              batch_loss)

                    if ii % 100 == 0 and ii > 0:
                        print("Validation accuracy after", ii, "batches is",
                        f'''{np.equal(np.argmax(self.net.forward(X_test, inference=True), axis=1),
                        np.argmax(y_test, axis=1)).sum() * 100.0 / X_test.shape[0]:.2f}%''')

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print()
                        print(f"Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f},",
                              f"\nusing the model from epoch {e+1-eval_every}")
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, 'net', self.net)
                        break
                else:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()


    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> Tuple[ndarray]:

        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch





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
import mnist

'''
Download the MNIST data. Read both the images and their corresponding labels into
training and testing sets. These are now vectors.
'''
#mnist.init()
X_train, y_train, X_test, y_test = load()


'''
one-hot encoding to transform the vectors representing the labels into an ndarray
of the same shape as the predictions.
'''
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

'''
scale the data to mean 0 and variance 1. To do this we will provide a global
scaling to the dataset hat subtracts off the overall mean and divides by the 
overall variance.
'''
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

###################################################
# Experiments: Softmax Cross Entropy
###################################################
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Sigmoid())],
            loss = MeanSquaredError(normalize=False), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.611
Validation loss after 20 epochs is 0.428
Validation loss after 30 epochs is 0.388
Validation loss after 40 epochs is 0.375
Validation loss after 50 epochs is 0.366

The model validation accuracy is: 72.51%

Note: Even if we normalize the outputs of a classification model with mean 
squared error loss, it still doesn't help.
'''


model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Sigmoid())],
            loss = MeanSquaredError(normalize=True), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)    
'''
Validation loss after 10 epochs is 0.952

Loss increased after epoch 20, final loss was 0.952, 
using the model from epoch 10
The model validation accuracy is: 41.73%
'''

# Trying Sigmoid Activation
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Sigmoid()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 130,
            eval_every = 1,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

'''
Validation loss after 1 epochs is 1.285
Validation loss after 2 epochs is 0.970
Validation loss after 3 epochs is 0.836
Validation loss after 4 epochs is 0.763
Validation loss after 5 epochs is 0.712
Validation loss after 6 epochs is 0.679
Validation loss after 7 epochs is 0.651
Validation loss after 8 epochs is 0.631
Validation loss after 9 epochs is 0.617
Validation loss after 10 epochs is 0.599
Validation loss after 11 epochs is 0.588
Validation loss after 12 epochs is 0.576
Validation loss after 13 epochs is 0.568
Validation loss after 14 epochs is 0.557
Validation loss after 15 epochs is 0.550
Validation loss after 16 epochs is 0.544
Validation loss after 17 epochs is 0.537
Validation loss after 18 epochs is 0.533
Validation loss after 19 epochs is 0.529
Validation loss after 20 epochs is 0.523
Validation loss after 21 epochs is 0.517
Validation loss after 22 epochs is 0.512
Validation loss after 23 epochs is 0.507

Loss increased after epoch 24, final loss was 0.507, 
using the model from epoch 23

The model validation accuracy is: 91.04%
'''

# Trying ReLU activation
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=ReLU()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 6.456
Validation loss after 20 epochs is 6.066

Loss increased after epoch 30, final loss was 6.066, 
using the model from epoch 20

The model validation accuracy is: 75.31%
'''


model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGD(0.1))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);
print()
calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.641
Validation loss after 20 epochs is 0.581
Validation loss after 30 epochs is 0.561

Loss increased after epoch 40, final loss was 0.561, 
using the model from epoch 30

The model validation accuracy is: 90.54%
'''

###################################################
# Experiments: SGD Momentum
###################################################
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Sigmoid()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optim = SGDMomentum(0.1, momentum=0.9)

trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 1,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)

'''
Validation loss after 1 epochs is 0.615
Validation loss after 2 epochs is 0.489
Validation loss after 3 epochs is 0.446

Loss increased after epoch 4, final loss was 0.446, 
using the model from epoch 3
The model validation accuracy is: 92.05%
'''

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optim = SGD(0.1)

optim = SGDMomentum(0.1, momentum=0.9)

trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.367
Validation loss after 20 epochs is 0.333
Validation loss after 30 epochs is 0.316

Loss increased after epoch 40, final loss was 0.316, 
using the model from epoch 30
The model validation accuracy is: 95.23%
'''


###################################################
# Experiments: Different Weight Decay
###################################################

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.402
Validation loss after 20 epochs is 0.314
Validation loss after 30 epochs is 0.306
Validation loss after 40 epochs is 0.301

Loss increased after epoch 50, final loss was 0.301, 
using the model from epoch 40
The model validation accuracy is: 95.83%
'''


model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.2, 
                        momentum=0.9, 
                        final_lr = 0.05, 
                        decay_type='exponential')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 50,
            eval_every = 10,
            seed=20190119,
            batch_size=60);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.504
Validation loss after 20 epochs is 0.337
Validation loss after 30 epochs is 0.291

Loss increased after epoch 40, final loss was 0.291, 
using the model from epoch 30
The model validation accuracy is: 95.94%
'''

###################################################
# Experiments: Changing Weight Init
###################################################
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')

trainer = Trainer(model, optimizer)
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.296
Validation loss after 20 epochs is 0.241
Validation loss after 30 epochs is 0.238
Validation loss after 40 epochs is 0.232

Loss increased after epoch 50, final loss was 0.232, 
using the model from epoch 40
The model validation accuracy is: 97.05%
'''

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.442
Validation loss after 20 epochs is 0.262
Validation loss after 30 epochs is 0.248

Loss increased after epoch 40, final loss was 0.248, 
using the model from epoch 30
The model validation accuracy is: 96.27%
'''

###################################################
# Experiments: Dropout
###################################################
model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 50,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.257
Validation loss after 20 epochs is 0.229
Validation loss after 30 epochs is 0.198
Validation loss after 40 epochs is 0.194

Loss increased after epoch 50, final loss was 0.194, 
using the model from epoch 40
The model validation accuracy is: 96.86%
'''

###################################################
# Experiments: Deep Learning with and without Dropout
###################################################
model = NeuralNetwork(
    layers=[Dense(neurons=178, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=46, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 100,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)

'''
Validation loss after 10 epochs is 0.316
Validation loss after 20 epochs is 0.286
Validation loss after 30 epochs is 0.243
Validation loss after 40 epochs is 0.241
Validation loss after 50 epochs is 0.200

Loss increased after epoch 60, final loss was 0.200, 
using the model from epoch 50
The model validation accuracy is: 96.75%
'''


model = NeuralNetwork(
    layers=[Dense(neurons=178, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=46, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train, train_labels, X_test, test_labels,
       epochs = 100,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=True);

calc_accuracy_model(model, X_test)

'''
Output from spyder call 'get_cwd':
Validation loss after 10 epochs is 0.431
Validation loss after 20 epochs is 0.362
Validation loss after 30 epochs is 0.319
Validation loss after 40 epochs is 0.287
Validation loss after 50 epochs is 0.264

Loss increased after epoch 60, final loss was 0.264, 
using the model from epoch 50
The model validation accuracy is: 95.91%
'''



###################################################
# Experiments: 
###################################################
'''
Import modules
'''
# import pandas
import pandas as pd

# import numpy
import numpy as np

# import seaborn
import seaborn as sb

# import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import math

'''

'''
# creating Dataframe object
df = pd.read_csv('winequalityN.csv')
print(df.head())
print(df.info())
print(df.describe())


'''
Visualization
'''
df.hist(bins=25,figsize=(10,10))
# display histogram
plt.show()

'''
The above histogram shows that the data is easily distributed on features.

Now we can plot the bar graph to check what value of alcohol can be used to make
changes in quality.
'''

plt.figure(figsize=[10,6])
# plot bar graph
plt.bar(df['quality'],df['alcohol'],color='red')
# label x-axis
plt.xlabel('quality')
#label y-axis
plt.ylabel('alcohol')

'''Correlation'''
# ploting heatmap
plt.figure(figsize=[19,10])
sb.heatmap(df.corr(),annot=True)

'''Prepare Data'''
df.head()
labels = np.unique(df['type'].values)
idx_to_labels = { k:v for k,v in enumerate(labels) }
labels_to_idx = { v:k for k,v in enumerate(labels) }

labels = df.replace(labels_to_idx)['type'].values
df = df.drop(columns=['type'])

# one hot encoding
labels = np.eye(len(idx_to_labels))[labels]

# normalized data
df = (df-df.mean())/df.std()

# replace NaN with Standard Deviation
df = df.fillna(df.std())

df.head()
features = df.values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

'''Parameters'''
gamma = {}
gamma["ndims"] = features.shape[1]
gamma["nclasses"] = len(idx_to_labels.values())


'''Generate Weights'''
np.random.seed(1337)
def generate_weights(gamma):
    '''
        Generate Weights and use Xavier Initiation
    '''
    scale = 1/max(1., (2+2)/2.)
    limit = math.sqrt(3.0 * scale)

    gamma['w0'] = np.random.uniform(-limit, limit, size=(gamma['ndims'], gamma['ndims']))
    gamma['w1'] = np.random.uniform(-limit, limit, size=(gamma['ndims'], gamma['nclasses']))
    
    return gamma

gamma = generate_weights(gamma)
print('w0 shape:', gamma['w0'].shape, ' - w1 shape:', gamma['w1'].shape)

'''Activation Function and Derivative'''
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dsigmoid(x):
    return x * (1. - x)

''' Training '''
def loss(y, y_hat):
    '''
        Addition of all Squared Mean Errors
    '''
    return np.sum(np.mean(np.square(np.subtract(y, y_hat)), axis=0))

def forward(X, gamma):
    '''
        Forward Propagation
    '''
    l0 = X
    l1 = sigmoid(np.dot(l0, gamma['w0']))
    l2 = sigmoid(np.dot(l1, gamma['w1']))
    
    return l0, l1, l2

def backward(y, theta, gamma, lr):
    '''
        Backward Propagation
    '''
    l0, l1, l2 = theta
    
    l2_error = y - l2
    l2_delta = l2_error * dsigmoid(l2)
    l1_error = l2_delta.dot(gamma['w1'].T)
    l1_delta = l1_error * dsigmoid(l1)
    
    # update using SGD
    gamma['w0'] += lr * l0.T.dot(l1_delta)
    gamma['w1'] += lr * l1.T.dot(l2_delta)
    
    return gamma

def train(X, y, gamma, iterations=60, lr=0.01):
    '''
        Function to Train Dataset
    '''
    errors = []
    for i in range(iterations):
        # forward propagation
        theta = forward(X, gamma)
        
        e = loss(theta[-1], y)
        if(i % 4 == 0):
            print('I:{0:4d}, --  Mean Error:{1:1.4f}'.format(i, np.mean(e)))
        errors.append(e)

        # backward propagation
        gamma = backward(y, theta, gamma, lr)
            
    return gamma, errors

gamma, errors = train(X_train, y_train, gamma)

''' Plot Error Lost '''
plt.plot(errors)

''' Accuracy '''
def accuracy(y, gamma):
    '''
    Function to calculate accuracy
    '''
    acc_y = []
    for x in X_test:
        y = np.argmax(forward(x.reshape(1, 12), gamma)[-1])
        y = np.eye(gamma["nclasses"])[y]
        acc_y.append(y)

    acc_y = np.array(acc_y)
    wrong = len(np.where(np.equal(y_test, acc_y).astype(int) == 0)[0])
    return 1 - (len(y) / wrong)

print('Accuracy:{0:3d}%'.format(int(accuracy(y_train, gamma) * 100)))

# Accuracy is 90%





