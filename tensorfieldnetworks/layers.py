from math import sqrt
import math
import torch
import numpy as np
from tensorfieldnetworks import utils
from .utils import FLOAT_TYPE, EPSILON
from torch import nn as nn
from schnetpack.nn import Dense
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional
CONSTANT_BIAS = 0.0

class R(nn.Module):

    def __init__(self, input_dim, nonlin= functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):

        """ input dimension is the rbf_dimension"""

        super(R, self).__init__()
        #TODO: figure out how to initialize weights
        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.R_net = nn.Sequential( Dense(input_dim, hidden_dim, bias = True,
            activation = nonlin, bias_init = lambda x: init.constant_(x,CONSTANT_BIAS) ),
                        Dense(hidden_dim, output_dim, bias = True,
            activation = None,  bias_init = lambda x: init.constant_(x,CONSTANT_BIAS)))


    def forward(self, inputs):
        #input dim: inputs.size()[-1] (rbf-dimension)
        return self.R_net(inputs)


def unit_vectors(v, axis=-1):
    return v / utils.norm_with_epsilon(v, axis=axis, keep_dims=True)


def Y_2(rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = torch.max(torch.sum(rij**2, dim=-1), EPSILON)
    # return : [N, N, 5]
    output = torch.stack([x * y / r2,
                       y * z / r2,
                       (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * sqrt(3) * r2),
                       z * x / r2,
                       (tf.square(x) - tf.square(y)) / (2. * r2)],
                      dim=-1)
    return output


class F(nn.Module):

    def __init__(self, l, input_dim, nonlin=functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):

        super(F, self).__init__()
        self.radial = R(input_dim, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                       weights_initializer=weights_initializer, biases_initializer=biases_initializer)

        self.output_dim = output_dim
        """ input dimension is the rbf_dimension"""
        if l == 0 :
            self.rep = None
            self.forward = self.forward_no_rep
        elif l == 1:
            self.rep = unit_vectors
            self.forward = self.forward_rep
        elif l == 2:
            self.rep == Y_2
            self.forward = self.forward_rep

    def forward_no_rep(self, rbf_input, rij):

        radial = self.radial(rbf_input)
        return radial.unsqueeze(-1)

    def forward_rep(self, rbf_input, rij):

        radial = self.radial(rbf_input)
        dij = torch.norm(rij, dim=-1)
        condition = (dij < EPSILON).unsqueeze(-1).repeat(1, 1, self.output_dim)
        masked_radial = torch.where(condition, torch.zeros_like(radial),
            radial)
        return self.rep(rij).unsqueeze(-2) * masked_radial.unsqueeze(-1)



class Filter(nn.Module):

    def __init__(self, l_filter, l_out, input_dim, nonlin=functional.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
        """ 4.1.3 Layer definition
            l1: l_input
            l2: l_output
            input_dim: rbf_dimension
        """

        super(Filter, self).__init__()
        self.F_out = F(l_filter, input_dim,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)

        self.eijk = utils.get_eijk()
        self.l_filter = l_filter
        self.l_out = l_out


        if l_out == 0:
            if l_filter == 0:
                self.cg = None
                self.forward = self.forward_00
            elif l_filter == 1:
                self.cg = torch.eye(3).unsqueeze(0)
        elif l_out == 1:
            self.cg = {1 : torch.eye(3).unsqueeze(-1), 3 : self.eijk}
            self.forward = self.forward_1
        elif l_out == 2:
            self.cg = torch.eye(5).unsqueeze(-1)
        else:
            raise ValueError('l2 = {} not implemented'.format(l2))

    def forward(self, layer_input, rbf_input, rij):
        cg = self.cg
        return torch.einsum('ijk,abfj,bfk->afi', cg, self.F_out(rbf_input, rij),
            layer_input)

    def forward_00(self, layer_input, rbf_input, rij):
        cg = torch.eye(layer_input.size()[-1]).unsqueeze(-2)
        return torch.einsum('ijk,abfj,bfk->afi', cg, self.F_out(rbf_input, rij),
            layer_input)

    def forward_1(self, layer_input, rbf_input, rij):
        cg = self.cg[layer_input.size()[-1]]
        return torch.einsum('ijk,abfj,bfk->afi', cg, self.F_out(rbf_input, rij),
            layer_input)


class Convolution(nn.Module):

    def __init__(self, input_dim, output_dim=1, weights_initializer=None,
        biases_initializer=None):
        """
        input_dim: rbf_dimension
        """
        super(Convolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f00 = {}
        self.f10 = {}
        self.f11 = {}
        self.forward = self.forward_build

    def forward_build(self, input_tensor_list, rbf, rij):
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                # L x 0 -> L
                name = '{}_{}'.format(key,i)
                self.f00[name] = Filter(0,0,input_dim = self.input_dim,
                    output_dim = self.output_dim)

                if key is 1:
                    # L x 1 -> 0
                    self.f10[name] = Filter(1,0,input_dim = self.input_dim,
                    output_dim = self.output_dim)

                if key is 0 or key is 1:
                    # L x 1 -> 1
                    self.f11[name] = Filter(1,1,input_dim = self.input_dim,
                    output_dim = self.output_dim)

        self.f00 = torch.nn.ModuleDict(self.f00)
        self.f10 = torch.nn.ModuleDict(self.f10)
        self.f11 = torch.nn.ModuleDict(self.f11)
        self.forward = self.forward_later

        return self.forward(input_tensor_list, rbf, rij)

    def forward_later(self, input_tensor_list, rbf, rij):

        output_tensor_list = {0:[], 1:[]}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = '{}_{}'.format(key,i)
                # L x 0 -> L
                tensor_out = self.f00[name](tensor,
                                      rbf,
                                      rij)
                output_tensor_list[key].append(tensor_out)
                if key is 1:
                    # L x 1 -> 0
                    tensor_out = self.f10[name](tensor,
                                          rbf,
                                          rij)
                    output_tensor_list[0].append(tensor_out)
                if key is 0 or key is 1:
                    # L x 1 -> 1
                    tensor_out = self.f11[name](tensor,
                                          rbf,
                                          rij)
                    output_tensor_list[1].append(tensor_out)
        return output_tensor_list

class SelfInteractionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, bias = False,
            weights_initializer=None, biases_initializer=None):
        #TODO: weights intializer
        """
            input_dim: channel_dimension
        """

        super(SelfInteractionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.bias = torch.zeros(output_dim)
            # self.register_parameter('bias', None)
        self.reset_parameters()
        self.forward = self.first_forward

    def reset_parameters(self):
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, CONSTANT_BIAS)

    def first_forward(self, layer_input):
        if not layer_input.size()[-2] == self.input_dim:
            self.input_dim = layer_input.size()[-2]
            self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))
            self.reset_parameters()
        self.forward = self.later_forward
        return self.forward(layer_input)

    def later_forward(self, layer_input):
        # Size (number of channels) needs to be inferred from input here

        return (torch.einsum('afi,gf->aig',
            layer_input, self.weight) + self.bias).permute(0, 2, 1)


class SelfInteraction(nn.Module):

    def __init__(self, input_dim, output_dim, weights_initializer=None,
        biases_initializer=None):

        """
            input_dim: channel_dimension
        """
        super(SelfInteraction, self).__init__()
        # self.SI_with_biases = SelfInteractionLayer(input_dim, output_dim, True,
            # weights_initializer, biases_initializer)
        # self.SI_without_biases = SelfInteractionLayer(input_dim, output_dim, False,
            # weights_initializer, biases_initializer)

        self.SI = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights_init = weights_initializer
        self.biases_init = biases_initializer

        self.forward = self.forward_init

    def forward_init(self, input_tensor_list):
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = '{}_{}'.format(key,i)
                if key ==0:
                    self.SI[name] = SelfInteractionLayer(self.input_dim, self.output_dim, True,
                        self.weights_init, self.biases_init)
                else:
                    self.SI[name] = SelfInteractionLayer(self.input_dim, self.output_dim, False,
                        self.weights_init, self.biases_init)

        self.SI = torch.nn.ModuleDict(self.SI)
        self.forward = self.forward_later
        return self.forward(input_tensor_list)

    def forward_later(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                tensor_out = self.SI['{}_{}'.format(key,i)](tensor)
                output_tensor_list[key].append(tensor_out)
        return output_tensor_list

class NonLinearity(nn.Module):

    def __init__(self, channels, nonlin=functional.elu, biases_initializer=None):

        super(NonLinearity, self).__init__()
        self.biases_initializer = biases_initializer
        self.nonlin = nonlin
        self.channels = channels
        self.biases = {}
        self.forward = self.forward_init
        # self.biases =  Parameter(torch.Tensor(channels))
        # self.reset_parameters()

    def forward_init(self, input_tensor_list):
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                name = '{}_{}'.format(key,i)
                self.biases[name] = Parameter(torch.Tensor(self.channels))

        self.biases = torch.nn.ParameterDict(self.biases)
        self.forward = self.forward_later
        self.reset_parameters()
        return self.forward(input_tensor_list)

    def forward_later(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                tensor_out = utils.rotation_equivariant_nonlinearity(tensor,
                                                                     nonlin=self.nonlin,
                                                                     biases=self.biases['{}_{}'.format(key,i)])
#                m = 0 if tensor_out.size()[-1] == 1 else 1
                # output_tensor_list[m].append(tensor_out)
                output_tensor_list[key].append(tensor_out)
        return output_tensor_list

    def reset_parameters(self):
        # init.zeros_(self.biases)
        for key in self.biases:
            init.constant_(self.biases[key], CONSTANT_BIAS)

class Concatenation(nn.Module):

    def __init__(self):
        super(Concatenation, self).__init__()

    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            # Concatenate along channel axis
            # [N, channels, M]
            output_tensor_list[key].append(torch.cat(input_tensor_list[key], dim=-2))
        return output_tensor_list
