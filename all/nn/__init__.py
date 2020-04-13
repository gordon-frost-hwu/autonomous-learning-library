import torch
from torch import nn
from torch.nn import *  # export everthing
from torch.nn import functional as F
import numpy as np
from all.environments import State

class Normalizer(object):
    def __init__(self):
        self.initialised = False
        self.n = None
        self.mean = None
        self.mean_diff = None
        self.var = None

    def initialise(self, inputs):
        num_inputs = inputs[0].shape
        device = inputs.device
        self.n = torch.zeros(num_inputs).to(device)
        self.mean = torch.zeros(num_inputs).to(device)
        self.mean_diff = torch.zeros(num_inputs).to(device)
        self.var = torch.zeros(num_inputs).to(device)
        self.initialised = True

    def _observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        if type(inputs) is State:
            inputs = inputs.raw

        if not self.initialised:
            self.initialise(inputs)

        result = torch.empty(inputs.shape[0], inputs.shape[1], dtype=torch.float).to("cuda")

        idx = 0
        for input in inputs:
            self._observe(input)
            obs_std = torch.sqrt(self.var)
            result[idx, :] = (input - self.mean)/obs_std
        return result

class newNormalizer:
    def __init__(self, size=0):
        self.maxValues = np.zeros((1, size))
        self.minValues = np.zeros((1, size))

    def setMinMaxFromGymEnv(self, box):
        self.maxValues = torch.tensor(box.high, device="cuda")
        self.minValues = torch.tensor(box.low, device="cuda")

    def update(self, values):
        compMax = np.greater(values, self.maxValues)
        idxMax = np.where(compMax==True)
        np.put(self.maxValues, idxMax, values[idxMax])
        compMin = np.less(values, self.minValues)
        idxMin = np.where(compMin==True)
        np.put(self.minValues, idxMin, values[idxMin])

    def normalize(self, values):
        normVal = 2*(values - self.minValues) / (self.maxValues - self.minValues) - 1
        return normVal

    def normalize_batch(self, X):
        normVal = np.zeros(np.shape(X))
        for i, values in enumerate(X):
            aux = 2*(values - self.minValues) / (self.maxValues - self.minValues) - 1
            normVal[i] = aux
        return normVal


class RLNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """

    def __init__(self, model, _=None, normalize_inputs=False, box=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self._normalizer = None
        if normalize_inputs is True and box is not None:
            self._normalizer = newNormalizer()
            self._normalizer.setMinMaxFromGymEnv(box)
        self.count = 0

    def forward(self, state):
        if self._normalizer is not None:
            # print("input state: {0}".format(state.raw))

            with torch.no_grad():
                features = self._normalizer.normalize(state.features)
            # print("features: {0}".format(features))

        else:
            features = state.features.float()
        # print(features)
        #
        # if self.count == 5:
        #     exit(0)
        # self.count += 1
        return self.model(features) * state.mask.float().unsqueeze(-1)


class Aggregation(nn.Module):
    """len()
    Aggregation layer for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This layer computes a Q function by combining
    an estimate of V with an estimate of the advantage.
    The advantage is normalized by substracting the average
    advantage so that we can propertly
    """

    def forward(self, value, advantages):
        return value + advantages - torch.mean(advantages, dim=1, keepdim=True)


class Dueling(nn.Module):
    """
    Implementation of the head for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This module computes a Q function by computing
    an estimate of V, and estimate of the advantage,
    and combining them with a special Aggregation layer.
    """

    def __init__(self, value_model, advantage_model):
        super(Dueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model
        self.aggregation = Aggregation()

    def forward(self, features):
        value = self.value_model(features)
        advantages = self.advantage_model(features)
        return self.aggregation(value, advantages)


class CategoricalDueling(nn.Module):
    """Dueling architecture for C51/Rainbow"""

    def __init__(self, value_model, advantage_model):
        super(CategoricalDueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model

    def forward(self, features):
        batch_size = len(features)
        value_dist = self.value_model(features)
        atoms = value_dist.shape[1]
        advantage_dist = self.advantage_model(features).view((batch_size, -1, atoms))
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        return (
            value_dist.view((batch_size, 1, atoms)) + advantage_dist - advantage_mean
        ).view((batch_size, -1))


class Flatten(nn.Module):  # pylint: disable=function-redefined
    """
    Flatten a tensor, e.g., between conv2d and linear layers.

    The maintainers FINALLY added this to torch.nn, but I am
    leaving it in for compatible for the moment.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class NoisyLinear(nn.Linear):
    """
    Implementation of Linear layer for NoisyNets

    https://arxiv.org/abs/1706.10295
    NoisyNets are a replacement for epsilon greedy exploration.
    Gaussian noise is added to the weights of the output layer, resulting in
    a stochastic policy. Exploration is implicitly learned at a per-state
    and per-action level, resulting in smarter exploration.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(sigma_init)
        )
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, x):
        bias = self.bias

        if not self.training:
            return F.linear(x, self.weight, bias)

        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        if self.bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_init=0.4, init_scale=3, bias=True):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_init / np.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(sigma_init)
        )
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.Tensor(out_features).fill_(sigma_init)
            )

    def reset_parameters(self):
        std = np.sqrt(self.init_scale / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)

        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class TanhActionBound(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.register_buffer(
            "weight", torch.tensor((action_space.high - action_space.low) / 2)
        )
        self.register_buffer(
            "bias", torch.tensor((action_space.high + action_space.low) / 2)
        )

    def forward(self, x):
        return torch.tanh(x) * self.weight + self.bias

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())

    return _loss

def weighted_mse_loss(input, target, weight, reduction='mean'):
    loss = (weight * ((target - input) ** 2))
    return torch.mean(loss) if reduction == 'mean' else torch.sum(loss)

def weighted_smooth_l1_loss(input, target, weight, reduction='mean'):
    t = torch.abs(input - target)
    loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    loss = weight * loss
    return torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
