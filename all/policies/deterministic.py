import torch
from all.approximation import Approximation
from all.nn import RLNetwork


class DeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name='policy',
            normalise_inputs=False,
            box=None,
            **kwargs
    ):
        model = DeterministicPolicyNetwork(model, space, normalise_inputs=normalise_inputs, box=box)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class DeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space, normalise_inputs=False, box=None):
        super().__init__(model, normalize_inputs=normalise_inputs, box=box)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2).to(self.device)

    def forward(self, state):
        return self._squash(super().forward(state))

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)