import torch
from all.environments import State
from .approximation import Approximation
from all.nn import Normalizer

class FeatureNetwork(Approximation):
    def __init__(self, model, optimizer=None, name='feature', normalize_input=False, **kwargs):
        model = FeatureModule(model, normalize_input=normalize_input)
        super().__init__(model, optimizer, name=name, **kwargs)
        self._cache = []
        self._out = []

    def __call__(self, states):
        features = self.model(states)
        graphs = features.raw
        # pylint: disable=protected-access
        features._raw = graphs.detach()
        features._raw.requires_grad = True
        self._enqueue(graphs, features._raw)
        return features

    def reinforce(self):
        graphs, grads = self._dequeue()
        graphs.backward(grads)
        self.step()

    def _enqueue(self, features, out):
        self._cache.append(features)
        self._out.append(out)

    def _dequeue(self):
        graphs = []
        grads = []
        for graph, out in zip(self._cache, self._out):
            if out.grad is not None:
                graphs.append(graph)
                grads.append(out.grad)
        self._cache = []
        self._out = []
        return torch.cat(graphs), torch.cat(grads)

class FeatureModule(torch.nn.Module):
    def __init__(self, model, normalize_input=False):
        super().__init__()
        self.model = model
        self._normalizer = Normalizer() if normalize_input is True else None

    def forward(self, states):
        if self._normalizer is not None:
            with torch.no_grad():
                input_state = self._normalizer.normalize(states.raw)
        else:
            input_state = states.features

        features = self.model(input_state.float())
        return State(
            features,
            mask=states.mask,
            info=states.info
        )
