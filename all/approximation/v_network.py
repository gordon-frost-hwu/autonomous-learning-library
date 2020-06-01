from all.nn import RLNetwork
from .approximation import Approximation

class VNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='v',
            quiet=False,
            normalise_inputs=False,
            box=None,
            **kwargs
    ):
        model = VModule(model, normalize_inputs=normalise_inputs, box=box)
        super().__init__(
            model,
            optimizer,
            quiet=quiet,
            name=name,
            **kwargs
        )

class VModule(RLNetwork):
    def __init__(self, model, normalize_inputs=False, box=None):
        super().__init__(model, normalize_inputs=normalize_inputs, box=box)

    def forward(self, states):
        return super().forward(states).squeeze(-1)
