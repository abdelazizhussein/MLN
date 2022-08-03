import torch
from monotone.group import GroupSort
from monotone.functional import direct_norm


class UnconstrainedModel(torch.nn.Module):
    def __init__(self,n_features, n_classes, n_nodes):
        super().__init__()        
        self.layers = []
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_nodes = n_nodes
        self._build_layers()

    def _build_layers(self):
        self.layers.append(direct_norm(torch.nn.Linear(self.n_features, self.n_nodes[0])))
        self.layers.append(torch.nn.LeakyReLU())
        last_nodes = self.n_nodes[0]
        for i_n_nodes in self.n_nodes[1:]:
            self.layers.append(direct_norm(torch.nn.Linear(last_nodes, i_n_nodes)))
            self.layers.append(torch.nn.LeakyReLU())
            last_nodes = i_n_nodes
        self.layers.append(direct_norm(torch.nn.Linear(last_nodes, self.n_classes)))
        for i, layer in enumerate(self.layers):
            setattr(self, "layer_%d" % i, layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Build a Lipschitz-1 network
class RobustModel(torch.nn.Module):
    def __init__(self,n_features, n_classes, n_nodes,group_size):
        super().__init__()
        self.layers = []
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.group_size = group_size
        self._build_layers()

    def _build_layers(self):
        self.layers.append(direct_norm(torch.nn.Linear(self.n_features, self.n_nodes[0])))
        self.layers.append(GroupSort(self.group_size))
        last_nodes = self.n_nodes[0]
        for i_n_nodes in self.n_nodes[1:]:
            self.layers.append(direct_norm(torch.nn.Linear(last_nodes, i_n_nodes)))
            print(self.group_size)
            self.layers.append(GroupSort(self.group_size))
            last_nodes = i_n_nodes
        self.layers.append(direct_norm(torch.nn.Linear(last_nodes, self.n_classes)))
        for i, layer in enumerate(self.layers):
            setattr(self, "layer_%d" % i, layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


import typing as T
class SigmaNet(torch.nn.Module):
    def __init__(
        self,
        nn: torch.nn.Module,  # Must already be sigma lipschitz
        sigma: float,
        monotone_constraints: T.Optional[T.Iterable] = None,
    ):
        """ Implementation of a monotone network with a sigma lipschitz constraint.

        Args:
            nn (torch.nn.Module): Lipschitz-constrained network with Lipschitz
                constant sigma.
            monotone_constraints (T.Optional[T.Iterable], optional): Iterable of the
                monotonic features. For example, if a network
                which takes a vector of size 3 is meant to be monotonic in the last
                feature only, then monotone_constraints should be [0, 0, 1].
                Defaults to all features (i.e. a vector of ones everywhere).
                """
        super().__init__()
        self.nn = nn
        self.register_buffer("sigma", torch.Tensor([sigma]))
        self.monotone_constraint = monotone_constraints or [1]
        self.monotone_constraint = torch.tensor(self.monotone_constraint).float()

    def forward(self, x: torch.Tensor):
        return self.nn(x) + self.sigma * (
            x * self.monotone_constraint.to(x.device)
        ).sum(axis=-1, keepdim=True)
