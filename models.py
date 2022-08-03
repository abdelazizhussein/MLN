import torch
from monotone.group import GroupSort
from monotone.functional import direct_norm


class UnconstrainedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 1),
)
    def forward(self, x):
        return self.model(x)

# Build a Lipschitz-1 network
class RobustModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.model = torch.nn.Sequential(
        self.l1 = direct_norm(torch.nn.Linear(5, 16))
        GroupSort(2)
        self.l2 = direct_norm(torch.nn.Linear(16, 16))
        GroupSort(2)
        self.l3 = direct_norm(torch.nn.Linear(16, 16))
        GroupSort(2)
        self.l4 = direct_norm(torch.nn.Linear(16, 16))
        GroupSort(2)
        self.l5 = direct_norm(torch.nn.Linear(16, 16))
        GroupSort(2)
        self.l6 = direct_norm(torch.nn.Linear(16, 1))
            #torch.nn.Sigmoid(),
        #)

    def forward(self, x):
        x = self.l1(x)
        #print("Layer1:",x)
        g= GroupSort(2)
        x = g.forward(x)
        #print("Layer1_sorted:",x)

        x = self.l2(x)
        #print("Layer2:",x)
        g= GroupSort(2)
        x = g.forward(x)
        #print("Layer2_sorted:",x)

        x = self.l3(x)
        #print("Layer3:",x)
        g= GroupSort(2)
        x = g.forward(x)
        #print("Layer3_sorted:",x)

        x = self.l4(x)
        #print("Layer4:",x)
        g= GroupSort(2)
        x = g.forward(x)
        #print("Layer4_sorted:",x)

        x = self.l5(x)
        #print("Layer5:",x)
        g= GroupSort(2)
        x = g.forward(x)
        #print("Layer5_sorted:",x)

        x = self.l6(x)
        #print("Layer6:", x)
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
