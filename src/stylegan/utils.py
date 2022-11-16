import torch
from torch import nn


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module) -> None:
    """Copies the parameters of the online network to the momentum network.
    Parameters
    ----------
    online_net : nn.Module
        The online network.
    momentum_net : nn.Module
        The momentum network.
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, tau: float = 0.996) -> None:
        """Updates momentum parameters using exponential moving average.

        Parameters
        ----------
        tau : float, optional
            base value of the weight decrease coefficient
            (should be in [0,1]). Defaults to 0.996.
        """

        super().__init__()
        self.cur_tau = tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module) -> None:
        """Updates the momentum network parameters using exponential moving average.

        Parameters
        ----------
        online_net : nn.Module
            The online network.
        momentum_net : nn.Module
            The momentum network.
        """
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data
