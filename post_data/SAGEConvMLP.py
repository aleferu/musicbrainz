from typing import override, Union, List, Optional

from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import SAGEConv


class SAGEConvMLP(SAGEConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: List[int],
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        assert len(mid_channels) >= 1, f"Invalid length of mid_channels, expected >= 1, got {len(mid_channels)}"

        assert all(channel_size > 0 for channel_size in mid_channels), "Invalid channel size in mid_channels"

        super().__init__(
            in_channels,
            out_channels,
            aggr,
            normalize,
            root_weight,
            project,
            bias,
            **kwargs
        )

        self.mlp = nn.Sequential()
        self.__add_layer_to_mlp(0, in_channels, mid_channels[0])
        for i in range(1, len(mid_channels)):
            self.__add_layer_to_mlp(i, mid_channels[i - 1], mid_channels[i])
        self.__add_layer_to_mlp(len(mid_channels), mid_channels[-1], in_channels, bn=False, do=False)

    def __add_layer_to_mlp(self, index: int, in_channels: int, out_channels: int, bn: bool = True, do: bool = True):
        self.mlp.add_module(f"dense{index}", nn.Linear(in_channels, out_channels))
        if bn:
            self.mlp.add_module(f"bn{index}", nn.BatchNorm1d(out_channels))
        self.mlp.add_module(f"relu{index}", nn.ReLU())
        if do:
            self.mlp.add_module(f"do{index}", nn.Dropout())

    @override
    def message(self, x_j: Tensor) -> Tensor:
        return self.mlp(x_j)
