
import torch
from typing import Tuple

try:
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError(
        "PyTorch is required for FCNNModel but is not installed.\n"
        "Install according to your system's requirements (see: https://pytorch.org/get-started/locally/)."
    )


class FCNNModel(nn.Module):
    """
    Fully connected neural network with arbitrary number of hidden linear layers of arbitrary size.

    All but the output layer uses ReLU activations. Softmax is intentionally omitted from the final layer because ``torch.nn.CrossEntropyLoss`` expects raw logits.

    The default parameters follow the configuration described in:

    DGCyTOF: Deep learning with graphic cluster visualization to predict cell types of single cell mass cytometry data (Cheng et al., 2022).

    Their implementation can be found at:

    https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb (22/27/2025).

    Args:
        in_size (int): Number of input features.
        out_size (int): Number of output classes.
        layer_sizes (Tuple[int, ...], optional): Sizes of the hidden layers. Defaults to `(128, 64, 32)`.

    Attributes:
        layers (nn.ModuleList): List of fully connected linear layers.
    """

    # Dimensions chosen as described in paper and
    # https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb
    def __init__(self, in_size, out_size, layer_sizes: Tuple[int, ...] = (128, 64, 32)):
        super(FCNNModel, self).__init__()

        sizes = (in_size, ) + layer_sizes + (out_size,)

        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCNN model. ReLU activation is applied after each layer except after the output layer.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Raw output logits with shape `(batch_size, out_size)`.

        """

        # Apply ReLU to all but last layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # No activation in final layer; Do not apply softmax, CrossEntropyLoss does so internally
        x = self.layers[-1](x)

        return x
