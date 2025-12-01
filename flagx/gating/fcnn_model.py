
from typing import Tuple

try:
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False


class FCNNModel(nn.Module):
    """
    Three layer fully connected neural network (FCNN).

    This model uses three hidden linear layers with ReLU activations. Softmax is intentionally omitted from the final layer because ``torch.nn.CrossEntropyLoss`` expects raw logits.

    The architecture follows the configuration described in:

    DGCyTOF: Deep learning with graphic cluster visualization to predict cell types of single cell mass cytometry data (Cheng et al., 2022).

    Their implementation can be found at:

    https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb (22/27/2025).

    Args:
        in_size (int): Number of input features.
        out_size (int): Number of output classes.
        layer_sizes (Tuple[int, int, int], optional): Sizes of the three hidden layers. Defaults to `(128, 64, 32)`.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output layer producing class logits.

    Returns:
        torch.Tensor: Raw output logits with shape `(batch_size, out_size)`.
    """

    # Dimensions chosen as described in paper and
    # https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb
    def __init__(self, in_size, out_size, layer_sizes: Tuple[int, int, int] = (128, 64, 32)):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for FCNNModel but is not installed.\n"
                "Install according to your system's requirements (see: https://pytorch.org/get-started/locally/)."
            )
        super(FCNNModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], out_size, bias=True)
        # self.softmax = nn.Softmax(dim=1)  # Softmax for the output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.softmax(x)  # Softmax activation for output
        # Do not apply softmax, CrossEntropyLoss does so internally
        return x
