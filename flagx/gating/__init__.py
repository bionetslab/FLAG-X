
import warnings

from .som_classifier import SomClassifier

try:
    from .mlp_classifier import MLPClassifier
    from .fcnn_model import FCNNModel
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not installed — MLPClassifier and underlying FCNNModel are unavailable.\n"
        "Install according to your system's requirements (see: https://pytorch.org/get-started/locally/).",
        UserWarning
    )
    MLPClassifier = None
    FCNNModel = None

__all__ = ['SomClassifier', 'MLPClassifier', 'FCNNModel', 'TORCH_AVAILABLE']


