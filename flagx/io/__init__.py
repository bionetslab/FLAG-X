
import warnings

# Always import FlowDataManager because it works without PyTorch
from .flowdatamanager import FlowDataManager
from .export import export_to_fcs

# Try to import torch-dependent components
try:
    from .flowdataset import FlowDataset
    from .flowdataloaders import FlowDataLoaders
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    FlowDataset = None
    FlowDataLoaders = None
    warnings.warn(
        "PyTorch is not installed. FlowDataset, FlowDataLoaders, and "
        "FlowDataManager.get_data_loader() are unavailable.\n"
        "All other IO functionality remains fully functional.\n"
        "To enable dataloaders, install PyTorch: https://pytorch.org/get-started/locally/",
        UserWarning
    )

__all__ = ['FlowDataManager', 'FlowDataset', 'FlowDataLoaders', 'export_to_fcs', 'TORCH_AVAILABLE']
