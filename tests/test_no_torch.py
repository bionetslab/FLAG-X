
import pytest
import importlib
from pathlib import Path
from unittest.mock import patch
import sys

TEST_DATA_DIR = Path(__file__).parent / 'test_data'

def test_import_without_torch():
    with patch.dict("sys.modules", {
        "torch": None,
        "torch.nn": None,
        "torch.optim": None,
        "torch.utils": None,
        "torch.utils.data": None,
    }):

        # Remove pre-imported flagx modules
        for mod in list(sys.modules):
            if mod.startswith("flagx"):
                del sys.modules[mod]

        gating = importlib.import_module("flagx.gating")
        io = importlib.import_module("flagx.io")

        assert gating.MLPClassifier is None
        assert gating.FCNNModel is None
        assert gating.TORCH_AVAILABLE is False

        assert io.FlowDataset is None
        assert io.FlowDataLoaders is None
        assert io.TORCH_AVAILABLE is False

def test_fdm_get_data_loader_without_torch():

    with patch.dict("sys.modules", {
        "torch": None,
        "torch.nn": None,
        "torch.optim": None,
        "torch.utils": None,
        "torch.utils.data": None,
    }):

        # Remove pre-imported flagx modules
        for mod in list(sys.modules):
            if mod.startswith("flagx"):
                del sys.modules[mod]

        from flagx.io import FlowDataManager

        fdm = FlowDataManager(
            data_file_names=[f'Case_{i}.fcs' for i in range (1, 6)],
            data_file_path=str(TEST_DATA_DIR)
        )
        fdm.load_data_files_to_anndata()

        with pytest.raises(ImportError):
            fdm.get_data_loader(
                data_set="all",
                channels=None,
                label_key=None
            )

def test_pipeline_mlp_raises_without_torch(tmp_path):
    with patch.dict("sys.modules", {
        "torch": None,
        "torch.nn": None,
        "torch.optim": None,
        "torch.utils": None,
        "torch.utils.data": None,
    }):

        # Remove pre-imported flagx modules
        for mod in list(sys.modules):
            if mod.startswith("flagx"):
                del sys.modules[mod]

        from flagx.pipeline import GatingPipeline

        # Use your real test data
        # MLP should IMMEDIATELY fail because torch is unavailable
        pipe = GatingPipeline(
            train_data_file_path=str(TEST_DATA_DIR),
            train_data_file_names=[f'Case_{i}.fcs' for i in range (1, 6)],
            channels=['FS INT', 'SS INT'],
            label_key='label',
            gating_method='mlp',
            gating_method_kwargs={'n_epochs': 2},
            save_path=str(tmp_path),
        )

        with pytest.raises(ImportError):
            pipe.train()

def test_pipeline_som_works_without_torch(tmp_path):
    with patch.dict("sys.modules", {
        "torch": None,
        "torch.nn": None,
        "torch.optim": None,
        "torch.utils": None,
        "torch.utils.data": None,
    }):

        # Remove pre-imported flagx modules
        for mod in list(sys.modules):
            if mod.startswith("flagx"):
                del sys.modules[mod]

        from flagx.pipeline import GatingPipeline

        pipe = GatingPipeline(
            train_data_file_path=str(TEST_DATA_DIR),
            train_data_file_names=[f'Case_{i}.csv' for i in range (1, 6)],
            channels=['FS INT', 'SS INT'],
            label_key='label',
            gating_method='som',
            gating_method_kwargs={'som_dimensions': (2, 2), 'n_epochs': 3},
            save_path=str(tmp_path),
        )

        # SOM training must NOT raise
        pipe.train()

        # Inference should also work
        outdir = tmp_path / "som_no_torch"
        outdir.mkdir()

        pipe.inference(
            data_file_path=str(TEST_DATA_DIR),
            data_file_names=['Case_1.csv', 'Case_2.csv', 'Case_3.csv'],
            gate=True,
            dim_red_methods=('pca',),
            save_path=str(outdir),
        )

        assert (outdir / "dimred.fcs").exists() or \
               (outdir / "annotated_data.fcs").exists()

