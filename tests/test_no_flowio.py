

import pytest
import importlib
from unittest.mock import patch
import sys
import numpy as np
import scanpy as sc


def test_export_to_fcs_safeguard_without_flowio(tmp_path):

    # Simulate import failure for flowio
    with patch.dict("sys.modules", {"flowio": None, "flowutils": None}):

        # Remove pre-imported flagx modules
        for mod in list(sys.modules):
            if mod.startswith("flagx"):
                del sys.modules[mod]

        # Import your module fresh
        flagx = importlib.import_module("flagx")

        assert flagx.io.FLOWIO_AVAILABLE is False

        # Create minimal AnnData object with 5 events × 3 features
        X = np.random.rand(5, 3)
        adata = sc.AnnData(X)
        adata.var_names = ["A", "B", "C"]
        adata.uns["filename"] = "testsample.fcs"

        save_path = tmp_path
        save_name = "output.fcs"

        # Run export
        with pytest.warns(UserWarning, match="FlowIO is required"):
            flagx.io.export_to_fcs(
                data_list=[adata],
                save_path=str(save_path),
                save_filenames=[save_name],
                sample_wise=True
            )

        # Check that NO FCS was written,
        # but a CSV file exists instead
        assert not (save_path / save_name).exists()

        csv_name = save_name.replace(".fcs", ".csv")
        csv_file = save_path / csv_name
        assert csv_file.exists(), f"CSV fallback not created: {csv_file}"
