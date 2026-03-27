
import warnings
import re
import flowio
import numpy as np
import pandas as pd
import scanpy as sc

from typing import List, Tuple, Dict, Any, Union


def determine_filetype(filename: str) -> str:
    if filename.endswith('.fcs')  or filename.endswith('.FCS'):
        data_file_type = 'fcs'
    elif filename.endswith('.csv') or filename.endswith('.CSV'):
        data_file_type = 'csv'
    elif filename.endswith('.lmd') or filename.endswith('.LMD'):
        data_file_type = 'lmd'
    else:
        data_file_type = 'unknown'
    return data_file_type


def flowdata_to_anndata(flowdata: flowio.FlowData, reindex: bool = False, verbosity: int = 0) -> sc.AnnData:

    # Extract data matrix
    data_mat = np.reshape(flowdata.events, (-1, flowdata.channel_count)).astype(np.float32)

    # Extract meta data
    meta_data = flowdata.text

    # Convert metadata into dataframe
    meta_data_df = _flowdata_meta_data_dict_to_df(meta_data_dict=meta_data)

    # Extract spillover matrix if present
    spillover_df = None
    for key in ['spill', 'spillover']:
        try:
            spillover_str = meta_data[key]
            spillover_df = _spillover_mat_from_str(spillover_str=spillover_str)
        except KeyError:
            continue
    if spillover_df is None:
        if verbosity >= 1:
            warnings.warn('No spillover matrix found.')
    else:
        # By convention spillover matrix should be annotated with PnN
        spill_cols = spillover_df.columns.tolist()
        pnn = meta_data_df['PnN'].tolist()

        if not set(spill_cols).issubset(set(pnn)):

            if verbosity >= 1:
                warnings.warn('Spillover columns do not match PnN. Attempting to interpret as index-based encoding.')

            if all(str(c).isdigit() for c in spill_cols):
                idx_to_pnn = {
                    str(i): meta_data_df.loc[i, 'PnN']
                    for i in meta_data_df.index
                }
                missing = set(spill_cols) - set(idx_to_pnn.keys())
                if not missing:
                    spillover_df.rename(index=idx_to_pnn, columns=idx_to_pnn, inplace=True)
                else:
                    raise ValueError('Cannot align Spillover matrix and other metadata.')
            else:
                raise ValueError('Cannot align Spillover matrix and other metadata.')

    # Build the AnnData
    meta_data_df.index = meta_data_df['PnN']
    meta_data['spill'] = spillover_df
    adata = sc.AnnData(
        X=data_mat,
        var=meta_data_df,
        uns={'meta': meta_data, 'fcs_version': flowdata.version},
    )

    if reindex:

        if 'PnS' not in meta_data_df.columns:
            if verbosity >= 1:
                warnings.warn('PnS not found. Cannot reindex.')
        else:
            pnn = meta_data_df['PnN']
            pns = meta_data_df['PnS']

            new_index = pns.where(pns != '', pnn)

            if not new_index.is_unique:
                if verbosity >= 1:
                    warnings.warn('PnS not unique. Cannot reindex.')
            else:
                # Reindex var
                adata.var.index = new_index

                # Reindex spillover matrix
                if spillover_df is not None:
                    mapper = dict(zip(pnn, new_index))
                    spillover_df = spillover_df.rename(index=mapper, columns=mapper)
                    adata.uns['meta']['spill'] = spillover_df
    return adata


def _flowdata_meta_data_dict_to_df(meta_data_dict: Dict[str, Any]) -> pd.DataFrame:

    # Extract PnXYZ entries and store (n, value) tuples in dict
    channel_groups = dict()
    for key, val in meta_data_dict.items():

        match = re.match(r'^p(\d+)([a-z]+)$', key.lower())
        if not match:
            continue

        idx, suffix = int(match.group(1)), match.group(2)

        group_key = f'Pn{suffix.upper()}'

        channel_groups.setdefault(group_key, []).append((idx, val))

    # Raise error if PnN is missing
    if 'PnN' not in channel_groups:
        raise ValueError('PnN (channel names) missing')

    # Convert groups to dataframes and change dtype
    dfs = []
    for key, group in channel_groups.items():
        df = pd.DataFrame(group, columns=['n', key]).set_index('n')

        series = df[key]
        numeric = pd.to_numeric(series, errors='coerce')  # Conversion fails -> Nan
        if numeric.notna().all():  # Fully numeric -> distinguish int vs float
            if (numeric % 1 == 0).all():
                df[key] = numeric.astype(int)
            else:
                df[key] = numeric.astype(float)
        else:
            df[key] = series.astype(str)

        dfs.append(df)

    # Concatenate
    df_groups = pd.concat(dfs, axis=1)

    # Reorder
    df_groups.insert(0, 'PnN', df_groups.pop('PnN'))
    if 'PnS' in df_groups.columns:
        df_groups.insert(1, 'PnS', df_groups.pop('PnS'))

    # Replace NaN entries
    df_groups.fillna('', inplace=True)

    return df_groups


def _spillover_mat_from_str(spillover_str: str) -> pd.DataFrame:

    tokens = [t.strip().replace('\n', '') for t in spillover_str.split(',')]

    num_channels = int(tokens[0])

    expected_len = 1 + num_channels + num_channels * num_channels
    if len(tokens) != expected_len:
        raise ValueError('Malformed spillover string')

    channels = tokens[1:num_channels + 1]

    try:
        values = list(map(float, tokens[num_channels + 1:]))
    except ValueError:
        raise ValueError('Spillover matrix contains non-numeric values')


    so_mat = np.array(values).reshape((num_channels, num_channels))  # Reshape, order row mayor by default

    return pd.DataFrame(so_mat, columns=channels, index=channels)


def compensate(
        adata: sc.AnnData,
        spillover_key: str = 'spill',
        uncompensated_layer_key: Union[str, None] = None,
        inplace: bool = True,
) -> Union[sc.AnnData, None]:

    if not inplace:
        adata = adata.copy()

    # Retrieve spillover matrix
    so_mat_df = adata.uns['meta'][spillover_key]
    so_mat = so_mat_df.to_numpy()

    channels_so_mat = so_mat_df.columns.tolist()
    channels_data = adata.var.index.tolist()
    intersection = set(channels_so_mat) & set(channels_data)
    if not len(intersection) == len(channels_so_mat):
        raise ValueError('Index of the spillover matrix does not match data index.')

    # Extract the data matrix to be compensated
    data_mat_sub = np.asarray(adata[:, channels_so_mat].X)

    # Compute the compensated data matrix
    # S^T * X^T = Y^T with S = spillover mat, Y = measured data, X = "true" data; solve for X
    data_mat_sub_compensated = np.linalg.solve(so_mat.T, data_mat_sub.T).T

    # Store uncompensated data in extra layer
    if uncompensated_layer_key is not None:
        adata.layers[uncompensated_layer_key] = adata.X.copy()

    # Replace data with compensated data
    adata.X[:, [adata.var.index.get_loc(c) for c in channels_so_mat]] = data_mat_sub_compensated

    return adata if not inplace else None


def log10_w_cutoff(adata: sc.AnnData, cutoff: float = 100):
    """
    Apply a `log10` transform to values above a cutoff and clamp smaller values.

    Args:
        adata (AnnData): Input AnnData object. Transformation is applied inplace.
        cutoff (float): Minimum value for the transform. Values ≤ cutoff are set to `log10(cutoff)`.

    Returns:
        None
    """
    x = adata.X
    x = np.log10(x, out=np.full(x.shape, np.log(cutoff), dtype=float), where=(x > cutoff))
    adata.X = x


def log10_w_custom_cutoffs(
        adata: sc.AnnData,
        cutoffs: Dict[str, int],
):
    """
    Apply per-channel `log10` transforms using custom cutoffs.

    Args:
        adata (AnnData): Input AnnData object modified inplace.
        cutoffs (dict): Mapping ``{channel_name: cutoff}``. Values above the cutoff are `log10`-transformed; values below are set to `log10(cutoff)`.

    Returns:
        None
    """
    x = adata.X.copy()
    for channel, cutoff in cutoffs.items():
        col_idx = np.where(adata.var_names == channel)[0][0]
        x_col = adata.X[:, col_idx].copy()
        mask = (x_col > cutoff)
        x_col[mask] = np.log10(x_col[mask])
        x_col[~mask] = np.log10(cutoff)
        x[:, col_idx] = x_col

    adata.X = x


def get_downsampling_bool(
        num_events: int,
        target_num_events: int,
        stratification: Union[np.ndarray, None] = None,
) -> np.ndarray:

    if stratification is not None:
        strat_shape = stratification.shape[0]
        if num_events != strat_shape:
            raise ValueError(f'Number of events ({num_events}) does not match shape of stratification vector ({strat_shape}).')

    keep_mask = np.zeros(num_events, dtype=bool)

    if target_num_events >= num_events:
        keep_mask[:] = True

    elif stratification is not None:

        y = stratification

        unique_labels, counts = np.unique(y, return_counts=True)
        selected_indices = []

        for label, count in zip(unique_labels, counts):

            # Get the number of events of this class to keep
            target_num_events_class = int(round(target_num_events * (count / num_events)))
            target_num_events_class = min(target_num_events_class, count)

            # Get the indices where y == class
            class_indices = np.where(y == label)[0]

            # Randomly draw from the indices and append to list
            if target_num_events_class > 0:
                selected = np.random.choice(class_indices, target_num_events_class, replace=False)
                selected_indices.extend(selected)

        # Adjust the number of events to the exact desired number (account for rounding errors)
        if len(selected_indices) > target_num_events:
            selected_indices = np.random.choice(selected_indices, target_num_events, replace=False)
        elif len(selected_indices) < target_num_events:
            remaining_unselected_events = np.setdiff1d(np.arange(num_events), selected_indices)
            additional_events = np.random.choice(
                remaining_unselected_events, target_num_events - len(selected_indices), replace=False
            )
            selected_indices.extend(additional_events)

        # Set mask to True for selected events
        keep_mask[selected_indices] = True

    else:

        # Select events by drawing uniform at random without replacement
        selected_indices = np.random.choice(np.arange(num_events), target_num_events, replace=False)

        # Set mask to True for selected events
        keep_mask[selected_indices] = True

    return keep_mask


def get_numpy_data_matrix(
        data_list: List[sc.AnnData],
        channels: Union[List[int], List[str]],
        layer_key: Union[str, None] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    # Get data matrix from each anndata in data_list
    arrays = []
    for adata in data_list:

        if layer_key is None:
            arrays.append(adata[:, channels].X.copy())
        else:
            arrays.append(adata[:, channels].layers[layer_key].copy())

    # Concatenate to single data matrix
    array = np.concatenate(arrays, axis=0)

    return array


def get_numpy_label_vector(
        data_list: List[sc.AnnData],
        label_key: Union[int, str, None],
        layer_key: Union[str, None] = None,
        verbosity: int = 1,
) -> np.ndarray:

    # Get labels from each anndata in data_list
    labels = []
    for adata in data_list:

        labels.append(
            get_labels(
                adata=adata,
                label_key=label_key,
                layer_key=layer_key,
                verbosity=verbosity,
            )
        )

    label_array = np.concatenate(labels, axis=0)

    return label_array.astype(int)


def get_labels(
        adata: sc.AnnData,
        label_key: Union[str, int],
        layer_key: Union[str, None] = None,
        verbosity: int = 1,
) -> np.ndarray:

    # Label key is index of data matrix
    if isinstance(label_key, int):
        try:
            if layer_key is None:
                labels = adata.X[:, label_key].copy()
            else:
                labels = adata.layers[layer_key][:, label_key].copy()
        except IndexError:
            raise ValueError("'label_key' index is out of bounds in .X/.layers[layer_key] matrix")

    # Label key is var name or obs key
    else:
        try:
            label_idx = adata.var_names.get_loc(label_key)
            if layer_key is None:
                labels = adata.X[:, label_idx].copy()
            else:
                labels = adata.layers[layer_key][:, label_idx].copy()
        except KeyError:
            if verbosity >= 1:
                warnings.warn(f"'label_key' not found in .var_names, trying .obs")
            try:
                labels = adata.obs[label_key].to_numpy().copy()
            except KeyError:
                raise ValueError("'label_key' not found in .obs or .var_names")
    return labels.astype(int)



