
import os
import warnings
import copy
import numpy as np

from typing import Tuple, List, Dict, Union, Any

from typing_extensions import Self
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from .fcnn_model import FCNNModel
except Exception as e:
    raise ImportError(
        "PyTorch is required for MLPClassifier but is not installed.\n"
        "Install according to your system's requirements (see: https://pytorch.org/get-started/locally/)."
    )


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A three layer perceptron (MLP) classifier.

    This classifier wraps a fully connected neural network implemented in PyTorch while exposing a scikit-learn–style API.
    The model supports multi-class classification, automatic device selection (CPU or GPU),
    and provides the methods ``fit()``, ``predict()``, ``predict_proba()``, ``score()``, ``save()``, and ``load()``.
    For model training CrossEntropyLoss and the Adam optimizer are used.

    Attributes:
        classes_ (np.ndarray or None): Class labels after re-indexing to integers starting from 0.
        class_counts_ (np.ndarray or None): Class counts from the training data.
        og_classes_ (np.ndarray or None): Original class labels before re-indexing.
        class_priors_ (np.ndarray or None): Empirical class priors.
        new_to_og_classes_dict_ (dict[int, Any] or None): Mapping from new integer labels back to original labels.
        data_set_ (TensorDataset or None): PyTorch tensor dataset constructed during fitting.
        data_loader_ (DataLoader or None): PyTorch DataLoader used for minibatch training.
        model_ (nn.Module or None): Neural network model.
        criterion_ (nn.Module or None): Loss function, PyTorch CrossEntropyLoss.
        optimizer_ (Optimizer or None): PyTorch Adam optimizer with learning rate 0.001.
        training_log_ (dict[str, int | list[int | float]] or None): Logged losses of the training run.
        is_fitted_ (bool): Whether the classifier has been fitted.
    """
    def __init__(
            self,
            layer_sizes: Tuple[int, ...] = (128, 64, 32),
            n_epochs: int = 20,
            loss_params: Union[Dict[str, Any], None] = None,
            optimizer_params: Union[Dict[str, Any], None] = None,
            data_loader_params: Union[Dict[str, Any], None] = None,
            validation_fraction: float = 0.1,
            early_stopping: bool = False,
            tol: float = 1e-4,
            n_iter_no_change: int = 5,
            device: Union[str, None] = None,  # If None use gpu if available, else cpu
            verbosity: int = 1,
    ):
        """
        Args:
            layer_sizes (Tuple[int, ...]): Sizes of the hidden layers in the fully connected neural network.
            n_epochs (int): Number of training epochs.
            loss_params (dict[str, Any] or None): Parameters passed to the PyTorch's CrossEntropyLoss.
            optimizer_params (dict[str, Any] or None): Parameters passed to the PyTorch's Adam optimizer. If None, defaults to ``{'lr': 0.001}``.
            data_loader_params (dict[str, Any] or None): Parameters passed to the PyTorch DataLoader. If None, defaults to ``{'batch_size': 128, 'shuffle': True, 'num_workers': 1}``.
            validation_fraction (float): Fraction of the training data used as validation set. Defaults to 0.1.
            early_stopping (bool): Whether early stopping is used or not. If `early_stopping` is `True` and `validation_fraction=0.0`, the training loss is used as an early stopping criterion. Defaults to False.
            tol (float): Tolerance for early stopping. When the validation/training loss is not improving by at least `tol` for `n_iter_no_change` consecutive iterations, training is stopped early.
            n_iter_no_change (int): Maximum number of epochs to not meet `tol` improvement.
            device (str or None): Device to use for training (e.g., ``'cpu'``, ``'cuda'``, ``'cuda:0'``). If None, CUDA is used when available, otherwise falls back to CPU.
            verbosity (int): Verbosity level for training logs.
        """

        if validation_fraction < 0 or validation_fraction > 1:
            raise ValueError(f'`validation_fraction` must be in [0, 1], got {validation_fraction}.')

        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_epochs = n_epochs
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params
        self.data_loader_params = data_loader_params
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.verbosity = verbosity

        # Set device to default cuda device if available, else cpu
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)

        self.classes_ = None
        self.class_counts_ = None
        self.og_classes_ = None
        self.class_priors_ = None
        self.new_to_og_classes_dict_ = None
        self.data_set_ = None
        self.data_loader_ = None
        self.model_ = None
        self.criterion_ = None
        self.optimizer_ = None
        self.training_log_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Self:
        """
        Fit the MLP classifier to the provided training data.

        Args:
            X (np.ndarray): Feature matrix of shape `(n_samples, n_features)`.
            y (np.ndarray): Target labels of shape `(n_samples,)`.

        Returns:
            Self: The fitted classifier instance.

        Raises:
            ValueError: If X and y have incompatible shapes.
        """

        # ### Data processing and preparation
        # Check input data format
        X, y = check_X_y(X, y)

        # Rename classes to integers starting from 0 and extract label information
        (
            y, self.classes_, self.class_counts_, self.class_priors_, self.new_to_og_classes_dict_, self.og_classes_
        ) = MLPClassifier._process_class_labels(y=y)

        if self.loss_params is None:
            self.loss_params = dict()

        if self.optimizer_params is None:
            self.optimizer_params = {'lr': 0.001}
            warnings.warn('No optimizer_params provided, using lr=0.001')

        # Get tensor dataset and data loader
        if self.data_loader_params is None:
            self.data_loader_params = {'batch_size': 128, 'shuffle': True, 'num_workers': 1}
            warnings.warn('No data_loader_params provided, using batch_size=128, shuffle=True, num_workers=1')


        if self.early_stopping and self.validation_fraction <= 0.0:
            warnings.warn(
                f'`early_stopping` is `True` but `validation_fraction` is {self.validation_fraction}. '
                f'Using training loss for early stopping.'
            )

        # Split data into training and validation set
        if self.validation_fraction > 0.0:
            x_train, x_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                stratify=y if np.unique(y).shape[0] > 1 else None,
                random_state=42,
            )
        else:
            x_train, y_train = X, y
            x_val, y_val = None, None

        # Create data loaders
        self.data_set_, self.data_loader_ = MLPClassifier._get_data_set_data_loader(
            X=x_train,
            y=y_train,
            data_loader_params=self.data_loader_params
        )
        if self.validation_fraction > 0.0:
            data_loader_params_val = copy.deepcopy(self.data_loader_params)
            data_loader_params_val.update({'shuffle': False, 'num_workers': 1})
            self.val_set_, self.val_loader_ = self._get_data_set_data_loader(
                X=x_val,
                y=y_val,
                data_loader_params=data_loader_params_val
            )
        else:
            self.val_set_, self.val_loader_ = None, None

        # ### Model training
        # Instantiate the softmax classifier
        self.model_ = FCNNModel(in_size=X.shape[1], out_size=self.classes_.shape[0], layer_sizes=self.layer_sizes)

        # Instantiate the loss function
        self.criterion_ = nn.CrossEntropyLoss(**self.loss_params)

        # Instantiate the optimizer
        self.optimizer_ = optim.Adam(self.model_.parameters(), **self.optimizer_params)

        # Train the model
        self.training_log_ = self._train_loop(
            model=self.model_,
            data_loader=self.data_loader_,
            criterion=self.criterion_,
            optimizer=self.optimizer_,
            n_epochs=self.n_epochs,
            device=self.device,
            val_loader=self.val_loader_,
            early_stopping=self.early_stopping,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            verbosity=self.verbosity,
        )

        self.is_fitted_ = True

        return self

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels for the given input samples.

        Args:
            X (np.ndarray): Feature matrix of shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Predicted class labels using the original label encoding.

        Raises:
           NotFittedError: If ``predict()`` is used before calling ``fit()``.
        """

        # Get softmax prediction
        y_proba = self.predict_proba(X)

        # Get class with max probability
        y_pred = y_proba.argmax(axis=1)

        # Get vector with the original labels
        y_pred = np.array([self.new_to_og_classes_dict_[key] for key in y_pred])

        return y_pred

    def predict_proba(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
       Predict class probabilities for the given samples.

       Args:
           X (np.ndarray): Feature matrix of shape `(n_samples, n_features)`.

       Returns:
           np.ndarray: Array of shape `(n_samples, n_classes)` containing class probabilities.

       Raises:
           NotFittedError: If ``predict()`` is used before calling ``fit()``.
       """

        # Check whether fit was already called
        check_is_fitted(self, 'is_fitted_')

        # Check input data format
        X = check_array(X)

        # Cast numpy array into torch Tensor
        x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)

        # Ensure tensor is on same device as model
        device = next(self.model_.parameters()).device
        x_tensor = x_tensor.to(device)

        # Set model to eval mode and disable gradient tracking, forward pass,
        self.model_.eval()
        with torch.no_grad():
            model_out = self.model_(x_tensor)
            y_proba = torch.softmax(model_out, dim=1).cpu().numpy()

        return y_proba

    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Union[np.ndarray, None] = None,
    ):
        """
        Compute the macro F1 score of the classifier on the given dataset.

        Args:
            X (np.ndarray): Feature matrix of shape `(n_samples, n_features)`.
            y (np.ndarray): True labels.
            sample_weight (np.ndarray or None): Optional sample weights.

        Returns:
            float: Macro-averaged F1 score.

        Raises:
           NotFittedError: If ``score()`` is used before calling ``fit()``.
        """

        y_pred = self.predict(X)

        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    @staticmethod
    def _train_loop(
            model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            n_epochs: int,
            device: torch.device,
            val_loader: Union[DataLoader, None],
            early_stopping: bool,
            tol: float,
            n_iter_no_change: int,
            verbosity: int = 1,
    ) -> Dict[str, Union[List[Union[float, int]], int]]:

        # Adopted from:
        # (https://github.com/lijcheng12/DGCyTOF/blob/main/DGCyTOF_Package/DGCyTOF/__init__.py)

        # Get number of events in train set
        n_events = len(data_loader.dataset)
        if val_loader is not None:
            n_events_val = len(val_loader.dataset)
        else:
            n_events_val = None

        # Move model to device
        model.to(device=device)

        losses_train = []
        n_corrects_train = []
        losses_val = []
        n_corrects_val = []
        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(n_epochs):

            # Set model to train mode
            model.train()

            loss_train = 0.0
            n_correct_train = 0
            for batch in data_loader:

                # Extract data from current batch
                x, y = batch

                # Move data to device
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                model_out = model(x)

                # Calculate loss
                loss = criterion(model_out, y)

                # Clear old gradients
                optimizer.zero_grad()

                # Backward pass, gradient computation
                loss.backward()

                # Update parameters
                optimizer.step()

                # Track loss and number of correct predictions
                loss_train += loss.item() * x.size(0)
                n_correct_train += MLPClassifier._get_num_correct(model_out, y)

            loss_train /= n_events
            losses_train.append(loss_train)
            n_corrects_train.append(n_correct_train)

            if val_loader is not None:
                model.eval()
                loss_val = 0.0
                n_correct_val = 0

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        out = model(x_val)
                        loss = criterion(out, y_val)
                        loss_val += loss.item() * x_val.size(0)
                        n_correct_val += MLPClassifier._get_num_correct(out, y_val)

                loss_val /= n_events_val
                current_loss = loss_val

                losses_val.append(loss_val)
                n_corrects_val.append(n_correct_val)

            else:
                current_loss = loss_train

            if verbosity >= 2:
                print(f'# --- Epoch {epoch + 1}/{n_epochs} --- #')
                if val_loader is not None:
                    print(f'# Training loss: {loss_train:.4f}, Validation loss: {loss_val:.4f}')
                    print(f'# Fraction of correct predictions: Train:{n_correct_train / n_events:.4f}, Val: {n_correct_val / n_events_val:.4f}')
                else:
                    print(f'# Training loss: {loss_train:.4f}')
                    print(
                        f'# Fraction of correct predictions: {n_correct_train / n_events:.4f}')

            if early_stopping:

                if best_loss - current_loss > tol:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= n_iter_no_change:
                    if verbosity >= 2:
                        print(f'# --- Early stopping at epoch {epoch + 1} --- #')
                    break

        if early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Move model to cpu
        model.cpu()

        # Set model to evaluation mode
        model.eval()

        training_log = {
            'training_loss': losses_train,
            'num_correct_train': n_corrects_train,
            'num_events_train': n_events
        }
        if val_loader is not None:
            training_log['validation_loss'] = losses_val
            training_log['num_correct_val'] = n_corrects_val
            training_log['num_events_val'] = n_events_val

        return training_log

    def save(
            self,
            filename: str = 'mlp_classifier.pkl',
            filepath: Union[str, None] = None,
    ) -> None:
        """
        Save the fitted classifier to disk using ``torch.save``.

        Args:
            filename (str): Name of the file to save the model to.
            filepath (str or None): Directory where the file will be saved. Defaults to current working directory.

        Returns:
            None
        """

        if filepath is None:
            filepath = os.getcwd()

        torch.save(self, os.path.join(filepath, filename))

    @classmethod
    def load(
            cls,
            filename: str = 'mlp_classifier.pkl',
            filepath: Union[str, None] = None,
            map_location: Union[str, torch.device] = 'cpu',
    ) -> Self:
        """
        Load a previously saved classifier from disk.

        Args:
            filename (str): Name of the saved file.
            filepath (str or None): Directory containing the saved file. Defaults to current working directory.
            map_location (str or torch.device): Device mapping for loading the model (e.g., ``'cpu'`` or ``'cuda'``).

        Returns:
            Self: The loaded classifier instance.
        """

        if filepath is None:
            filepath = os.getcwd()

        return torch.load(os.path.join(filepath, filename), map_location=map_location, weights_only=False)

    @staticmethod
    def _process_class_labels(
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Any, float], np.ndarray]:

        og_classes, counts = np.unique(y, return_counts=True)

        class_priors = counts / counts.sum()
        new_classes = np.array(list(range(og_classes.shape[0])))
        og_to_new_classes_dict = {key: value for key, value in zip(og_classes, new_classes)}
        y_new = np.vectorize(og_to_new_classes_dict.get)(y)

        new_to_og_classes_dict = {key: value for key, value in zip(new_classes, og_classes)}
        new_to_og_classes_dict[-1] = -1  # Predict label -1 for unclassifiable events

        return y_new, new_classes, counts, class_priors, new_to_og_classes_dict, og_classes

    @staticmethod
    def _get_data_set_data_loader(
            X: np.ndarray,
            y: np.ndarray,
            data_loader_params: Dict[str, Any],
    ) -> Tuple[torch.utils.data.TensorDataset, DataLoader]:

        # Turn input data into tensors and create corresponding torch dataset
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)

        data_loader = DataLoader(dataset=dataset, **data_loader_params)

        return dataset, data_loader

    @staticmethod
    def _get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()



