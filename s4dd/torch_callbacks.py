import json
import os
from abc import ABC
from typing import Callable, Dict, List

import numpy as np

_SAVE_FORMAT = "{basedir}/epoch-{epoch_ix:03d}"


class TorchCallback(ABC):
    """Base class for all Torch callbacks."""

    def __init__(self) -> None:
        """Creates a TorchCallback. Sets the `stop_training` flag to `False`, which would be common attribute of all callbacks."""
        super().__init__()
        self.stop_training = False

    def on_epoch_end(self, epoch_ix, history, **kwargs):
        """Called at the end of an epoch.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, List[float]]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        **kwargs
            Any additional keyword arguments.
        """
        pass

    def on_train_end(self, epoch_ix, history, **kwargs):
        """Called at the end of training.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, List[float]]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        **kwargs
            Any additional keyword arguments.
        """
        pass


class DenovoDesign(TorchCallback):
    """A callback for de novo design that designs SMILES strings in the end of every epoch."""

    def __init__(
        self,
        design_fn: Callable[[float], List[str]],
        basedir: str,
        temperatures: List[float],
    ) -> None:
        """Creates a `DenovoDesign` instance.

        Parameters
        ----------
        design_fn : Callable[[float], List[str]]
            A function that takes a temperature and returns a list of SMILES strings.
        basedir : str
            The base directory to save the generated molecules to.
        temperatures : List[float]
            A list of temperatures to use for sampling.
        """
        super().__init__()
        self.design_fn = design_fn
        self.basedir = basedir
        self.temperatures = temperatures

    def on_epoch_end(self, epoch_ix, **kwargs) -> None:
        """Designs and saves molecules in the end of every epoch with their log-likelihoods.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        """

        epoch_ix = epoch_ix + 1  # switch to 1-indexing
        print("Designing molecules. Epoch", epoch_ix)
        epoch_dir = _SAVE_FORMAT.format(basedir=self.basedir, epoch_ix=epoch_ix)
        os.makedirs(epoch_dir, exist_ok=True)
        for temperature in self.temperatures:
            molecules, log_likelihoods = self.design_fn(temperature)

            with open(
                f"{epoch_dir}/designed_chemicals-T_{temperature}.smiles", "w"
            ) as f:
                f.write("\n".join(molecules))

            np.savetxt(
                f"{epoch_dir}/designed_loglikelihoods-T_{temperature}.csv",
                log_likelihoods,
                delimiter=",",
            )


class EarlyStopping(TorchCallback):
    """A callback that stops training when a monitored metric has stopped improving."""

    def __init__(self, patience: int, delta: float, criterion: str, mode: str) -> None:
        """Creates an `EarlyStopping` callback.

        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement before stopping the training.
        delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        criterion : str
            The name of the metric to monitor.
        mode : str
            One of `"min"` or `"max"`. In `"min"` mode, training will stop when the quantity monitored has stopped decreasing;
            in `"max"` mode it will stop when the quantity monitored has stopped increasing.
        """
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.criterion = criterion
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch_ix: int, history: Dict[str, float], **kwargs) -> None:
        """Called at the end of an epoch. Updates the best metric value and the number of epochs waited for improvement.
        `stop_training` attribute is set to `True` if the training should be stopped.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, float]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        """
        monitor_values = history[self.criterion]
        self.wait += 1
        if len(monitor_values) < self.patience:
            return

        current = monitor_values[epoch_ix]
        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch_ix
            self.wait = 0
        elif self.wait >= self.patience:
            self.stop_training = True
            self.stopped_epoch = epoch_ix

    def _is_improvement(self, current):
        if self.mode == "min":
            return current < self.best - self.delta

        return current > self.best + self.delta


class ModelCheckpoint(TorchCallback):
    """A callback that saves the model in the end of every epoch."""

    def __init__(
        self,
        save_fn: Callable[[str], None],
        save_per_epoch: int,
        basedir: str,
    ) -> None:
        """Creates a `ModelCheckpoint` instance that runs per a fixed number of epoch and at the end of training.

        Parameters
        ----------
        save_fn : Callable[[str], None]
            A function that takes a directory and saves the model to that directory.
        save_per_epoch : int
            The number of epochs to wait between saves.
        basedir : str
            The base directory to save the model to.
        """
        super().__init__()
        self.save_fn = save_fn
        self.save_per_epoch = save_per_epoch
        self.basedir = basedir

    def _save(self, epoch_ix: int, **kwargs) -> None:
        savedir = os.path.join(self.basedir, f"epoch-{epoch_ix:03d}")
        os.makedirs(savedir, exist_ok=True)
        self.save_fn(savedir)

    def on_epoch_end(self, epoch_ix: int, **kwargs) -> None:
        """Saves the model in the end of every epoch.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        """

        epoch_ix = epoch_ix + 1  # 1-indexed
        if epoch_ix % self.save_per_epoch == 0:
            self._save(epoch_ix)

    def on_train_end(self, epoch_ix: int, **kwargs) -> None:
        """Saves the model in the end of training.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        """
        self._save(epoch_ix + 1)


class HistoryLogger(TorchCallback):
    """A callback that saves the training history in the end of every epoch."""

    def __init__(self, savedir: str) -> None:
        """Creates a `HistoryLogger` instance.

        Parameters
        ----------
        savedir : str
            The directory to save the training history to.
        """
        super().__init__()
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

    def on_epoch_end(self, history: Dict[str, List[float]], **kwargs) -> None:
        """Saves the training history in the end of every epoch.

        Parameters
        ----------
        history : Dict[str, List[float]]
            A dictionary containing the training history. The keys are the names of the metrics (`"val_loss"` and `"train_loss"`), and the values are lists of the metric values at each epoch.
        """
        with open(os.path.join(self.savedir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)
