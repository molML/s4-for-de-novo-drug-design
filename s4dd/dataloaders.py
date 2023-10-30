# %%
from typing import Dict, List, Tuple

import torch

from . import smiles_utils


class PaddedLabelEncodedDataset(torch.utils.data.Dataset):
    """A dataset that returns a tuple of `(X, y)` where `X` and `y` are both
    torch tensors. `X` is a sequence of integers representing the SMILES
    tokens, and `y` is the same sequence shifted by one position to the
    right.

    The outputs are padded to the same length and label encoded.
    """

    def __init__(
        self,
        label_encoded_molecules: List[List[int]],
        token2label: Dict[str, int],
    ):
        """Creates a `PaddedLabelEncodedDataset`.

        Parameters
        ----------
        label_encoded_molecules : List[List[int]]
            A list of label encoded and padded molecules, where each molecule is a list of
            integers representing the SMILES tokens. The integers are the labels of the
            tokens in the token2label dictionary. All molecules must be padded to the same
            length.
        token2label : Dict[str, int]
            A dictionary mapping SMILES tokens to integer labels.
        """
        self.label_encoded_molecules = label_encoded_molecules
        self.token2label = token2label

    def __len__(self) -> int:
        """Returns the number of molecules in the dataset.

        Returns
        -------
        int
            Number of molecules in the dataset.
        """
        return len(self.label_encoded_molecules)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of `(X, y)` where `X` and `y` are both torch tensors. `X` is a
        sequence of integers representing the SMILES tokens, and `y` is the same
        sequence shifted by one position to the right.

        Parameters
        ----------
        idx : int
            Index of the molecule to return.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of `(X, y)` where `X` and `y` are both torch tensors. `X` is a sequence of
            integers representing the SMILES tokens, and `y` is the same sequence shifted
            by one position to the right.
        """
        molecule = self.label_encoded_molecules[idx]
        X = torch.tensor(molecule[:-1])
        y = torch.tensor(molecule[1:])
        return X, y


def create_dataloader(
    path_to_data: str,
    batch_size: int,
    sequence_length: int = 100,
    num_workers: int = 8,
    shuffle: bool = True,
    token2label: Dict[str, int] = None,
) -> torch.utils.data.DataLoader:
    """Creates a dataloader for a dataset of SMILES strings. The input sequences will be
    tokenized, pre/appended with `"[BEG]`/`"[END]"` tokens, label encoded, and padded to the same length.

    Parameters
    ----------
    path_to_data : str
        Path to the dataset. Can be a zip file or a text file. The dataset must be a
        list of SMILES strings, one per line.
    batch_size : int
        Batch size.
    sequence_length : int, optional
        Number of tokens in the tokenized SMILES sequences. If a SMILES sequence has more tokens than this limit, it will be
        pre-truncated. If a sequence has less tokens than this, it will be post-padded with the value `"[PAD]"`.
        Note that the output sequences will be shifted by one position to the right,
        and the training sequence length will be `sequence_length - 1`.
        The default is 100.
    num_workers : int, optional
        Number of workers for the dataloader.
        The default is 8.
    shuffle : bool, optional
        Whether to shuffle the dataset.
        The default is True.
    token2label : Dict[str, int], optional
        A dictionary mapping SMILES tokens to integer labels. If `None`, the labels will
        be learned from the dataset, which is useful to create the train dataloader. The validation and test dataloaders
        should use the same `token2label` learned during the creation of the train dataloader.
        The default is `None`.

    Returns
    -------
    torch.utils.data.DataLoader
        A dataloader for the dataset.
    """
    if path_to_data.endswith(".zip"):
        import zipfile

        with open(path_to_data, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                fname = zf.namelist()[0]
                with zf.open(fname) as g:
                    dataset = g.read().decode("utf-8").splitlines()
    else:
        with open(path_to_data, "r") as f:
            dataset = f.read().splitlines()

    tokenized_dataset = [
        ["[BEG]"] + smiles_utils.segment_smiles(smiles) + ["[END]"]
        for smiles in dataset
    ]
    if token2label is None:
        token2label = smiles_utils.learn_label_encoding(tokenized_dataset)

    padded_dataset = smiles_utils.pad_sequences(
        tokenized_dataset, sequence_length, padding_value="[PAD]"
    )
    dataset = [[token2label[token] for token in tokens] for tokens in padded_dataset]

    return torch.utils.data.DataLoader(
        PaddedLabelEncodedDataset(
            dataset,
            token2label=token2label,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    train_loader = create_dataloader(
        "./datasets/chemblv31/train.zip",
        batch_size=16,
        sequence_length=100,
        num_workers=8,
        shuffle=True,
        token2label=None,
    )
    val_loader = create_dataloader(
        "./datasets/chemblv31/valid.zip",
        batch_size=16,
        sequence_length=100,
        num_workers=8,
        shuffle=False,
        token2label=train_loader.dataset.token2label,
    )

# %%
