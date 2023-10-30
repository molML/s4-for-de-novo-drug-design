# %%
import re
from typing import Dict, List, Union

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def learn_label_encoding(tokenized_inputs: List[List[str]]) -> Dict[str, int]:
    """Learn a label encoding from a tokenized dataset. The padding token, `"[PAD]"` is always assigned the label 0.

    Parameters
    ----------
    tokenized_inputs : List[List[str]]
        SMILES of the molecules in the dataset, tokenized into a list of tokens.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping SMILES tokens to integer labels.
    """
    token2label = dict()
    token2label["[PAD]"] = len(token2label)
    for inp in tokenized_inputs:
        for token in inp:
            if token not in token2label:
                token2label[token] = len(token2label)

    return token2label


def pad_sequences(
    sequences: List[List[Union[str, int]]],
    padding_length: int,
    padding_value: Union[str, int],
) -> List[List[Union[str, int]]]:
    """Pad sequences to a given length. The padding is done at the end of the sequences.
    Longer sequences are truncated from the beginning.

    Parameters
    ----------
    sequences : List[List[Union[str, int]]
        A list of sequences, either tokenized or label encoded SMILES.
    padding_length : int
        The length to pad the sequences to.
    padding_value : Union[str, int]
        The value to pad the sequences with.

    Returns
    -------
    List[List[Union[str, int]]
        The padded sequences.
    """
    lens = [len(seq) for seq in sequences]
    diffs = [max(padding_length - len, 0) for len in lens]
    padded_sequences = [
        seq + [padding_value] * diff for seq, diff in zip(sequences, diffs)
    ]
    truncated_sequences = [seq[-padding_length:] for seq in padded_sequences]

    return truncated_sequences


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    """Segment a SMILES string into tokens.

    Parameters
    ----------
    smiles : str
        A SMILES string.
    segment_sq_brackets : bool
        Whether to segment the square brackets `"["` and `"]"` as tokens.
        The default is `True`.

    Returns
    -------
    List[str]
        A list of tokens.
    """
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)


def segment_smiles_batch(
    smiles_batch: List[str], segment_sq_brackets=True
) -> List[List[str]]:
    """Segment a batch of SMILES strings into tokens.

    Parameters
    ----------
    smiles_batch : List[str]
        A batch of SMILES strings.
    segment_sq_brackets : bool
        Whether to segment the square brackets `"["` and `"]"` as tokens.
        The default is `True`.

    Returns
    -------
    List[List[str]]
        A list of lists of tokens.
    """
    return [segment_smiles(smiles, segment_sq_brackets) for smiles in smiles_batch]
