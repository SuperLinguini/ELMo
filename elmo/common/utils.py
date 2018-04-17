"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""

from itertools import zip_longest, islice
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Iterable, Iterator
# import importlib
# import logging
# import pkgutil
# import random
# import resource
# import subprocess
# import sys
# import os
#
# import torch
# import numpy
# import spacy
# from spacy.cli.download import download as spacy_download
# from spacy.language import Language as SpacyModelType
#
# from elmo.common.params import Params
# from elmo.common.tqdm import Tqdm
# from elmo.common.tee_logger import TeeLogger

# pylint: disable=invalid-name,protected-access
import logging
import mxnet as mx
from mxnet import nd
from mxnet.ndarray.ndarray import NDArray
from collections import defaultdict
import os
import shutil
from unittest import TestCase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.allennlp'))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")

A = TypeVar('A')

def log_mxnet_version_info():
    logger.info("MXNet version: %s", mx.__version__)

class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    """
    A custom subclass of :class:`~unittest.TestCase` that disables some of the
    more verbose AllenNLP logging and that creates and destroys a temp directory
    as a test fixture.
    """
    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
        log_mxnet_version_info()

        self.TEST_DIR = "/tmp/allennlp_tests/"
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)

def get_dropout_mask(dropout_probability: float, tensor_for_masking: mx.ndarray.ndarray.NDArray):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = mx.nd.random.uniform(0, 1, tensor_for_masking.shape) > dropout_probability
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask

def pad_sequence_to_length(sequence: List,
                           desired_length: int,
                           default_value: Callable[[], Any] = lambda: 0,
                           padding_on_right: bool = True) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence

def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def namespace_match(pattern: str, namespace: str):
    """
    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False

def batch_tensor_dicts(tensor_dicts: List[Dict[str, NDArray]],
                       remove_trailing_dimension: bool = False) -> Dict[str, NDArray]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.

    Parameters
    ----------
    tensor_dicts : ``List[Dict[str, NDArray]]``
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : ``bool``
        If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
        batched, and remove it if we find it.
    """
    key_to_tensors: Dict[str, List[NDArray]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = mx.nd.stack(*tensor_list, axis=0)
        if remove_trailing_dimension and all(tensor.shape[-1] == 1 for tensor in tensor_list):
            batched_tensor = mx.nd.reshape(batched_tensor, batched_tensor.shape[:-2] + (-3,))
        batched_tensors[key] = batched_tensor
    return batched_tensors

def add_sentence_boundary_token_ids(tensor: NDArray,
                                    mask: NDArray,
                                    sentence_begin_token: Any,
                                    sentence_end_token: Any) -> Tuple[NDArray, NDArray]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.

    Returns both the new tensor and updated mask.

    Parameters
    ----------
    tensor : ``NDArray``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``NDArray``
         A tensor of shape ``(batch_size, timesteps)``
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.

    Returns
    -------
    tensor_with_boundary_tokens : ``NDArray``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : ``NDArray``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(axis=1).asnumpy().astype(int).tolist()
    tensor_shape = list(tensor.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = nd.zeros(new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(axis=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask

def remove_sentence_boundaries(tensor: NDArray,
                               mask: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps, dim)``
    this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
    to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of ``add_sentence_boundary_token_ids``.

    Parameters
    ----------
    tensor : ``NDArray``
        A tensor of shape ``(batch_size, timesteps, dim)``
    mask : ``NDArray``
         A tensor of shape ``(batch_size, timesteps)``

    Returns
    -------
    tensor_without_boundary_tokens : ``NDArray``
        The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
    new_mask : ``NDArray``
        The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
    """
    sequence_lengths = mask.sum(axis=1).asnumpy().astype(int).tolist()
    tensor_shape = list(tensor.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = nd.zeros(new_shape)
    new_mask = nd.zeros((new_shape[0], new_shape[1]))
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, :(j - 2), :] = tensor[i, 1:(j - 1), :]
            new_mask[i, :(j - 2)] = 1

    return tensor_without_boundary_tokens, new_mask