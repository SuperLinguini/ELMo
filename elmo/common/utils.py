"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""

# pylint: disable=invalid-name,protected-access
import logging
import mxnet as mx
import os
import shutil
from unittest import TestCase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.allennlp'))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")


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


