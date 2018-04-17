from typing import List

from mxnet import gluon, nd
from mxnet.ndarray.ndarray import NDArray

from elmo.common.utils import ConfigurationError

class ScalarMix(gluon.Block):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(self, mixture_size: int, do_layer_norm: bool = False) -> None:
        super(ScalarMix, self).__init__()

        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm


        self.scalar_parameters = nd.zeros((3,))
        self.gamma = nd.array([1.0])

    def forward(self, tensors: List[NDArray],  # pylint: disable=arguments-differ
                mask: NDArray = None) -> NDArray:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError("{} tensors were passed, but the module was initialized to "
                                     "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = nd.sum(tensor_masked) / num_elements_not_masked
            variance = nd.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / nd.sqrt(variance + 1E-12)

        normed_weights = nd.softmax(self.scalar_parameters, axis=0)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            broadcast_mask = mask.expand_dims(-1)
            input_dim = tensors[0].shape[-1]
            num_elements_not_masked = nd.sum(mask) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)
