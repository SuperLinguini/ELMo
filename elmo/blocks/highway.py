"""
A `Highway layer <https://arxiv.org/abs/1505.00387>`_ that does a gated combination of a linear
transformation and a non-linear transformation of its input.
"""

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.ndarray.ndarray import NDArray

from overrides import overrides


class Highway(gluon.Block):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``nn.activations.Activation``, optional (default=``nn.Activation('relu')``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: nn.activations.Activation = nn.Activation('relu')) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim

        self._layers = []
        for _ in range(num_layers):
            layer = nn.Dense(input_dim * 2, in_units=input_dim)
            self.register_child(layer)
            self._layers.append(layer)

        self._activation = activation

    def set_bias(self):
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias.data()[self._input_dim:] = 1

    @overrides
    def forward(self, inputs: NDArray) -> NDArray:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = gate.sigmoid()
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
