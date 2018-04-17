# pylint: disable=no-self-use,invalid-name

from mxnet import nd

import pytest
import numpy

from elmo.common.utils import AllenNlpTestCase, ConfigurationError
from elmo.blocks.scalar_mix import ScalarMix


class TestScalarMix(AllenNlpTestCase):
    def test_scalar_mix_can_run_forward(self):
        mixture = ScalarMix(3)
        tensors = [nd.random.normal(shape=(3, 4, 5)) for _ in range(3)]
        for k in range(3):
            mixture.scalar_parameters[k] = 0.1 * (k + 1)
        mixture.gamma[0] = 0.5
        result = mixture(tensors)

        weights = [0.1, 0.2, 0.3]
        normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
        expected_result = sum(normed_weights[k] * tensors[k].asnumpy() for k in range(3))
        expected_result *= 0.5
        numpy.testing.assert_almost_equal(expected_result, result.asnumpy())

    def test_scalar_mix_throws_error_on_incorrect_number_of_inputs(self):
        mixture = ScalarMix(3)
        tensors = [nd.random.normal(shape=(3, 4, 5)) for _ in range(5)]
        with pytest.raises(ConfigurationError):
            _ = mixture(tensors)

    def test_scalar_mix_layer_norm(self):
        mixture = ScalarMix(3, do_layer_norm='scalar_norm_reg')

        tensors = [nd.random.normal(shape=(3, 4, 5)) for _ in range(3)]
        numpy_mask = numpy.ones((3, 4), dtype='int32')
        numpy_mask[1, 2:] = 0
        mask = nd.array(numpy_mask)

        weights = [0.1, 0.2, 0.3]
        for k in range(3):
            mixture.scalar_parameters[k] = 0.1 * (k + 1)
        mixture.gamma[0] = 0.5
        result = mixture(tensors, mask)

        normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
        expected_result = numpy.zeros((3, 4, 5))
        for k in range(3):
            mean = numpy.mean(tensors[k].asnumpy()[numpy_mask == 1])
            std = numpy.std(tensors[k].asnumpy()[numpy_mask == 1])
            normed_tensor = (tensors[k].asnumpy() - mean) / (std + 1E-12)
            expected_result += (normed_tensor * normed_weights[k])
        expected_result *= 0.5

        numpy.testing.assert_almost_equal(expected_result, result.asnumpy())
