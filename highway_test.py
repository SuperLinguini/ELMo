# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal

from mxnet import nd

from blocks.highway import Highway
from common.utils import AllenNlpTestCase


class TestHighway(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        highway = Highway(2, 2)
        highway.collect_params().initialize()
        highway.set_bias()
        # pylint: disable=protected-access
        highway._layers[0].weight.data()[:] = 1
        highway._layers[0].bias.data()[:] = 0
        highway._layers[1].weight.data()[:] = 2
        highway._layers[1].bias.data()[:] = -2
        input_tensor = nd.array([[-2, 1], [3, -2]])
        result = highway(input_tensor).asnumpy()
        assert result.shape == (2, 2)
        # This was checked by hand.
        assert_almost_equal(result, [[-0.0394, 0.0197], [1.7527, -0.5550]], decimal=4)
