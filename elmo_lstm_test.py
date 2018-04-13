# pylint: disable=no-self-use,invalid-name,protected-access
import numpy
import mxnet as mx

from elmo_lstm import ElmoLstm
from common.utils import AllenNlpTestCase


class TestElmoLstmCell(AllenNlpTestCase):
    def test_elmo_lstm(self):
        input_tensor = mx.nd.random.uniform(0, 1, (4,5,3))
        input_tensor[1, 4:, :] = 0.
        input_tensor[2, 2:, :] = 0.
        input_tensor[3, 1:, :] = 0.
        mask = mx.nd.ones([4, 5])
        mask[1, 4:] = 0.
        mask[2, 2:] = 0.
        mask[3, 1:] = 0.

        lstm = ElmoLstm(num_layers=2,
                        input_size=3,
                        hidden_size=5,
                        cell_size=7,
                        memory_cell_clip_value=2,
                        state_projection_clip_value=1)
        output_sequence = lstm(input_tensor, mask)

        # Check all the layer outputs are masked properly.
        numpy.testing.assert_array_equal(output_sequence[:, 1, 4:, :].asnumpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence[:, 2, 2:, :].asnumpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence[:, 3, 1:, :].asnumpy(), 0.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm._states[0].shape) == [2, 4, 10]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm._states[1].shape)) == [2, 4, 14]
