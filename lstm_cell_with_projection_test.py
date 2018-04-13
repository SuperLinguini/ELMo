# pylint: disable=no-self-use,invalid-name
import numpy
import mxnet as mx
from lstm_cell_with_projection import LstmCellWithProjection
from common.utils import AllenNlpTestCase


class TestLstmCellWithProjection(AllenNlpTestCase):
    def test_elmo_lstm_cell_completes_forward_pass(self):
        input_tensor = mx.nd.random.uniform(0, 1, (4,5,3))
        input_tensor[1, 4:, :] = 0.
        input_tensor[2, 2:, :] = 0.
        input_tensor[3, 1:, :] = 0.

        initial_hidden_state = mx.nd.ones([1, 4, 5])
        initial_memory_state = mx.nd.ones([1, 4, 7])

        lstm = LstmCellWithProjection(input_size=3,
                                      hidden_size=5,
                                      cell_size=7,
                                      memory_cell_clip_value=2,
                                      state_projection_clip_value=1)
        lstm.collect_params().initialize(mx.init.Xavier())
        output_sequence, lstm_state = lstm(input_tensor, [5, 4, 2, 1],
                                           (initial_hidden_state, initial_memory_state))
        numpy.testing.assert_array_equal(output_sequence[1, 4:, :].asnumpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence[2, 2:, :].asnumpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence[3, 1:, :].asnumpy(), 0.0)

        # Test the state clipping.
        numpy.testing.assert_array_less(output_sequence.asnumpy(), 1.0)
        numpy.testing.assert_array_less(-output_sequence.asnumpy(), 1.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm_state[0].shape) == [1, 4, 5]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm_state[1].shape)) == [1, 4, 7]

        # Test the cell clipping.
        numpy.testing.assert_array_less(lstm_state[0].asnumpy(), 2.0)
        numpy.testing.assert_array_less(-lstm_state[0].asnumpy(), 2.0)
