# pylint: disable=invalid-name,no-self-use,too-many-public-methods
import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import torch
from torch.autograd import Variable
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn import util


class TestNnUtil(AllenNlpTestCase):
    def test_add_sentence_boundary_token_ids_handles_2D_input(self):
        tensor = Variable(torch.from_numpy(numpy.array([[1, 2, 3], [4, 5, 0]])))
        mask = (tensor > 0).long()
        bos = 9
        eos = 10
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor = numpy.array([[9, 1, 2, 3, 10],
                                           [9, 4, 5, 10, 0]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == (expected_new_tensor > 0)).all()

    def test_add_sentence_boundary_token_ids_handles_3D_input(self):
        tensor = Variable(torch.from_numpy(
                numpy.array([[[1, 2, 3, 4],
                              [5, 5, 5, 5],
                              [6, 8, 1, 2]],
                             [[4, 3, 2, 1],
                              [8, 7, 6, 5],
                              [0, 0, 0, 0]]])))
        mask = ((tensor > 0).sum(dim=-1) > 0).type(torch.LongTensor)
        bos = Variable(torch.from_numpy(numpy.array([9, 9, 9, 9])))
        eos = Variable(torch.from_numpy(numpy.array([10, 10, 10, 10])))
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor = numpy.array([[[9, 9, 9, 9],
                                            [1, 2, 3, 4],
                                            [5, 5, 5, 5],
                                            [6, 8, 1, 2],
                                            [10, 10, 10, 10]],
                                           [[9, 9, 9, 9],
                                            [4, 3, 2, 1],
                                            [8, 7, 6, 5],
                                            [10, 10, 10, 10],
                                            [0, 0, 0, 0]]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == ((expected_new_tensor > 0).sum(axis=-1) > 0)).all()

    def test_remove_sentence_boundaries(self):
        tensor = Variable(torch.from_numpy(numpy.random.rand(3, 5, 7)))
        mask = Variable(torch.from_numpy(
                # The mask with two elements is to test the corner case
                # of an empty sequence, so here we are removing boundaries
                # from  "<S> </S>"
                numpy.array([[1, 1, 0, 0, 0],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 0]]))).long()
        new_tensor, new_mask = util.remove_sentence_boundaries(tensor, mask)

        expected_new_tensor = Variable(torch.zeros(3, 3, 7))
        expected_new_tensor[1, 0:3, :] = tensor[1, 1:4, :]
        expected_new_tensor[2, 0:2, :] = tensor[2, 1:3, :]
        assert_array_almost_equal(new_tensor.data.numpy(), expected_new_tensor.data.numpy())

        expected_new_mask = Variable(torch.from_numpy(
                numpy.array([[0, 0, 0],
                             [1, 1, 1],
                             [1, 1, 0]]))).long()
        assert (new_mask.data.numpy() == expected_new_mask.data.numpy()).all()