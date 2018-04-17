"""
A :class:`~elmo.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from elmo.data.fields.field import DataArray, Field
# from elmo.data.fields.array_field import ArrayField
# from elmo.data.fields.index_field import IndexField
from elmo.data.fields.label_field import LabelField
# from elmo.data.fields.list_field import ListField
# from elmo.data.fields.metadata_field import MetadataField
from elmo.data.fields.sequence_field import SequenceField
# from elmo.data.fields.sequence_label_field import SequenceLabelField
# from elmo.data.fields.span_field import SpanField
from elmo.data.fields.text_field import TextField
