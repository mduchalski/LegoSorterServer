# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LegoSorter.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Messages_pb2 as Messages__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='LegoSorter.proto',
  package='sorter',
  syntax='proto3',
  serialized_options=b'\n\022com.lsorter.sorterB\017LegoSorterProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10LegoSorter.proto\x12\x06sorter\x1a\x0eMessages.proto\"F\n\x14\x42oundingBoxWithIndex\x12\r\n\x05index\x18\x01 \x01(\x05\x12\x1f\n\x02\x62\x62\x18\x02 \x01(\x0b\x32\x13.common.BoundingBox\"N\n\x1eListOfBoundingBoxesWithIndexes\x12,\n\x06packet\x18\x01 \x03(\x0b\x32\x1c.sorter.BoundingBoxWithIndex\"\x15\n\x13SorterConfiguration2\x9d\x02\n\nLegoSorter\x12P\n\x10processNextImage\x12\x14.common.ImageRequest\x1a&.sorter.ListOfBoundingBoxesWithIndexes\x12>\n\x10getConfiguration\x12\r.common.Empty\x1a\x1b.sorter.SorterConfiguration\x12O\n\x13updateConfiguration\x12\x1b.sorter.SorterConfiguration\x1a\x1b.sorter.SorterConfiguration\x12,\n\x0cstartMachine\x12\r.common.Empty\x1a\r.common.EmptyB%\n\x12\x63om.lsorter.sorterB\x0fLegoSorterProtob\x06proto3'
  ,
  dependencies=[Messages__pb2.DESCRIPTOR,])




_BOUNDINGBOXWITHINDEX = _descriptor.Descriptor(
  name='BoundingBoxWithIndex',
  full_name='sorter.BoundingBoxWithIndex',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='sorter.BoundingBoxWithIndex.index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bb', full_name='sorter.BoundingBoxWithIndex.bb', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=114,
)


_LISTOFBOUNDINGBOXESWITHINDEXES = _descriptor.Descriptor(
  name='ListOfBoundingBoxesWithIndexes',
  full_name='sorter.ListOfBoundingBoxesWithIndexes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='packet', full_name='sorter.ListOfBoundingBoxesWithIndexes.packet', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=194,
)


_SORTERCONFIGURATION = _descriptor.Descriptor(
  name='SorterConfiguration',
  full_name='sorter.SorterConfiguration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=196,
  serialized_end=217,
)

_BOUNDINGBOXWITHINDEX.fields_by_name['bb'].message_type = Messages__pb2._BOUNDINGBOX
_LISTOFBOUNDINGBOXESWITHINDEXES.fields_by_name['packet'].message_type = _BOUNDINGBOXWITHINDEX
DESCRIPTOR.message_types_by_name['BoundingBoxWithIndex'] = _BOUNDINGBOXWITHINDEX
DESCRIPTOR.message_types_by_name['ListOfBoundingBoxesWithIndexes'] = _LISTOFBOUNDINGBOXESWITHINDEXES
DESCRIPTOR.message_types_by_name['SorterConfiguration'] = _SORTERCONFIGURATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BoundingBoxWithIndex = _reflection.GeneratedProtocolMessageType('BoundingBoxWithIndex', (_message.Message,), {
  'DESCRIPTOR' : _BOUNDINGBOXWITHINDEX,
  '__module__' : 'LegoSorter_pb2'
  # @@protoc_insertion_point(class_scope:sorter.BoundingBoxWithIndex)
  })
_sym_db.RegisterMessage(BoundingBoxWithIndex)

ListOfBoundingBoxesWithIndexes = _reflection.GeneratedProtocolMessageType('ListOfBoundingBoxesWithIndexes', (_message.Message,), {
  'DESCRIPTOR' : _LISTOFBOUNDINGBOXESWITHINDEXES,
  '__module__' : 'LegoSorter_pb2'
  # @@protoc_insertion_point(class_scope:sorter.ListOfBoundingBoxesWithIndexes)
  })
_sym_db.RegisterMessage(ListOfBoundingBoxesWithIndexes)

SorterConfiguration = _reflection.GeneratedProtocolMessageType('SorterConfiguration', (_message.Message,), {
  'DESCRIPTOR' : _SORTERCONFIGURATION,
  '__module__' : 'LegoSorter_pb2'
  # @@protoc_insertion_point(class_scope:sorter.SorterConfiguration)
  })
_sym_db.RegisterMessage(SorterConfiguration)


DESCRIPTOR._options = None

_LEGOSORTER = _descriptor.ServiceDescriptor(
  name='LegoSorter',
  full_name='sorter.LegoSorter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=220,
  serialized_end=505,
  methods=[
  _descriptor.MethodDescriptor(
    name='processNextImage',
    full_name='sorter.LegoSorter.processNextImage',
    index=0,
    containing_service=None,
    input_type=Messages__pb2._IMAGEREQUEST,
    output_type=_LISTOFBOUNDINGBOXESWITHINDEXES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getConfiguration',
    full_name='sorter.LegoSorter.getConfiguration',
    index=1,
    containing_service=None,
    input_type=Messages__pb2._EMPTY,
    output_type=_SORTERCONFIGURATION,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='updateConfiguration',
    full_name='sorter.LegoSorter.updateConfiguration',
    index=2,
    containing_service=None,
    input_type=_SORTERCONFIGURATION,
    output_type=_SORTERCONFIGURATION,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='startMachine',
    full_name='sorter.LegoSorter.startMachine',
    index=3,
    containing_service=None,
    input_type=Messages__pb2._EMPTY,
    output_type=Messages__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_LEGOSORTER)

DESCRIPTOR.services_by_name['LegoSorter'] = _LEGOSORTER

# @@protoc_insertion_point(module_scope)
