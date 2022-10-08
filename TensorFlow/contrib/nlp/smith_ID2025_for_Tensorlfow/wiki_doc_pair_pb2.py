# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: smith/wiki_doc_pair.proto
"""Generated protocol buffer code."""
from npu_bridge.npu_init import *
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='smith/wiki_doc_pair.proto',
  package='smith',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x19smith/wiki_doc_pair.proto\x12\x05smith\"\xa6\x02\n\x0bWikiDocPair\x12\n\n\x02id\x18\x01 \x01(\t\x12(\n machine_label_for_classification\x18\x02 \x01(\x05\x12&\n\x1ehuman_label_for_classification\x18\x03 \x01(\x05\x12$\n\x1cmachine_label_for_regression\x18\x04 \x01(\x02\x12\"\n\x1ahuman_label_for_regression\x18\x05 \x01(\x02\x12\x1f\n\x07\x64oc_one\x18\x06 \x01(\x0b\x32\x0e.smith.WikiDoc\x12\x1f\n\x07\x64oc_two\x18\x07 \x01(\x0b\x32\x0e.smith.WikiDoc\x12\x18\n\x10model_prediction\x18\x08 \x01(\x02\x12\x13\n\x0bhuman_label\x18\t \x03(\x05\"\x83\x01\n\x07WikiDoc\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0b\n\x03url\x18\x02 \x01(\t\x12\r\n\x05title\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12(\n\x10section_contents\x18\x05 \x03(\x0b\x32\x0e.smith.Section\x12\x11\n\timage_ids\x18\x06 \x03(\t\"&\n\x07Section\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t'
)




_WIKIDOCPAIR = _descriptor.Descriptor(
  name='WikiDocPair',
  full_name='smith.WikiDocPair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='smith.WikiDocPair.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='machine_label_for_classification', full_name='smith.WikiDocPair.machine_label_for_classification', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='human_label_for_classification', full_name='smith.WikiDocPair.human_label_for_classification', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='machine_label_for_regression', full_name='smith.WikiDocPair.machine_label_for_regression', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='human_label_for_regression', full_name='smith.WikiDocPair.human_label_for_regression', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='doc_one', full_name='smith.WikiDocPair.doc_one', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='doc_two', full_name='smith.WikiDocPair.doc_two', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_prediction', full_name='smith.WikiDocPair.model_prediction', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='human_label', full_name='smith.WikiDocPair.human_label', index=8,
      number=9, type=5, cpp_type=1, label=3,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=37,
  serialized_end=331,
)


_WIKIDOC = _descriptor.Descriptor(
  name='WikiDoc',
  full_name='smith.WikiDoc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='smith.WikiDoc.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='url', full_name='smith.WikiDoc.url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='title', full_name='smith.WikiDoc.title', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='description', full_name='smith.WikiDoc.description', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='section_contents', full_name='smith.WikiDoc.section_contents', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_ids', full_name='smith.WikiDoc.image_ids', index=5,
      number=6, type=9, cpp_type=9, label=3,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=334,
  serialized_end=465,
)


_SECTION = _descriptor.Descriptor(
  name='Section',
  full_name='smith.Section',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='title', full_name='smith.Section.title', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='text', full_name='smith.Section.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=467,
  serialized_end=505,
)

_WIKIDOCPAIR.fields_by_name['doc_one'].message_type = _WIKIDOC
_WIKIDOCPAIR.fields_by_name['doc_two'].message_type = _WIKIDOC
_WIKIDOC.fields_by_name['section_contents'].message_type = _SECTION
DESCRIPTOR.message_types_by_name['WikiDocPair'] = _WIKIDOCPAIR
DESCRIPTOR.message_types_by_name['WikiDoc'] = _WIKIDOC
DESCRIPTOR.message_types_by_name['Section'] = _SECTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WikiDocPair = _reflection.GeneratedProtocolMessageType('WikiDocPair', (_message.Message,), {
  'DESCRIPTOR' : _WIKIDOCPAIR,
  '__module__' : 'smith.wiki_doc_pair_pb2'
  # @@protoc_insertion_point(class_scope:smith.WikiDocPair)
  })
_sym_db.RegisterMessage(WikiDocPair)

WikiDoc = _reflection.GeneratedProtocolMessageType('WikiDoc', (_message.Message,), {
  'DESCRIPTOR' : _WIKIDOC,
  '__module__' : 'smith.wiki_doc_pair_pb2'
  # @@protoc_insertion_point(class_scope:smith.WikiDoc)
  })
_sym_db.RegisterMessage(WikiDoc)

Section = _reflection.GeneratedProtocolMessageType('Section', (_message.Message,), {
  'DESCRIPTOR' : _SECTION,
  '__module__' : 'smith.wiki_doc_pair_pb2'
  # @@protoc_insertion_point(class_scope:smith.Section)
  })
_sym_db.RegisterMessage(Section)


# @@protoc_insertion_point(module_scope)

