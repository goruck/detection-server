# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: detection_server.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='detection_server.proto',
  package='detection_server',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x16\x64\x65tection_server.proto\x12\x10\x64\x65tection_server\"\x90\x02\n\x0e\x44\x65tectedObject\x12\r\n\x05label\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x0c\n\x04\x61rea\x18\x03 \x01(\x02\x12;\n\x08\x63\x65ntroid\x18\x04 \x01(\x0b\x32).detection_server.DetectedObject.Centroid\x12\x33\n\x04\x62\x62ox\x18\x05 \x01(\x0b\x32%.detection_server.DetectedObject.BBox\x1a \n\x08\x43\x65ntroid\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x1a>\n\x04\x42\x42ox\x12\x0c\n\x04xmin\x18\x01 \x01(\x02\x12\x0c\n\x04ymin\x18\x02 \x01(\x02\x12\x0c\n\x04xmax\x18\x03 \x01(\x02\x12\x0c\n\x04ymax\x18\x04 \x01(\x02\"D\n\x12\x44\x65tectedObjectData\x12.\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32 .detection_server.DetectedObject\"1\n\x10\x43\x61meraResolution\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\"K\n\x19\x43\x61meraIntrinsicParameters\x12\n\n\x02\x66x\x18\x01 \x01(\x02\x12\n\n\x02\x66y\x18\x02 \x01(\x02\x12\n\n\x02\x63x\x18\x03 \x01(\x02\x12\n\n\x02\x63y\x18\x04 \x01(\x02\"\x07\n\x05\x45mpty\"\x1f\n\rDesiredLabels\x12\x0e\n\x06labels\x18\x01 \x03(\t2\xae\x02\n\x0f\x44\x65tectionServer\x12]\n\x12GetDetectedObjects\x12\x1f.detection_server.DesiredLabels\x1a$.detection_server.DetectedObjectData\"\x00\x12T\n\x13GetCameraResolution\x12\x17.detection_server.Empty\x1a\".detection_server.CameraResolution\"\x00\x12\x66\n\x1cGetCameraIntrinsicParameters\x12\x17.detection_server.Empty\x1a+.detection_server.CameraIntrinsicParameters\"\x00\x62\x06proto3'
)




_DETECTEDOBJECT_CENTROID = _descriptor.Descriptor(
  name='Centroid',
  full_name='detection_server.DetectedObject.Centroid',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='detection_server.DetectedObject.Centroid.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='detection_server.DetectedObject.Centroid.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=221,
  serialized_end=253,
)

_DETECTEDOBJECT_BBOX = _descriptor.Descriptor(
  name='BBox',
  full_name='detection_server.DetectedObject.BBox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='xmin', full_name='detection_server.DetectedObject.BBox.xmin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='detection_server.DetectedObject.BBox.ymin', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='detection_server.DetectedObject.BBox.xmax', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='detection_server.DetectedObject.BBox.ymax', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=255,
  serialized_end=317,
)

_DETECTEDOBJECT = _descriptor.Descriptor(
  name='DetectedObject',
  full_name='detection_server.DetectedObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='detection_server.DetectedObject.label', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='detection_server.DetectedObject.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='area', full_name='detection_server.DetectedObject.area', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='centroid', full_name='detection_server.DetectedObject.centroid', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='detection_server.DetectedObject.bbox', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DETECTEDOBJECT_CENTROID, _DETECTEDOBJECT_BBOX, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=317,
)


_DETECTEDOBJECTDATA = _descriptor.Descriptor(
  name='DetectedObjectData',
  full_name='detection_server.DetectedObjectData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='detection_server.DetectedObjectData.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=319,
  serialized_end=387,
)


_CAMERARESOLUTION = _descriptor.Descriptor(
  name='CameraResolution',
  full_name='detection_server.CameraResolution',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='detection_server.CameraResolution.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='detection_server.CameraResolution.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=389,
  serialized_end=438,
)


_CAMERAINTRINSICPARAMETERS = _descriptor.Descriptor(
  name='CameraIntrinsicParameters',
  full_name='detection_server.CameraIntrinsicParameters',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='fx', full_name='detection_server.CameraIntrinsicParameters.fx', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fy', full_name='detection_server.CameraIntrinsicParameters.fy', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cx', full_name='detection_server.CameraIntrinsicParameters.cx', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cy', full_name='detection_server.CameraIntrinsicParameters.cy', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=440,
  serialized_end=515,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='detection_server.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
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
  serialized_start=517,
  serialized_end=524,
)


_DESIREDLABELS = _descriptor.Descriptor(
  name='DesiredLabels',
  full_name='detection_server.DesiredLabels',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='labels', full_name='detection_server.DesiredLabels.labels', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=526,
  serialized_end=557,
)

_DETECTEDOBJECT_CENTROID.containing_type = _DETECTEDOBJECT
_DETECTEDOBJECT_BBOX.containing_type = _DETECTEDOBJECT
_DETECTEDOBJECT.fields_by_name['centroid'].message_type = _DETECTEDOBJECT_CENTROID
_DETECTEDOBJECT.fields_by_name['bbox'].message_type = _DETECTEDOBJECT_BBOX
_DETECTEDOBJECTDATA.fields_by_name['data'].message_type = _DETECTEDOBJECT
DESCRIPTOR.message_types_by_name['DetectedObject'] = _DETECTEDOBJECT
DESCRIPTOR.message_types_by_name['DetectedObjectData'] = _DETECTEDOBJECTDATA
DESCRIPTOR.message_types_by_name['CameraResolution'] = _CAMERARESOLUTION
DESCRIPTOR.message_types_by_name['CameraIntrinsicParameters'] = _CAMERAINTRINSICPARAMETERS
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['DesiredLabels'] = _DESIREDLABELS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DetectedObject = _reflection.GeneratedProtocolMessageType('DetectedObject', (_message.Message,), {

  'Centroid' : _reflection.GeneratedProtocolMessageType('Centroid', (_message.Message,), {
    'DESCRIPTOR' : _DETECTEDOBJECT_CENTROID,
    '__module__' : 'detection_server_pb2'
    # @@protoc_insertion_point(class_scope:detection_server.DetectedObject.Centroid)
    })
  ,

  'BBox' : _reflection.GeneratedProtocolMessageType('BBox', (_message.Message,), {
    'DESCRIPTOR' : _DETECTEDOBJECT_BBOX,
    '__module__' : 'detection_server_pb2'
    # @@protoc_insertion_point(class_scope:detection_server.DetectedObject.BBox)
    })
  ,
  'DESCRIPTOR' : _DETECTEDOBJECT,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.DetectedObject)
  })
_sym_db.RegisterMessage(DetectedObject)
_sym_db.RegisterMessage(DetectedObject.Centroid)
_sym_db.RegisterMessage(DetectedObject.BBox)

DetectedObjectData = _reflection.GeneratedProtocolMessageType('DetectedObjectData', (_message.Message,), {
  'DESCRIPTOR' : _DETECTEDOBJECTDATA,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.DetectedObjectData)
  })
_sym_db.RegisterMessage(DetectedObjectData)

CameraResolution = _reflection.GeneratedProtocolMessageType('CameraResolution', (_message.Message,), {
  'DESCRIPTOR' : _CAMERARESOLUTION,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.CameraResolution)
  })
_sym_db.RegisterMessage(CameraResolution)

CameraIntrinsicParameters = _reflection.GeneratedProtocolMessageType('CameraIntrinsicParameters', (_message.Message,), {
  'DESCRIPTOR' : _CAMERAINTRINSICPARAMETERS,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.CameraIntrinsicParameters)
  })
_sym_db.RegisterMessage(CameraIntrinsicParameters)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.Empty)
  })
_sym_db.RegisterMessage(Empty)

DesiredLabels = _reflection.GeneratedProtocolMessageType('DesiredLabels', (_message.Message,), {
  'DESCRIPTOR' : _DESIREDLABELS,
  '__module__' : 'detection_server_pb2'
  # @@protoc_insertion_point(class_scope:detection_server.DesiredLabels)
  })
_sym_db.RegisterMessage(DesiredLabels)



_DETECTIONSERVER = _descriptor.ServiceDescriptor(
  name='DetectionServer',
  full_name='detection_server.DetectionServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=560,
  serialized_end=862,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetDetectedObjects',
    full_name='detection_server.DetectionServer.GetDetectedObjects',
    index=0,
    containing_service=None,
    input_type=_DESIREDLABELS,
    output_type=_DETECTEDOBJECTDATA,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetCameraResolution',
    full_name='detection_server.DetectionServer.GetCameraResolution',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_CAMERARESOLUTION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetCameraIntrinsicParameters',
    full_name='detection_server.DetectionServer.GetCameraIntrinsicParameters',
    index=2,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_CAMERAINTRINSICPARAMETERS,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DETECTIONSERVER)

DESCRIPTOR.services_by_name['DetectionServer'] = _DETECTIONSERVER

# @@protoc_insertion_point(module_scope)
