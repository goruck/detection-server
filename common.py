"""
Common module for detection-server.

Copyright (c) 2020 Lindo St. Angel.
"""

import cv2
import numpy as np
import re
import os
import collections
import tflite_runtime.interpreter as tflite
from PIL import Image

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

# Calibration constants for ELP-USBFHD01M-L21 (2.1mm lens).
# Sensor: 1/3" CMOS OV2710.
CAMERA_MATRIX = np.array(
    [[470.78994798, 0., 332.55335541],
    [ 0., 472.60572742, 255.95334452],
    [0., 0., 1.]],
    dtype=np.float32
)
DIST_COEFFS = np.array(
    [-4.08279482e-01, 2.15993538e-01, 1.96674511e-04, 3.31323115e-04, -7.38822273e-02],
    dtype=np.float32
)

DetectedObject = collections.namedtuple(
    'DetectedObject', ['label', 'score', 'area', 'centroid', 'bbox'])

Centroid = collections.namedtuple('Centroid', ['x', 'y'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    @property
    def centroid(self):
        """Returns centroid of bounding box."""
        return Centroid((self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2)

    @property
    def width(self):
        """Returns bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """Returns bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """Returns bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Returns whether bounding box is valid or not.

        Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
        width >= 0 and height >= 0.
        """
        return self.width >= 0 and self.height >= 0

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
            {'device': device[0]} if device else {})
        ])

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, image, resample=Image.NEAREST):
    """Copies data to input tensor."""
    image = image.resize((input_image_size(interpreter)[0:2]), resample)
    input_tensor(interpreter)[:, :] = image

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def append_object_data(objs, camera_res, img):
    """ Create proto buffer message and add to stack.
    Annotate image with label name and object centroid.
    """
    for obj in objs:
        cx = int(obj.centroid.x*camera_res[0])
        cy = int(obj.centroid.y*camera_res[1])
        label = obj.label
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(img, 'centroid', (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
        cv2.putText(img, label, (cx + 25, cy + 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
    return img

def get_output(interpreter, score_threshold, labels):
    """Returns list of detected objects."""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))

    def get_label(i):
        id = int(class_ids[i])
        return labels.get(id, id)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        bbox = BBox(
            xmin = np.maximum(0.0, xmin),
            ymin = np.maximum(0.0, ymin),
            xmax = np.minimum(1.0, xmax),
            ymax = np.minimum(1.0, ymax))
        return DetectedObject(
            label = get_label(i),
            score = scores[i],
            area = bbox.area,
            centroid = bbox.centroid,
            bbox = bbox)

    return [make(i) for i in range(count) if scores[i] >= score_threshold]