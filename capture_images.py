"""
Capture images of classes for transfer learning.

Copyright (c) 2020 Lindo St. Angel.
"""

import argparse
import cv2
import numpy as np
import re
import os
import collections
import time
import tflite_runtime.interpreter as tflite
from PIL import Image

IMG_DIR = './images'
DESIRED_CLASSES = ['person', 'dog', 'cat']
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

def capture(camera_idx, interpreter, threshold, labels, camera_res, display,
    frame_rate, start_sample, capture_samples):
    """ Capture images from camera frames and write to disk. """
    sample = start_sample
    prev = 0

    try:
        cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            time_elapsed = time.time() - prev

            if sample > start_sample + capture_samples:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if time_elapsed > 1.0 / frame_rate:
                prev = time.time()
                cv2_im = frame
                cv2_im_u = cv2.undistort(cv2_im, CAMERA_MATRIX, DIST_COEFFS)
                cv2_im_u_rgb = cv2.cvtColor(cv2_im_u, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_u_rgb)
                set_input(interpreter, pil_im)
                interpreter.invoke()

                objs = get_output(interpreter,
                    score_threshold=threshold, labels=labels)

                for obj in objs:
                    if obj.label in DESIRED_CLASSES:
                        img_name = ''.join((str(sample), '.jpg'))
                        img_path = '{}'.format(os.path.join(IMG_DIR, img_name))
                        print('Found "{}" at t+ {:.2f} sec. Writing "{}".'.format(
                            obj.label, time_elapsed, img_name))
                        cv2.imwrite(img_path, cv2_im_u)
                        sample += 1

                if display:
                    cv2_im_u = append_object_data(objs, camera_res, cv2_im_u)
                    cv2.imshow('frame', cv2_im_u)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except cv2.error as e:
        print('cv2 error: {e}'.format(e))
    except Exception as e:
        print('Unhandled error: {e}'.format(e))
    finally: 
        cap.release()
        cv2.destroyAllWindows()

    return

def main():
    default_model_dir = '/media/mendel/detection-server/models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=1)
    parser.add_argument('--threshold', type=float, help='Detector threshold. ', default=0.1)
    parser.add_argument('--frame_rate', type=float, help='Capture frame rate. ', default=2.0)
    parser.add_argument('--start_sample', type=int, help='Starting sample index. ', required=True)
    parser.add_argument('--capture_samples', type=int, help='Starting sample index. ', default=100)
    parser.add_argument('--display', dest='display', action='store_true', help='Display object data. ')
    parser.set_defaults(display=False)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(os.path.join(default_model_dir, args.model))
    interpreter.allocate_tensors()
    labels = load_labels(os.path.join(default_model_dir, args.labels))

    # Get native camera resolution.
    cap = cv2.VideoCapture(args.camera_idx)
    camera_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    capture(args.camera_idx, interpreter, args.threshold, labels, camera_res,
        args.display, args.frame_rate, args.start_sample, args.capture_samples)

if __name__ == '__main__':
    main()