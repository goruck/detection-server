"""
Implements a TPU-based object detection server using grpc.
Returns labels, coordinates of bounding boxes and centroids.

Copyright (c) 2020 Lindo St. Angel.
"""

import argparse
import cv2
import numpy as np
import re
import os
import collections
import grpc
import detection_server_pb2
import detection_server_pb2_grpc
import concurrent
import common
from PIL import Image

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

# Define a stack using deque for inter-thread communication.
STACK_DEPTH = 100
detected_objects = collections.deque(maxlen=STACK_DEPTH)

# We don't care about all labels from detection, just these.
# TODO - make parameter or read from config file.
VALID_LABELS = ['person', 'dog', 'cat']

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
        detected_object = detection_server_pb2.DetectedObject(
            label = obj.label,
            score = obj.score,
            area = obj.area,
            centroid = detection_server_pb2.DetectedObject.Centroid(
                x = obj.centroid.x,
                y = obj.centroid.y
            ),
            bbox = detection_server_pb2.DetectedObject.BBox(
                xmin = obj.bbox.xmin,
                ymin = obj.bbox.ymin,
                xmax = obj.bbox.xmax,
                ymax = obj.bbox.ymax
            )
        )

        detected_objects.appendleft(detected_object)

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
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def get_label(i):
        id = int(class_ids[i])
        return labels.get(id, id)

    def valid_label(i):
        """ Check for labels we care about. """
        return get_label(i) in VALID_LABELS

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

    return [make(i) for i in range(count) if (scores[i] >= score_threshold) and valid_label(i)]

def start_detector(camera_idx, interpreter, threshold, labels, camera_res, display):
    """ Detect objects from camera frames. """
    detected_objects.clear()

    try:
        cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2_im = frame

            cv2_im_u = cv2.undistort(cv2_im, CAMERA_MATRIX, DIST_COEFFS)

            cv2_im_u_rgb = cv2.cvtColor(cv2_im_u, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_u_rgb)

            common.set_input(interpreter, pil_im)
            interpreter.invoke()

            objs = get_output(interpreter,
                score_threshold=threshold,
                labels=labels)

            cv2_im_u = append_object_data(objs, camera_res, cv2_im_u)

            if display:
                cv2.imshow('frame', cv2_im_u)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except cv2.error as e:
        print('cv2 error: {e}'.format(e))
    finally: 
        cap.release()
        cv2.destroyAllWindows()

    return

# DetectionServerServicer provides an implementation of the methods of the DetectionServer service.
class DetectionServerServicer(detection_server_pb2_grpc.DetectionServerServicer):
    def __init__(self, camera_res):
        self.camera_res = camera_res

    def GetCameraResolution(self, request, context):
        """ Return camera resolution. """
        return detection_server_pb2.CameraResolution(
            width=self.camera_res[0], height=self.camera_res[1])

    def GetCameraIntrinsicParameters(self, request, context):
        """ Return intrinstic params from camera matrix. """
        return detection_server_pb2.CameraIntrinsicParameters(
            fx=CAMERA_MATRIX[0,0], fy=CAMERA_MATRIX[1,1],
            cx=CAMERA_MATRIX[0,2], cy=CAMERA_MATRIX[1,2])

    def GetDetectedObjects(self, request, context):
        """ Return all objects in stack, then clear it.
            if stack is empty, return empty object.
        """
        if not detected_objects:
            data = [detection_server_pb2.DetectedObject()]
        else:
            # Since stack was appended left, most recent objs will be read first.
            data = [obj for obj in detected_objects]
            detected_objects.clear()

        return detection_server_pb2.DetectedObjectData(data=data)

def serve():
    default_model_dir = '/media/mendel/detection-server/models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=1)
    parser.add_argument('--threshold', type=float, help='Detector threshold. ', default=0.7)
    parser.add_argument('--display', dest='display', action='store_true', help='Display object data. ')
    parser.set_defaults(display=False)
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(os.path.join(default_model_dir, args.model))
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    # Get native camera resolution.
    cap = cv2.VideoCapture(args.camera_idx)
    camera_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start a thread to detect objects in camera frames.
        future = executor.submit(start_detector, args.camera_idx, interpreter,
            args.threshold, labels, camera_res, args.display)

        # Start other threads for the gprc server. 
        server = grpc.server(executor)
        detection_server_pb2_grpc.add_DetectionServerServicer_to_server(
            DetectionServerServicer(camera_res), server)
        server.add_insecure_port('[::]:50051')
        server.start()

        # Show the value returned by the executor.submit call.
        # This will wait forever unless a runtime error is encountered.
        future.result()

        server.stop(None)

if __name__ == '__main__':
    serve()