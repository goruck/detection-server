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

# Define a stack using deque for inter-thread communication.
STACK_DEPTH = 100
detected_objects = collections.deque(maxlen=STACK_DEPTH)

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

            set_input(interpreter, pil_im)
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
        """ Return desired objects in stack, then clear it.
            if stack is empty, return empty object.
        """
        # Fetch the desired labels to return.
        desired_labels = [label for label in request.labels]

        if not detected_objects:
            data = [detection_server_pb2.DetectedObject()]
        else:
            # Since stack was appended left, most recent objs will be read first.
            data = [obj for obj in detected_objects if obj.label in desired_labels]
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
    interpreter = make_interpreter(os.path.join(default_model_dir, args.model))
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