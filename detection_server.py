"""
Implements a TPU-based object detection server using grpc.
Returns labels, coordinates of bounding boxes and centroids.

Copyright (c) 2020 Lindo St. Angel.
"""

import common
import argparse
import cv2
import os
import collections
import grpc
import detection_server_pb2
import detection_server_pb2_grpc
import concurrent
from PIL import Image

# Define a stack using deque for inter-thread communication.
STACK_DEPTH = 100
detected_objects = collections.deque(maxlen=STACK_DEPTH)

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

            cv2_im_u = cv2.undistort(cv2_im, common.CAMERA_MATRIX, common.DIST_COEFFS)

            cv2_im_u_rgb = cv2.cvtColor(cv2_im_u, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_u_rgb)

            common.set_input(interpreter, pil_im)
            interpreter.invoke()

            objs = common.get_output(interpreter,
                score_threshold=threshold,
                labels=labels)

            cv2_im_u = common.annotate_image(objs, camera_res, cv2_im_u)

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
            fx=common.CAMERA_MATRIX[0,0], fy=common.CAMERA_MATRIX[1,1],
            cx=common.CAMERA_MATRIX[0,2], cy=common.CAMERA_MATRIX[1,2])

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
    interpreter = common.make_interpreter(os.path.join(default_model_dir, args.model))
    interpreter.allocate_tensors()
    labels = common.load_labels(os.path.join(default_model_dir, args.labels))

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