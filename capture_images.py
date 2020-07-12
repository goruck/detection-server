"""
Capture images of classes for transfer learning.

Each captured image is given a unique id and placed in a named directory.

Copyright (c) 2020 Lindo St. Angel.
"""

import argparse
import cv2
import os
import time
import uuid
import common
from PIL import Image

def capture(parse_args, interpreter, labels, camera_res):
    """ Capture images from camera frames and write to disk. """
    sample = 0
    prev = 0

    try:
        cap = cv2.VideoCapture(parse_args.camera_idx)
        while cap.isOpened():
            time_elapsed = time.time() - prev

            if sample > parse_args.num_samples:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if time_elapsed > 1.0 / parse_args.frame_rate:
                prev = time.time()
                cv2_im = frame
                cv2_im_u = cv2.undistort(cv2_im, common.CAMERA_MATRIX,
                    common.DIST_COEFFS)
                cv2_im_u_rgb = cv2.cvtColor(cv2_im_u, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_u_rgb)
                common.set_input(interpreter, pil_im)
                interpreter.invoke()

                objs = common.get_output(interpreter,
                    score_threshold=parse_args.threshold, labels=labels)

                for obj in objs:
                    if obj.label in parse_args.capture:
                        id = uuid.uuid4()
                        img_name = ''.join((str(id), '.jpg'))
                        img_path = '{}'.format(os.path.join(parse_args.images,
                            obj.label, img_name))
                        print('Found "{}" at t+ {:.2f} sec. Writing "{}".'.format(
                            obj.label, time_elapsed, img_name))
                        cv2.imwrite(img_path, cv2_im_u)
                        sample += 1

                if parse_args.display:
                    cv2_im_u = common.append_object_data(objs, camera_res, cv2_im_u)
                    cv2.imshow('detections', cv2_im_u)
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
    default_capture_labels = ['person', 'dog', 'cat']
    default_capture_dir = '/media/mendel/detection-server/images'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='Label file path.',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--images', help='Captured image path.', default=default_capture_dir)
    parser.add_argument('--capture', nargs='+', type=str, help='Labels to capture.',
                        default=default_capture_labels)
    parser.add_argument('--num_samples', type=int, help='Number of samples to capture.', default=100)
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use.', default=1)
    parser.add_argument('--threshold', type=float, help='Detector threshold.', default=0.1)
    parser.add_argument('--frame_rate', type=float, help='Capture frame rate.', default=2.0)
    parser.add_argument('--display', dest='display', action='store_true', help='Display object data.')
    parser.set_defaults(display=False)
    parse_args = parser.parse_args()

    print('Loading {} with {} labels.'.format(parse_args.model, parse_args.labels))
    interpreter = common.make_interpreter(os.path.join(default_model_dir, parse_args.model))
    interpreter.allocate_tensors()
    labels = common.load_labels(os.path.join(default_model_dir, parse_args.labels))

    # Get native camera resolution.
    cap = cv2.VideoCapture(parse_args.camera_idx)
    camera_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create named directories for captured images.
    for name in parse_args.capture:
        dir = os.path.join(parse_args.images, name)
        try:
            os.mkdir(dir)
            print('Created {}'.format(dir))
        except FileExistsError:
            pass

    capture(parse_args, interpreter, labels, camera_res)

if __name__ == '__main__':
    main()