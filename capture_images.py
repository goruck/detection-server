"""
Capture images of classes for transfer learning.

Copyright (c) 2020 Lindo St. Angel.
"""

import argparse
import cv2
import os
import time
import common
from PIL import Image

IMG_DIR = './images'
DESIRED_CLASSES = ['person', 'dog', 'cat']

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
                cv2_im_u = cv2.undistort(cv2_im, common.CAMERA_MATRIX,
                    common.DIST_COEFFS)
                cv2_im_u_rgb = cv2.cvtColor(cv2_im_u, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_u_rgb)
                common.set_input(interpreter, pil_im)
                interpreter.invoke()

                objs = common.get_output(interpreter,
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
                    cv2_im_u = common.append_object_data(objs, camera_res, cv2_im_u)
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
    interpreter = common.make_interpreter(os.path.join(default_model_dir, args.model))
    interpreter.allocate_tensors()
    labels = common.load_labels(os.path.join(default_model_dir, args.labels))

    # Get native camera resolution.
    cap = cv2.VideoCapture(args.camera_idx)
    camera_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    capture(args.camera_idx, interpreter, args.threshold, labels, camera_res,
        args.display, args.frame_rate, args.start_sample, args.capture_samples)

if __name__ == '__main__':
    main()