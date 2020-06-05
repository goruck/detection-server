# detection_server

This project implements object detection served over grpc using the Google Edge TPU. These detections are returned from the server in the form of the centroid coordinates of the object's bounding box and its label. 

This is part of the [radar-ml](https://github.com/goruck/radar-ml) project.

# Installation
Clone this directory and cd to it.

Install the required Python modules in [requirements.txt](./requirements.txt) and OpenCV per the instructions [here](https://github.com/goruck/smart-zoneminder/tree/master/tpu-servers#installation).

If you'd like to compile the Protocol Buffers from scratch, use the following command.

```bash
$ python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./detection_server.proto
```

# Usage
See [detection_server.proto](./detection_server.proto) for the server's interface definitions. 

The following illustrates an example client calling the detection server's services.

```python
"""
Example client for detection_server.py.

Copyright (c) 2020 Lindo St. Angel
"""

import grpc
import detection_server_pb2
import detection_server_pb2_grpc

def get_camera_resolution(stub):
    request = detection_server_pb2.Empty()
    try:
        response = stub.GetCameraResolution(request)
        print('Camera resolution fetched.')
        return response
    except grpc.RpcError as err:
        print(err.details()) #pylint: disable=no-member
        print('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def get_detected_objects(stub):
    request = detection_server_pb2.DesiredLabels(labels=['person', 'dog', 'cat'])
    try:
        response = stub.GetDetectedObjects(request)
        print('Detected object(s) fetched.')
        return response
    except grpc.RpcError as err:
        print(err.details()) #pylint: disable=no-member
        print('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = detection_server_pb2_grpc.DetectionServerStub(channel)
        while True:
            res = get_camera_resolution(stub)
            print(res)
            res = get_detected_objects(stub)
            print(res)

if __name__ == '__main__':
    run()
```

# License
Everything here is licensed under the [MIT license](./LICENSE).

# Contact
For questions or comments about this project please contact the author goruck (Lindo St. Angel) at {lindostangel} AT {gmail} DOT {com}.