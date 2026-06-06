import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
res = model(img, conf=0.3)
print('results count', len(res))
result = res[0]
print('boxes', result.boxes)
print('keypoints type', type(result.keypoints), getattr(result, 'keypoints', None))
kp = result.keypoints
if kp is not None:
    print('keypoints shape', kp.shape)
    print('keypoints sample', kp[0])
else:
    print('no keypoints')
