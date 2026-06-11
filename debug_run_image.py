import cv2
from app import process_image

img = cv2.imread(r'D:\newppe\roboflow\test\images\000007_jpg.rf.49b2d719a70bee09ead2b9091ff330c4.jpg')
print('image shape', img.shape, 'dtype', img.dtype)
annotated, rows = process_image(img, 0.35, 0.45)
print('annotated type', type(annotated), 'shape', annotated.shape, 'dtype', annotated.dtype)
print('rows', rows[:3])
cv2.imwrite('debug_annotated.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print('wrote debug_annotated.jpg')
