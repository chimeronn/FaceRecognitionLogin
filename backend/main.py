from FaceModel import FaceModel
from CreateDataset import create_dataset
import numpy as np
import cv2

def crop_center(frame, num_pixels):
    h, w = frame.shape
    h //= 2
    w //= 2
    return frame[h - num_pixels:h + num_pixels, w - num_pixels:w + num_pixels]

'''data, test_data, labels, testLabels = create_dataset("Dataset", True)
negData, test_neg_data, negLabels, test_negLabels = create_dataset("DatasetNegative", False)
data = np.concatenate([data, negData], axis=0)
labels = np.concatenate([labels, negLabels], axis=0)
test_data = np.concatenate([test_data, test_neg_data], axis=0)
testLabels = np.concatenate([testLabels, test_negLabels], axis=0)
model = FaceModel(data, len(labels), 512 * 512, labels)
model.train(1000)
'''
#for img in test_data:
#    print(model.predict(img))
#print(testLabels)
model = FaceModel(np.empty((0, 512*512)), 0, 512*512, np.empty(0))
model.load("face_model.npz")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    half_size = 256

    top_left = (cx - half_size, cy - half_size)
    bottom_right = (cx + half_size, cy + half_size)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = crop_center(gray, 256)

    input_img = face.astype(np.float32).flatten() / 255.0

    pred = model.predict(input_img)
    label = "My Face" if pred else "Unknown"
    color = (0, 255, 0) if pred else (0, 0, 255)
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break