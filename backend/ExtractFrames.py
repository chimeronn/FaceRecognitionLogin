import cv2
import os

def extract_frames(path, output_dir, every_n_frames=3, max_frames=1000):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    count = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % every_n_frames == 0:
            face = crop_center(frame, 256)
            filename = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(filename, face)
            saved += 1
        count += 1
    cap.release()
    print(f"Saved {saved} frames to {output_dir}")

def crop_center(frame, num_pixels):
    h, w, _ = frame.shape
    h //= 2
    w //= 2
    return frame[h - num_pixels:h + num_pixels, w - num_pixels:w + num_pixels]

extract_frames("FaceVideo/FaceVideo.mp4", "Dataset")
extract_frames("NoFaceVideo/NoFaceVideo.mp4", "DatasetNegative")
extract_frames("FaceVideo/FaceVideo2.mp4", "Dataset")
extract_frames("NoFaceVideo/NoFaceVideo2.mp4", "DatasetNegative")
