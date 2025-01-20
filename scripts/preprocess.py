import cv2

def extract_frames(video_path, frame_size=(640, 640)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to YOLOv8's expected input size
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()
    return frames