import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

model = YOLO("../models/best.pt")
track_history = defaultdict(lambda: [])

def main_loop(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            if ret:
                result = model.track(frame, persist=True, conf=0.5, iou=0.5)[0]

                if result.boxes and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    frame = result.plot()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop()

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            cv2.imshow('YOLO Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop(camera_index=3)