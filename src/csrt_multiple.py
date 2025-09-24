import cv2
import numpy as np

cap = cv2.VideoCapture(3)

trackers = []
frame_count = 0
DETECTION_INTERVAL = 20  # detect every 20 frames


def detect_cards(frame):
    """Detect rectangular shapes with card-like ratio."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:  # Quadrilateral
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.6 < aspect_ratio < 0.8 and w > 40 and h > 60:  # card-like
                bboxes.append((x, y, w, h))

    return bboxes


def is_overlapping(new_bbox, existing_bbox, thresh=0.3):
    """Check if two bboxes overlap significantly."""
    x1, y1, w1, h1 = new_bbox
    x2, y2, w2, h2 = existing_bbox

    # intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return False

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    overlap = inter_area / float(union_area)

    return overlap > thresh


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update trackers
    new_trackers = []
    updated_bboxes = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            new_trackers.append(tracker)
            updated_bboxes.append((x, y, w, h))
    trackers = new_trackers

    # Run detection every N frames
    if frame_count % DETECTION_INTERVAL == 0:
        detected_bboxes = detect_cards(frame)

        for bbox in detected_bboxes:
            # Skip if overlapping with an existing tracker
            if any(is_overlapping(bbox, ub) for ub in updated_bboxes):
                continue

            # Add new tracker
            tracker = cv2.legacy.TrackerCSRT.create()
            tracker.init(frame, bbox)
            trackers.append(tracker)

    frame_count += 1
    cv2.imshow("Multiple Card Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
