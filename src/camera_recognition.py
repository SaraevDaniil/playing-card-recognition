import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque
from torchvision import transforms
import torch.nn.functional as F
from train import SimpleCardClassifier


def detect_cards(frame):
    """
        Detect playing cards in a frame using color thresholding, morphology,
        and contour detection.

        Args:
            frame (np.ndarray): Input BGR frame.

        Returns:
            list: A list of tuples (cropped_card_image, bounding_box)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color threshold
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Edge detection
    blur = cv2.GaussianBlur(mask, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000 or area > frame.shape[0] * frame.shape[1] * 0.5:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        warp = frame[y1:y2, x1:x2]

        cards.append((warp, (x, y, w, h)))
    return cards

def match_to_existing_cards(bbox, existing_cards, max_dist=50):
    """
        Match a bounding box to an existing tracked card using nearest center distance.
        Used for smoothing the prediction output

        Args:
            bbox (tuple): Bounding box (x, y, w, h)
            existing_cards (dict): Dictionary {card_id: (center_x, center_y)}
            max_dist (int): Maximum distance for a match.

        Returns:
            int | None: Card ID if matched, else None.
    """
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    for card_id, (px, py) in existing_cards.items():
        dist = np.hypot(cx - px, cy - py)
        if dist < max_dist:
            return card_id
    return None


def main_loop(camera_index: int = 0, history_length: int = 5) -> None:
    """
        Main webcam loop for detecting and classifying cards.

        Args:
            camera_index (int): Index of the webcam device.
            history_length (int): Number of past predictions used for smoothing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and class names
    checkpoint = torch.load("../models/card_classifier.pth", map_location=device)
    class_names = checkpoint["class_names"]

    model = SimpleCardClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Tracking state
    next_id = 0
    card_histories = {}
    prev_cards = {}

    # Webcam loop
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            detected_cards = []
            current_cards = {}
            existing_centers = {cid: (x + w // 2, y + h // 2)
                                for cid, (x, y, w, h) in prev_cards.items()
                                }

            # Step 1: detect cards and predict
            for card, bbox in detect_cards(frame):
                img = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(tensor)
                    predicted_idx = torch.argmax(outputs, 1).item()
                    predicted_class = class_names[predicted_idx]
                    probs = F.softmax(outputs, dim=1)
                    confidence = probs[0, predicted_idx].item() * 100

                card_id = match_to_existing_cards(bbox, existing_centers)
                if card_id is None:
                    card_id = next_id
                    next_id += 1
                    card_histories[card_id] = deque(maxlen=history_length)

                # Only keep confident predictions
                if confidence >= 50:
                    card_histories[card_id].append(predicted_class)

                # Smooth prediction
                if len(card_histories[card_id]) == history_length:
                    final_class = max(set(card_histories[card_id]), key=card_histories[card_id].count)
                else:
                    final_class = predicted_class

                detected_cards.append((bbox, final_class, confidence))
                current_cards[card_id] = bbox

            prev_cards = current_cards

            # Draw predictions
            for bbox, cls, conf in detected_cards:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{cls}: {conf:.1f}%"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                            )
                # cv2.putText(frame, f'id: {card_id}', (x, y + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Card Recognition", frame)
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop(camera_index=3)
