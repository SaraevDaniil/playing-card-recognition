from symbol import continue_stmt

import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from main import SimpleCardClassifier, PlayingCardDataset
import torch.nn.functional as F


# Function to detect and crop the card
def crop_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Filter out too small or too large contours
    area = cv2.contourArea(largest)
    if area < 5000 or area > frame.shape[0] * frame.shape[1] * 0.9:
        return None, None

    # Approximate contour to a polygon and check if it has 4 corners
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = h / float(w)
    if aspect_ratio < 1.2 or aspect_ratio > 1.6:
        return None, None
    if len(approx) != 4:
        card_crop = frame[y:y+h, x:x+w]
        return card_crop, (x, y, w, h)


    # Sort the points in consistent order: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    # Compute width and height of new image
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxHeight = max(int(heightA), int(heightB))

    # Perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    # Return the warped card and the bounding box in the original frame
    x, y, w, h = cv2.boundingRect(largest)
    return warp, (x, y, w, h)


# Usage of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load("card_classifier_1.pth", map_location=device)
class_names = checkpoint["class_names"]
model = SimpleCardClassifier(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Webcam loop
cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    card, bbox = crop_card(frame)
    if card is not None:
        img = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            predicted_idx = torch.argmax(outputs, 1).item()
            predicted_class = class_names[predicted_idx]
            probs = F.softmax(outputs, dim=1)
            confidence = probs[0, predicted_idx].item() * 100  # percentage

        # Draw rectangle and prediction using bbox from the original frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{predicted_class}: {confidence:.1f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Card Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
