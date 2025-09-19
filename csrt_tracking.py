import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ---------- User config ----------
CAM_INDEX = 3
DETECTION_INTERVAL = 12       # run detection every N frames
MIN_AREA_RATIO = 0.0008       # min area relative to frame area
MAX_AREA_RATIO = 0.5          # max area relative to frame area
ASPECT_MIN = 1.25             # card long/short ratio accepted
ASPECT_MAX = 1.65
OVERLAP_DIST = 40             # pixels: skip detection if near existing tracker center
CONF_THRESHOLD = 0.35         # only initialize tracker if model confidence >= this (0..1)
MARGIN_SCALE = 1.06           # expand crop corners by this factor to avoid cutting edges
DEBUG = False                 # True to show mask / debug windows
# ---------------------------------

# device & model (update to your model path/name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("card_classifier_1.pth", map_location=device)
class_names = checkpoint["class_names"]
from main import SimpleCardClassifier  # your model class
model = SimpleCardClassifier(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ---------- helpers ----------
def create_csrt_tracker():
    """Robust creation of CSRT tracker across OpenCV versions."""
    # try legacy -> old -> class.create -> raise
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT.create()
    if hasattr(cv2, "TrackerCSRT.create"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    raise RuntimeError("CSRT tracker not available: install opencv-contrib-python or use a different tracker.")

def order_points(pts):
    # pts: Nx2 array of 4 points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def expand_pts(pts, scale, frame_w, frame_h):
    center = pts.mean(axis=0)
    pts_exp = center + (pts - center) * scale
    pts_exp[:,0] = np.clip(pts_exp[:,0], 0, frame_w-1)
    pts_exp[:,1] = np.clip(pts_exp[:,1], 0, frame_h-1)
    return pts_exp

def four_point_warp(img, pts, margin_scale=MARGIN_SCALE):
    # pts: shape (4,2) float32
    h, w = img.shape[:2]
    pts = pts.astype("float32")
    pts = expand_pts(pts, margin_scale, w, h)

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp, (int(min(rect[:,0])), int(min(rect[:,1])), maxWidth, maxHeight)

# ---------- improved detect_cards ----------
def detect_cards(frame):
    """
    Returns list of tuples: (warp_bgr, bbox_on_frame)
    bbox_on_frame is (x, y, w, h) - upright bounding rect in frame coordinates
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    min_area = max(2000, frame_area * MIN_AREA_RATIO)

    # 1) white mask (HSV) - tune the thresholds if your lighting is weird
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) use masked grayscale for Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(masked_gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 120)

    # optionally dilate edges a little
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > frame_area * MAX_AREA_RATIO:
            continue

        # use rotated rectangle to handle tilt
        rect = cv2.minAreaRect(cnt)   # ((cx,cy),(w,h), angle)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # compute aspect ratio independent of rotation
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = long_side / short_side

        if not (ASPECT_MIN <= ratio <= ASPECT_MAX):
            continue

        # solidity = contour_area / bounding_rect_area
        bx, by, bw, bh = cv2.boundingRect(box.astype(np.int32))
        rect_area = bw * bh
        if rect_area <= 0:
            continue
        solidity = area / rect_area
        if solidity < 0.4:   # allow some tolerance
            continue

        # get a usable warp (perspective) - expand a little to keep edges
        warped, bbox_upright = four_point_warp(frame, box, margin_scale=MARGIN_SCALE)
        if warped is None:
            continue

        results.append((warped, bbox_upright))

    if DEBUG:
        cv2.imshow("white_mask", mask)
        cv2.imshow("edges", edges)
    return results

# ---------- main loop ----------
def main_loop():
    cap = cv2.VideoCapture(CAM_INDEX)
    trackers = []   # list of tuples (tracker_obj, label, bbox_int_tuple)
    frame_count = 0

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        # --- 1) update trackers ---
        new_trackers = []
        for trk, label, last_bbox in trackers:
            success, new_bbox = trk.update(frame)
            if success:
                x, y, w, h = map(int, new_bbox)
                # draw
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                new_trackers.append((trk, label, (x, y, w, h)))
        trackers = new_trackers

        # --- 2) periodic detection for new cards ---
        if frame_count % DETECTION_INTERVAL == 0:
            dets = detect_cards(frame)
            for warp_bgr, bbox in dets:
                x, y, w_box, h_box = bbox
                # check overlap by comparing centers with existing tracker bboxes
                ncx, ncy = x + w_box//2, y + h_box//2
                already_tracked = False
                for _, _, tb in trackers:
                    tx, ty, tw, th = tb
                    tcx, tcy = tx + tw//2, ty + th//2
                    if np.hypot(ncx - tcx, ncy - tcy) < OVERLAP_DIST:
                        already_tracked = True
                        break
                if already_tracked:
                    continue

                # classify the warped crop
                try:
                    pil = Image.fromarray(cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
                except Exception:
                    continue
                input_tensor = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    confidence = float(conf.item())
                    predicted_label = class_names[int(idx.item())]

                # optionally skip low-confidence
                if confidence < CONF_THRESHOLD:
                    if DEBUG:
                        print(f"Low confidence {confidence:.2f} for bbox {bbox}, skipping tracker creation")
                    continue

                # initialize tracker (safe creation + bbox tuple of floats)
                try:
                    tr = create_csrt_tracker()
                    # ensure bbox inside frame and positive width/height
                    fx = max(0, x)
                    fy = max(0, y)
                    fw = max(1, min(w.shape[1]-fx, w_box)) if False else w_box  # placeholder (not used)
                    # safer: clamp width/height to frame dims
                    fw = min(w_box, frame.shape[1] - fx)
                    fh = min(h_box, frame.shape[0] - fy)
                    bbox_for_init = [float(fx), float(fy), float(fw), float(fh)]
                    tr.init(frame, bbox_for_init)
                    trackers.append((tr, predicted_label, (int(fx), int(fy), int(fw), int(fh))))
                    print(f"Added tracker: {predicted_label} ({confidence:.2f}), bbox={bbox_for_init}")
                except Exception as e:
                    print("Tracker init failed:", e)

        frame_count += 1

        # show
        cv2.imshow("Card Recognition", frame)
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
