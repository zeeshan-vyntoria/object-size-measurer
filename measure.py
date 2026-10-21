"""
Measure real-world size of objects in a photo using a known-size reference object.

Usage:
    python measure.py --image path/to/photo.jpg --ref-width 3.37 --unit in

How it works:
    1. Place a reference object of known width (e.g. a credit card = 3.37in / 8.56cm)
       as the LEFTMOST object in the photo, on a plain, contrasting background.
    2. The script finds all object contours, uses the leftmost one as the reference
       to compute a pixels-per-unit ratio, then reports width/height for every
       other object found.
    3. An annotated image is saved next to the input (suffixed "_measured").
"""

import argparse
import os

import cv2
import imutils
import numpy as np
from imutils import contours as imutils_contours
from imutils import perspective
from scipy.spatial import distance as dist

_POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")


def midpoint(a, b):
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def detect_pose_landmarks(image_path, min_visibility=0.5):
    """
    Run MediaPipe Pose on the photo and return key upper-body landmark
    pixel coordinates (shoulders, elbows, wrists, hips) as a dict, or None
    if no person is confidently detected. This is how the tool tells a
    "worn by a model" photo apart from a flat-laid garment photo, and it's
    how it measures body-relative proportions on the former — contour
    geometry alone can't locate a shoulder or wrist on a person.
    """
    from mediapipe.tasks.python import vision, BaseOptions
    import mediapipe as mp

    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]

    base_options = BaseOptions(model_asset_path=_POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    landmarker = vision.PoseLandmarker.create_from_options(options)
    # Build the mp.Image from the array cv2 already decoded, rather than
    # mp.Image.create_from_file() re-reading the file with MediaPipe's own
    # native loader — that loader supports far fewer formats than OpenCV
    # (e.g. it fails outright on .webp, a common e-commerce photo format).
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]

    PL = vision.PoseLandmark
    needed = {
        "left_shoulder": PL.LEFT_SHOULDER,
        "right_shoulder": PL.RIGHT_SHOULDER,
        "left_elbow": PL.LEFT_ELBOW,
        "right_elbow": PL.RIGHT_ELBOW,
        "left_wrist": PL.LEFT_WRIST,
        "right_wrist": PL.RIGHT_WRIST,
        "left_hip": PL.LEFT_HIP,
        "right_hip": PL.RIGHT_HIP,
    }
    points = {}
    for name, landmark in needed.items():
        p = lm[landmark.value]
        if p.visibility < min_visibility:
            return None
        points[name] = (int(p.x * w), int(p.y * h))
    return points


def find_object_contours(image, min_area=800):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

    if not cnts:
        raise ValueError(
            "No object contours found. Use a plain background with good contrast, "
            "even lighting, and make sure objects are clearly separated."
        )

    cnts, _ = imutils_contours.sort_contours(cnts, method="left-to-right")
    return cnts


def measure_objects(image_path, ref_width, unit, min_area=800):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    cnts = find_object_contours(image, min_area=min_area)
    output = image.copy()
    pixels_per_unit = None
    results = []

    for i, c in enumerate(cnts):
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        cv2.drawContours(output, [box.astype("int")], -1, (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # height in pixels
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # width in pixels

        if pixels_per_unit is None:
            # First (leftmost) contour is the reference object.
            pixels_per_unit = dB / ref_width
            label = "REFERENCE"
            dim_w, dim_h = ref_width, dA / pixels_per_unit
        else:
            label = f"Object {i}"
            dim_w = dB / pixels_per_unit
            dim_h = dA / pixels_per_unit

        results.append({"label": label, "width": dim_w, "height": dim_h})

        text = f"{dim_w:.2f} x {dim_h:.2f} {unit}"
        cv2.putText(
            output,
            text,
            (int(tl[0]), int(tl[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return results, output


def _calibrate_from_reference(ref_cnt, ref_width):
    box = cv2.minAreaRect(ref_cnt)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    ref_width_px = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    return ref_width_px / ref_width


def _label(img, text, x, y, color=(0, 0, 255), scale=0.6, thickness=2):
    """Draw text with a white background box behind it so it stays readable over any photo."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 4
    x = max(x, pad)
    y = max(y, th + pad)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _row_width(mask, row, top_y, bottom_y):
    """Widest run of foreground pixels on a row, searching nearby rows if empty."""
    for delta in range(0, 20):
        for r in {row - delta, row + delta}:
            if top_y <= r <= bottom_y:
                xs = np.where(mask[r] == 255)[0]
                if xs.size:
                    return xs.max() - xs.min(), xs.min(), xs.max(), r
    return 0, 0, 0, row


def _clean_garment_contour(mask):
    """Re-extract a smooth external contour from the filled mask (drops Canny jitter)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = max(cnts, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(clean, True)
    return cv2.approxPolyDP(clean, epsilon, True)


def _find_armpit_notches(contour, top_y, height_px, center_x):
    """
    Locate the two deepest concave points (armpit notches) on either side of
    center via convexity defects. Returns (left_y, right_y) or None if the
    contour is too noisy / no clear notches were found.
    """
    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hull_idx = np.unique(hull_idx.flatten())
        if len(hull_idx) < 3:
            return None
        defects = cv2.convexityDefects(contour, hull_idx.reshape(-1, 1))
    except cv2.error:
        return None
    if defects is None:
        return None
    defects = defects.reshape(-1, 4)

    left_best = right_best = None
    for i in range(defects.shape[0]):
        _, _, f, depth = defects[i]
        fx, fy = contour[f][0]
        if not (top_y + 0.1 * height_px <= fy <= top_y + 0.65 * height_px):
            continue
        margin = 0.05 * height_px
        if fx < center_x - margin:
            if left_best is None or depth > left_best[0]:
                left_best = (depth, fy)
        elif fx > center_x + margin:
            if right_best is None or depth > right_best[0]:
                right_best = (depth, fy)

    if left_best is None or right_best is None:
        return None
    return int(left_best[1]), int(right_best[1])


def _find_neckline(mask, top_y, bottom_y, center_x, height_px, search_frac=0.3):
    """
    Find the neckline opening by scanning top rows for a gap (background)
    between the two shoulder humps, centered near the garment's midline.
    Returns (gap_px, row, x1, x2) or None if no clear notch was found.
    """
    best = None
    end_row = min(top_y + int(search_frac * height_px), bottom_y)
    for row in range(top_y, end_row + 1):
        xs = np.where(mask[row] == 255)[0]
        if xs.size < 2:
            continue
        diffs = np.diff(xs)
        idx = np.argmax(diffs)
        gap = diffs[idx]
        if gap <= 2:
            continue
        gap_start, gap_end = int(xs[idx]), int(xs[idx + 1])
        gap_center = (gap_start + gap_end) / 2
        if abs(gap_center - center_x) > 0.25 * (xs.max() - xs.min() + 1):
            continue
        if best is None or gap > best[0]:
            best = (gap, row, gap_start, gap_end)
    return best


def _sleeve_length(mask, top_y, bottom_y, left_shoulder, right_shoulder, search_frac=0.85, tol_px=3):
    """
    Approx straight-line sleeve length: shoulder point to farthest sleeve edge.

    Searches most of the garment's height (not just down to the armpit) since
    a long sleeve laid flat usually runs alongside the torso well past the
    underarm level, often close to hem height — stopping the search at the
    armpit (as an earlier version of this did) finds only the shoulder cap's
    width and badly undercounts long sleeves. The bottom ~15% is still
    excluded to avoid picking up a flared hem as a "sleeve tip".

    Finds the sleeve's outer edge first (the widest points on each side,
    which is where a sleeve reaches beyond the body), then — since a
    straight sleeve edge often holds nearly that same extreme x across many
    rows — takes the lowest of those rows as the cuff. Simply maximizing raw
    distance from the shoulder point (an earlier version of this did) picks
    up a low torso/hem point instead, since its extra vertical distance can
    outweigh a sleeve tip's smaller x offset even though it isn't part of
    the sleeve at all.
    """
    height_px = bottom_y - top_y
    search_bottom = min(bottom_y, top_y + int(search_frac * height_px))
    left_edge, right_edge = [], []
    for row in range(top_y, search_bottom + 1):
        xs = np.where(mask[row] == 255)[0]
        if xs.size == 0:
            continue
        left_edge.append((row, int(xs.min())))
        right_edge.append((row, int(xs.max())))
    if not left_edge or not right_edge:
        return None, None, None

    global_min_x = min(lx for _, lx in left_edge)
    global_max_x = max(rx for _, rx in right_edge)
    left_row, left_x = max((r for r in left_edge if r[1] <= global_min_x + tol_px), key=lambda r: r[0])
    right_row, right_x = max((r for r in right_edge if r[1] >= global_max_x - tol_px), key=lambda r: r[0])
    left_tip = (left_x, left_row)
    right_tip = (right_x, right_row)

    left_len = dist.euclidean(left_shoulder, left_tip)
    right_len = dist.euclidean(right_shoulder, right_tip)
    return (left_len + right_len) / 2, left_tip, right_tip


# Flat garment width x2 approximates circumference since a garment laid flat
# is folded front-to-back — this is the standard convention for chest/waist
# spec sheets. The neck opening isn't a full body cross-section the same way,
# so its circumference estimate is rougher; flagged as such in the result.
CIRCUMFERENCE_FACTOR = 2.0


# Adult short-sleeve tee size chart, taken from a real published spec sheet
# (a "Pro Fit" sizing chart). chest/bottom_width/length/sleeve are as-listed
# on that chart — note chest and bottom_width there are already the FLAT
# (pit-to-pit / hem-edge) measurement, not a circumference; a real spec
# sheet reports "Chest: 22in" meaning the flat width, same convention this
# tool uses elsewhere for chest_width_flat. shoulder/neck aren't on that
# chart (a basic tee doesn't call those out) so they're rough placeholders,
# clearly the weakest part of this fallback. Used ONLY when no reference
# object is in the photo — actual garments still vary by brand and cut.
GENERIC_SHIRT_SIZE_CHART = {
    "S": {"chest_flat": 20, "length": 27.5, "bottom_width_flat": 20, "sleeve": 7.5, "shoulder": 16.5, "neck": 14.5},
    "M": {"chest_flat": 22, "length": 28.5, "bottom_width_flat": 22, "sleeve": 8, "shoulder": 17.5, "neck": 15.5},
    "L": {"chest_flat": 24, "length": 29.5, "bottom_width_flat": 24, "sleeve": 8.5, "shoulder": 18.5, "neck": 16.5},
    "XL": {"chest_flat": 26, "length": 30.5, "bottom_width_flat": 26, "sleeve": 9, "shoulder": 19.5, "neck": 17.5},
    "XXL": {"chest_flat": 27, "length": 31.5, "bottom_width_flat": 27, "sleeve": 9.5, "shoulder": 20.5, "neck": 18.5},
}


def _closest_chart_size(ratios, ratio_keys):
    """
    Find the GENERIC_SHIRT_SIZE_CHART size whose own ratios are closest
    (least squared distance) to the given detected ratios.

    ratios: {"chest": 0.7, ...} — detected, unitless proportions.
    ratio_keys: {"chest": ("chest_flat", "length"), ...} — for each ratio
      name, which two chart fields to divide to get that chart's version of
      the same ratio.
    """
    best_label, best_dist = None, None
    for label, entry in GENERIC_SHIRT_SIZE_CHART.items():
        dist_sq = 0.0
        for name, (num_key, den_key) in ratio_keys.items():
            chart_ratio = entry[num_key] / entry[den_key]
            dist_sq += (ratios[name] - chart_ratio) ** 2
        if best_dist is None or dist_sq < best_dist:
            best_dist, best_label = dist_sq, label
    return best_label


def estimate_shirt_size_no_reference(
    image_path, min_area=800, shoulder_pct=0.16, chest_pct=0.35
):
    """
    AI ESTIMATE ONLY. With no reference object in the photo there is no way
    to derive the garment's true real-world size — this instead reads the
    garment's on-photo proportions (chest width and shoulder width relative
    to length) and matches them against GENERIC_SHIRT_SIZE_CHART to guess
    the closest standard size, returning that size's typical measurements.
    Treat the result as a rough guess, never as a measurement.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    cnts = find_object_contours(image, min_area=min_area)
    garment_cnt = max(cnts, key=cv2.contourArea)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [garment_cnt], -1, 255, -1)
    ys, xs = np.where(mask == 255)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    height_px = bottom_y - top_y
    center_x = int((xs.min() + xs.max()) / 2)

    clean_cnt = _clean_garment_contour(mask)

    shoulder_row_guess = top_y + int(shoulder_pct * height_px)
    _, sh_x1, sh_x2, shoulder_row = _row_width(mask, shoulder_row_guess, top_y, bottom_y)

    notches = _find_armpit_notches(clean_cnt, top_y, height_px, center_x)
    chest_row = int((notches[0] + notches[1]) / 2) if notches else top_y + int(chest_pct * height_px)
    chest_px, ch_x1, ch_x2, chest_row = _row_width(mask, chest_row, top_y, bottom_y)

    detected_chest_ratio = chest_px / height_px
    detected_shoulder_ratio = (sh_x2 - sh_x1) / height_px

    best_label = _closest_chart_size(
        {"chest": detected_chest_ratio, "shoulder": detected_shoulder_ratio},
        {"chest": ("chest_flat", "length"), "shoulder": ("shoulder", "length")},
    )

    chart = GENERIC_SHIRT_SIZE_CHART[best_label]
    result = {
        "photo_type": "flat",
        "estimated_size_label": best_label,
        "estimated_chest_flat_in": chart["chest_flat"],
        "estimated_chest_circumference_in": chart["chest_flat"] * CIRCUMFERENCE_FACTOR,
        "estimated_length_in": chart["length"],
        "estimated_bottom_width_flat_in": chart["bottom_width_flat"],
        "estimated_bottom_width_circumference_in": chart["bottom_width_flat"] * CIRCUMFERENCE_FACTOR,
        "estimated_shoulder_width_in": chart["shoulder"],
        "estimated_sleeve_length_in": chart["sleeve"],
        "estimated_neck_circumference_in": chart["neck"],
        "armpit_detected": bool(notches),
    }

    output = image.copy()
    cv2.drawContours(output, [garment_cnt], -1, (0, 200, 0), 2)

    _label(
        output,
        f"Estimated size: {best_label}  (rough guess, no reference object)",
        int(xs.min()),
        top_y - 14,
        color=(180, 0, 180),
        scale=0.75,
        thickness=2,
    )

    cv2.line(output, (center_x, shoulder_row), (center_x, bottom_y), (255, 165, 0), 2)
    _label(
        output,
        f"Length ~{chart['length']}in",
        center_x + 10,
        (shoulder_row + bottom_y) // 2,
        color=(255, 140, 0),
    )

    for (x1, x2, row, name, value_text) in (
        (sh_x1, sh_x2, shoulder_row, "Shoulder", f"~{chart['shoulder']}in"),
        (ch_x1, ch_x2, chest_row, "Chest", f"~{chart['chest_flat']}in flat"),
    ):
        cv2.line(output, (x1, row), (x2, row), (0, 0, 255), 2)
        _label(output, f"{name} {value_text}", x1, row - 10, color=(0, 0, 255))

    left_shoulder, right_shoulder = (sh_x1, shoulder_row), (sh_x2, shoulder_row)
    _, left_tip, right_tip = _sleeve_length(mask, top_y, bottom_y, left_shoulder, right_shoulder)
    if left_tip and right_tip:
        cv2.line(output, left_shoulder, left_tip, (200, 0, 200), 2)
        cv2.line(output, right_shoulder, right_tip, (200, 0, 200), 2)
        _label(
            output,
            f"Arm/sleeve ~{chart['sleeve']}in",
            left_tip[0],
            left_tip[1] + 20,
            color=(200, 0, 200),
        )

    return result, output


def estimate_worn_shirt_size(image_path, points):
    """
    AI ESTIMATE ONLY, for a photo of a person WEARING the shirt (rather than
    laid flat). There's no reference object and no reliable body-size input
    here either, so real-world size can't be derived — same hard limit as
    estimate_shirt_size_no_reference, just for a different photo type.

    Uses detected body pose landmarks (shoulders, elbows, wrists, hips) to
    read body proportions — shoulder width and arm length, both relative to
    torso length — and matches them against GENERIC_SHIRT_SIZE_CHART's own
    ratios to guess the closest standard size. Treat the result as a rough
    guess, never as a measurement: it reads the WEARER's build, not the
    garment's actual cut, so a loose or fitted style will read the same way.

    `points` comes from detect_pose_landmarks() — pass it in rather than
    detecting again, since the caller typically already ran it to decide
    this is a worn photo in the first place.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    ls, rs = points["left_shoulder"], points["right_shoulder"]
    le, re = points["left_elbow"], points["right_elbow"]
    lw, rw = points["left_wrist"], points["right_wrist"]
    lh, rh = points["left_hip"], points["right_hip"]

    shoulder_width_px = dist.euclidean(ls, rs)
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    torso_length_px = dist.euclidean(shoulder_mid, hip_mid)
    if torso_length_px <= 0:
        raise ValueError("Could not determine body proportions from the detected pose.")

    left_arm_px = dist.euclidean(ls, le) + dist.euclidean(le, lw)
    right_arm_px = dist.euclidean(rs, re) + dist.euclidean(re, rw)
    arm_length_px = (left_arm_px + right_arm_px) / 2

    shoulder_ratio = shoulder_width_px / torso_length_px
    sleeve_ratio = arm_length_px / torso_length_px

    best_label = _closest_chart_size(
        {"shoulder": shoulder_ratio, "sleeve": sleeve_ratio},
        {"shoulder": ("shoulder", "length"), "sleeve": ("sleeve", "length")},
    )
    chart = GENERIC_SHIRT_SIZE_CHART[best_label]

    result = {
        "photo_type": "worn",
        "estimated_size_label": best_label,
        "estimated_chest_flat_in": chart["chest_flat"],
        "estimated_chest_circumference_in": chart["chest_flat"] * CIRCUMFERENCE_FACTOR,
        "estimated_length_in": chart["length"],
        "estimated_bottom_width_flat_in": chart["bottom_width_flat"],
        "estimated_bottom_width_circumference_in": chart["bottom_width_flat"] * CIRCUMFERENCE_FACTOR,
        "estimated_shoulder_width_in": chart["shoulder"],
        "estimated_sleeve_length_in": chart["sleeve"],
        "estimated_neck_circumference_in": chart["neck"],
    }

    output = image.copy()
    for a, b, color, thickness in (
        (ls, rs, (0, 0, 255), 2),
        (ls, le, (200, 0, 200), 2),
        (le, lw, (200, 0, 200), 2),
        (rs, re, (200, 0, 200), 2),
        (re, rw, (200, 0, 200), 2),
        (shoulder_mid, hip_mid, (255, 165, 0), 2),
    ):
        a_i, b_i = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        cv2.line(output, a_i, b_i, color, thickness)
    for p in points.values():
        cv2.circle(output, p, 5, (0, 200, 0), -1)

    _label(
        output,
        f"Estimated size: {best_label}  (rough guess, worn photo — no reference)",
        20,
        30,
        color=(180, 0, 180),
        scale=0.7,
    )
    _label(output, f"Shoulder ~{chart['shoulder']}in", ls[0], ls[1] - 15, color=(0, 0, 255))
    _label(output, f"Arm/sleeve ~{chart['sleeve']}in", int(lw[0]), int(lw[1]) + 22, color=(200, 0, 200))

    return result, output


def measure_shirt(
    image_path,
    ref_width,
    unit,
    min_area=800,
    shoulder_pct=0.16,
    chest_pct=0.35,
    waist_pct=0.7,
):
    """
    Approximate garment measurements for a shirt laid flat, calibrated
    against a reference object (leftmost contour in the photo). Anchor
    points follow standard size-chart convention:

    - length: shoulder line to hem, straight down the center (not from the
      topmost pixel — that would include a tall/popped collar's height)
    - shoulder_width: at shoulder_pct down from the top (fraction of garment
      height) — auto-detecting the exact shoulder seam corner via contour
      geometry alone proved unreliable (it's easily confused with the
      neckline or sleeve-tip corners), so this stays percentage-based;
      tune shoulder_pct if a photo comes out off
    - chest_width_flat / chest_circumference: pit-to-pit, doubled
    - waist_width_flat / waist_circumference: at waist_pct down, doubled
    - bottom_width_flat / bottom_width_circumference: hem edge, doubled
    - sleeve_length: shoulder point to sleeve edge (short sleeves are more
      reliable than long, since long sleeves often fold/bend in photos)
    - sleeve_with_shoulder_length: back-neck point to sleeve edge — a second
      sleeve convention some size charts use instead of (or alongside)
      sleeve_length
    - neck_width_flat / neck_circumference_est: from the neckline notch, if
      one is clearly detected

    Chest row is found via convexity-defect armpit detection when possible,
    falling back to chest_pct (fraction of garment height) if the contour
    is too noisy for that.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    cnts = find_object_contours(image, min_area=min_area)
    if len(cnts) < 2:
        raise ValueError(
            "Need both a reference object and the shirt in frame — only one "
            "contour was found."
        )

    ref_cnt = cnts[0]
    garment_cnt = max(cnts[1:], key=cv2.contourArea)
    pixels_per_unit = _calibrate_from_reference(ref_cnt, ref_width)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [garment_cnt], -1, 255, -1)
    ys, xs = np.where(mask == 255)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    height_px = bottom_y - top_y
    center_x = int((xs.min() + xs.max()) / 2)

    clean_cnt = _clean_garment_contour(mask)
    neck = _find_neckline(mask, top_y, bottom_y, center_x, height_px)

    shoulder_row_guess = top_y + int(shoulder_pct * height_px)
    _, sh_x1, sh_x2, shoulder_row = _row_width(mask, shoulder_row_guess, top_y, bottom_y)
    left_shoulder, right_shoulder = (sh_x1, shoulder_row), (sh_x2, shoulder_row)
    shoulder_width_px = sh_x2 - sh_x1

    notches = _find_armpit_notches(clean_cnt, top_y, height_px, center_x)
    if notches:
        chest_row = int((notches[0] + notches[1]) / 2)
    else:
        chest_row = top_y + int(chest_pct * height_px)
    chest_px, ch_x1, ch_x2, chest_row = _row_width(mask, chest_row, top_y, bottom_y)

    waist_row_guess = top_y + int(waist_pct * height_px)
    waist_px, w_x1, w_x2, waist_row = _row_width(mask, waist_row_guess, top_y, bottom_y)

    bottom_row_guess = bottom_y - max(1, int(0.02 * height_px))
    bottom_px, b_x1, b_x2, bottom_row = _row_width(mask, bottom_row_guess, top_y, bottom_y)

    sleeve_px, left_tip, right_tip = _sleeve_length(mask, top_y, bottom_y, left_shoulder, right_shoulder)

    # "Sleeve" (above) runs shoulder seam to cuff. Size charts also commonly
    # show "Sleeve with shoulder" — center-back neck to cuff, i.e. including
    # the shoulder-width portion too — as a second convention. Approximated
    # here from (center_x, top_y), since that's close to the back-neck point
    # on a flat, symmetric lay.
    neck_point = (center_x, top_y)
    if left_tip and right_tip:
        sleeve_with_shoulder_px = (
            dist.euclidean(neck_point, left_tip) + dist.euclidean(neck_point, right_tip)
        ) / 2
    else:
        sleeve_with_shoulder_px = None

    # Length runs from the shoulder line down to the hem, not from the
    # topmost pixel of the whole silhouette — a shirt with a tall/popped
    # collar has its topmost point at the collar tip, well above the true
    # shoulder seam, which would otherwise make Length include the collar's
    # height on top of the real back-length.
    length_px = bottom_y - shoulder_row

    result = {
        "photo_type": "flat",
        "length": length_px / pixels_per_unit,
        "shoulder_width": shoulder_width_px / pixels_per_unit,
        "chest_width_flat": chest_px / pixels_per_unit,
        "chest_circumference": chest_px / pixels_per_unit * CIRCUMFERENCE_FACTOR,
        "waist_width_flat": waist_px / pixels_per_unit,
        "waist_circumference": waist_px / pixels_per_unit * CIRCUMFERENCE_FACTOR,
        "bottom_width_flat": bottom_px / pixels_per_unit,
        "bottom_width_circumference": bottom_px / pixels_per_unit * CIRCUMFERENCE_FACTOR,
        "sleeve_length": sleeve_px / pixels_per_unit if sleeve_px is not None else None,
        "sleeve_with_shoulder_length": (
            sleeve_with_shoulder_px / pixels_per_unit if sleeve_with_shoulder_px is not None else None
        ),
        "neck_width_flat": neck[0] / pixels_per_unit if neck else None,
        "neck_circumference_est": neck[0] / pixels_per_unit * CIRCUMFERENCE_FACTOR if neck else None,
        "armpit_detected": bool(notches),
    }

    output = image.copy()
    cv2.drawContours(output, [ref_cnt], -1, (255, 0, 0), 2)
    cv2.drawContours(output, [garment_cnt], -1, (0, 255, 0), 2)

    cv2.line(output, (center_x, shoulder_row), (center_x, bottom_y), (255, 165, 0), 2)
    _label(
        output,
        f"Length {result['length']:.2f}{unit}",
        center_x + 10,
        (shoulder_row + bottom_y) // 2,
        color=(255, 140, 0),
    )
    cv2.line(output, left_shoulder, right_shoulder, (0, 0, 255), 2)
    _label(
        output,
        f"Shoulder {result['shoulder_width']:.2f}{unit}",
        left_shoulder[0],
        left_shoulder[1] - 10,
        color=(0, 0, 255),
    )
    for (x1, x2, row, label, value) in (
        (ch_x1, ch_x2, chest_row, "Chest (flat)", result["chest_width_flat"]),
        (w_x1, w_x2, waist_row, "Waist (flat)", result["waist_width_flat"]),
        (b_x1, b_x2, bottom_row, "Bottom width (flat)", result["bottom_width_flat"]),
    ):
        cv2.line(output, (x1, row), (x2, row), (0, 0, 255), 2)
        _label(output, f"{label} {value:.2f}{unit}", x1, row - 10, color=(0, 0, 255))
    if neck:
        _, neck_row, nx1, nx2 = neck
        cv2.line(output, (nx1, neck_row), (nx2, neck_row), (0, 200, 200), 2)
        _label(
            output,
            f"Neck {result['neck_width_flat']:.2f}{unit}",
            nx1,
            neck_row + 24,
            color=(0, 150, 150),
        )
    if left_tip and right_tip:
        cv2.line(output, left_shoulder, left_tip, (200, 0, 200), 2)
        cv2.line(output, right_shoulder, right_tip, (200, 0, 200), 2)
        _label(
            output,
            f"Sleeve {result['sleeve_length']:.2f}{unit}",
            left_tip[0],
            left_tip[1] + 20,
            color=(200, 0, 200),
        )
        cv2.line(output, neck_point, right_tip, (120, 70, 0), 1)
        _label(
            output,
            f"Sleeve+shoulder {result['sleeve_with_shoulder_length']:.2f}{unit}",
            right_tip[0] - 160,
            right_tip[1] - 10,
            color=(120, 70, 0),
            scale=0.5,
        )

    return result, output


_NO_REFERENCE_ERROR = "only one contour was found"


def measure_shirt_auto(
    image_path,
    ref_width,
    unit,
    min_area=800,
    shoulder_pct=0.16,
    chest_pct=0.35,
    waist_pct=0.7,
):
    """
    Single entry point for clothing measurement, auto-routed by what the
    photo actually contains — the caller doesn't need to know in advance:

    1. A person is confidently detected wearing the shirt (pose landmarks
       found) -> estimate_worn_shirt_size (rough guess from body
       proportions; a flat-lay contour approach can't locate a shoulder or
       wrist on someone, so this needs pose detection instead).
    2. Otherwise, a reference object is also visible alongside the flat-laid
       shirt -> measure_shirt (real, calibrated numbers).
    3. Otherwise, just the flat-laid shirt alone -> estimate_shirt_size_no_reference
       (rough size-chart guess).

    This replaces an earlier design that required picking the right mode
    for the photo, which was a common source of confusing "only one
    contour" errors when a worn or reference-less photo hit the wrong path.

    Returns (result, output) where result always includes "accuracy"
    ("measured" or "estimated") and "photo_type" ("flat" or "worn").
    """
    pose_points = detect_pose_landmarks(image_path)
    if pose_points is not None:
        result, output = estimate_worn_shirt_size(image_path, pose_points)
        return {"accuracy": "estimated", **result}, output

    try:
        result, output = measure_shirt(image_path, ref_width, unit, min_area, shoulder_pct, chest_pct, waist_pct)
        return {"accuracy": "measured", **result}, output
    except ValueError as e:
        if _NO_REFERENCE_ERROR not in str(e):
            raise
        result, output = estimate_shirt_size_no_reference(image_path, min_area, shoulder_pct, chest_pct)
        return {"accuracy": "estimated", **result}, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--ref-width",
        type=float,
        help="Known width of the reference object (leftmost object in image), "
        "e.g. 3.37 for a credit card in inches, or 8.56 in cm. Required "
        "unless --mode ai-estimate",
    )
    parser.add_argument("--unit", default="in", help="Unit label to display (e.g. in, cm)")
    parser.add_argument(
        "--min-area",
        type=float,
        default=800,
        help="Minimum contour area to consider as an object (filters noise)",
    )
    parser.add_argument(
        "--mode",
        choices=["object", "shirt", "ai-estimate"],
        default="object",
        help="'object' measures every contour's bounding box; 'shirt' reports "
        "garment length/shoulder/chest width for a flat-laid shirt (needs a "
        "reference object in frame); 'ai-estimate' guesses a size label from "
        "a bare photo with no reference object (rough estimate only)",
    )
    args = parser.parse_args()

    if args.mode != "ai-estimate" and args.ref_width is None:
        parser.error("--ref-width is required unless --mode ai-estimate")

    if args.mode == "ai-estimate":
        result, output = estimate_shirt_size_no_reference(args.image, args.min_area)
        print(f"Estimated size: {result['estimated_size_label']} (rough guess, no reference object)")
        print(
            f"  Chest: ~{result['estimated_chest_flat_in']} in flat "
            f"(~{result['estimated_chest_circumference_in']} in circumference)"
        )
        print(f"  Length: ~{result['estimated_length_in']} in")
        print(
            f"  Bottom width: ~{result['estimated_bottom_width_flat_in']} in flat "
            f"(~{result['estimated_bottom_width_circumference_in']} in circumference)"
        )
        print(f"  Shoulder width: ~{result['estimated_shoulder_width_in']} in")
        print(f"  Sleeve length: ~{result['estimated_sleeve_length_in']} in")
        print(f"  Neck circumference: ~{result['estimated_neck_circumference_in']} in")
    elif args.mode == "shirt":
        result, output = measure_shirt(args.image, args.ref_width, args.unit, args.min_area)
        print(f"Length: {result['length']:.2f} {args.unit}")
        print(f"Shoulder width: {result['shoulder_width']:.2f} {args.unit}")
        print(
            f"Chest: {result['chest_width_flat']:.2f} {args.unit} flat "
            f"(~{result['chest_circumference']:.2f} {args.unit} circumference)"
        )
        print(
            f"Waist: {result['waist_width_flat']:.2f} {args.unit} flat "
            f"(~{result['waist_circumference']:.2f} {args.unit} circumference)"
        )
        print(
            f"Bottom width: {result['bottom_width_flat']:.2f} {args.unit} flat "
            f"(~{result['bottom_width_circumference']:.2f} {args.unit} circumference)"
        )
        if result["sleeve_length"] is not None:
            print(f"Sleeve length (approx): {result['sleeve_length']:.2f} {args.unit}")
        if result["sleeve_with_shoulder_length"] is not None:
            print(
                f"Sleeve with shoulder (approx): "
                f"{result['sleeve_with_shoulder_length']:.2f} {args.unit}"
            )
        if result["neck_width_flat"] is not None:
            print(
                f"Neck: {result['neck_width_flat']:.2f} {args.unit} flat "
                f"(~{result['neck_circumference_est']:.2f} {args.unit} circumference, rough estimate)"
            )
        else:
            print("Neck: not detected")
    else:
        results, output = measure_objects(args.image, args.ref_width, args.unit, args.min_area)
        print(f"Found {len(results)} object(s):")
        for r in results:
            print(f"  {r['label']}: {r['width']:.2f} x {r['height']:.2f} {args.unit}")

    out_path = args.image.rsplit(".", 1)
    out_path = out_path[0] + "_measured." + (out_path[1] if len(out_path) > 1 else "jpg")
    cv2.imwrite(out_path, output)
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
