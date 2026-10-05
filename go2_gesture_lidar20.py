#!/usr/bin/env python3

import cv2
import time
import math
import json
import threading
import numpy as np
import mediapipe as mp


from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

HAND_MODEL = "/home/srge/go2_hand_models/hand_landmarker.task"
FRAME_PATH = "/tmp/d435i_frame_latest.jpg"

WINDOW_NAME = "Go2 Gesture - LiDAR20"


# ============================================================
# NORMAL TRACKING
# ============================================================

MIN_HAND_DETECTION_CONF = 0.25
MIN_HAND_PRESENCE_CONF = 0.25
MIN_HAND_TRACKING_CONF = 0.25


# ============================================================
# REACQUISITION
# ============================================================

SEARCH_DETECTION_CONF = 0.32
SEARCH_PRESENCE_CONF = 0.32

REACQUISITION_INTERVAL = 0.16

MAX_RESULT_AGE = 0.25

# Keep old hand alive during tiny detector failures
HAND_TIMEOUT = 0.50


# ============================================================
# LOCAL SEARCH
# ============================================================

# Search around last known hand location first.
LOCAL_SEARCH_INTERVAL = 0.08

LOCAL_SEARCH_RADIUS_X = 0.28
LOCAL_SEARCH_RADIUS_Y = 0.30


# ============================================================
# GESTURE STABILITY
# ============================================================

GESTURE_CONFIRMATIONS = 2

COMMAND_COOLDOWN = 0.75


# ============================================================
# MOTION GESTURES
# ============================================================

MOTION_DISTANCE = 0.12
MOTION_WINDOW = 0.55

POSE_COMMAND_COOLDOWN = 2.0

PALM_UP_THRESHOLD = -0.30
PALM_DOWN_THRESHOLD = 0.30


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode


# ============================================================
# DATA
# ============================================================

@dataclass
class HandResult:

    landmarks: list
    timestamp: float
    source: str


# ============================================================
# IMAGE PROCESSING
# ============================================================

def increase_contrast(
    img,
    alpha=1.16,
    beta=0
):

    return cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=beta
    )


# ============================================================
# LANDMARK VALIDATION
# ============================================================

def landmark_bbox(
    landmarks,
    width,
    height
):

    xs = [
        p.x for p in landmarks
    ]

    ys = [
        p.y for p in landmarks
    ]

    return (
        int(min(xs) * width),
        int(min(ys) * height),
        int(max(xs) * width),
        int(max(ys) * height)
    )


def bbox_area(
    landmarks,
    width,
    height
):

    x1, y1, x2, y2 = landmark_bbox(
        landmarks,
        width,
        height
    )

    return max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )


def valid_landmarks(
    landmarks,
    width,
    height
):

    if (
        landmarks is None
        or len(landmarks) != 21
    ):
        return False

    for p in landmarks:

        if not (
            math.isfinite(p.x)
            and
            math.isfinite(p.y)
            and
            math.isfinite(p.z)
        ):
            return False

    area = bbox_area(
        landmarks,
        width,
        height
    )

    if area < 30 * 30:
        return False

    # Don't accept completely impossible positions.
    for p in landmarks:

        if (
            p.x < -0.15
            or p.x > 1.15
            or p.y < -0.15
            or p.y > 1.15
        ):
            return False

    return True


# ============================================================
# HAND CENTER
# ============================================================

def hand_center(landmarks):

    xs = [
        p.x for p in landmarks
    ]

    ys = [
        p.y for p in landmarks
    ]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys)
    )


def hand_size(landmarks):

    xs = [
        p.x for p in landmarks
    ]

    ys = [
        p.y for p in landmarks
    ]

    return max(
        max(xs) - min(xs),
        max(ys) - min(ys)
    )


# ============================================================
# TRACKING CONTINUITY
# ============================================================

def spatially_plausible(
    new_hand,
    old_hand
):

    if (
        new_hand is None
        or old_hand is None
    ):
        return True

    nx, ny = hand_center(
        new_hand
    )

    ox, oy = hand_center(
        old_hand
    )

    movement = math.sqrt(
        (nx - ox) ** 2
        +
        (ny - oy) ** 2
    )

    # A hand shouldn't teleport across
    # half the camera in one frame.
    if movement > 0.42:
        return False

    return True


# ============================================================
# GEOMETRY
# ============================================================

def dist(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2
        +
        (a.y - b.y) ** 2
        +
        (a.z - b.z) ** 2
    )


def angle(a, b, c):

    ba = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    denom = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denom < 1e-8:
        return 180.0

    cosine = (
        np.dot(ba, bc)
        /
        denom
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )



# ============================================================
# D435i PERSON TRACKING
# ============================================================

PERSON_XYZ_FILE = "/tmp/go2_person_xyz.json"


def get_d435i_person_xyz():

    try:

        with open(PERSON_XYZ_FILE, "r") as f:
            data = json.load(f)

        if not data.get("tracking", False):
            return None

        x = data.get("x")
        y = data.get("y")
        z = data.get("z")

        if x is None or y is None or z is None:
            return None

        return float(x), float(y), float(z)

    except Exception:
        return None


# ============================================================
# GESTURE CLASSIFIER
# ============================================================

def classify_gesture(lm):

    if lm is None or len(lm) != 21:
        return "NONE"

    def finger_straight(mcp, pip, tip):

        a = angle(
            lm[mcp],
            lm[pip],
            lm[tip]
        )

        # Slightly more tolerant than the original 145 degrees.
        return a > 135.0

    index_straight = finger_straight(5, 6, 8)
    middle_straight = finger_straight(9, 10, 12)
    ring_straight = finger_straight(13, 14, 16)
    pinky_straight = finger_straight(17, 18, 20)

    # ------------------------------------------------
    # FOLLOW
    # Index + middle extended, ring + pinky curled.
    # ------------------------------------------------

    if (
        index_straight
        and middle_straight
        and not ring_straight
        and not pinky_straight
    ):

        index_tip = np.array([
            lm[8].x,
            lm[8].y
        ])

        middle_tip = np.array([
            lm[12].x,
            lm[12].y
        ])

        finger_gap = np.linalg.norm(
            index_tip - middle_tip
        )

        if finger_gap < 0.22:
            return "FOLLOW"

    # ------------------------------------------------
    # OPEN
    # ------------------------------------------------

    if (
        index_straight
        and middle_straight
        and ring_straight
        and pinky_straight
    ):

        return "OPEN"

    # ------------------------------------------------
    # POINT
    # ------------------------------------------------

    if (
        index_straight
        and not middle_straight
        and not ring_straight
        and not pinky_straight
    ):

        return "POINT"

    # ------------------------------------------------
    # FIST
    # ------------------------------------------------

    if (
        not index_straight
        and not middle_straight
        and not ring_straight
        and not pinky_straight
    ):

        return "FIST"

    return "NONE"


class GestureStabilizer:

    def __init__(
        self,
        confirmations=2
    ):

        self.confirmations = (
            confirmations
        )

        self.last_raw = "NONE"
        self.count = 0
        self.stable = "NONE"

    def update(
        self,
        gesture
    ):

        if gesture == self.last_raw:

            self.count += 1

        else:

            self.last_raw = gesture
            self.count = 1

        if (
            self.count
            >= self.confirmations
        ):

            self.stable = gesture

        return self.stable

    def reset(self):

        self.last_raw = "NONE"
        self.count = 0
        self.stable = "NONE"


# ============================================================
# NORMAL DETECTOR
# ============================================================

def create_video_detector():

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=HAND_MODEL
        ),

        running_mode=RunningMode.VIDEO,

        num_hands=1,

        min_hand_detection_confidence=(
            MIN_HAND_DETECTION_CONF
        ),

        min_hand_presence_confidence=(
            MIN_HAND_PRESENCE_CONF
        ),

        min_tracking_confidence=(
            MIN_HAND_TRACKING_CONF
        ),
    )

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# SEARCH DETECTOR
# ============================================================

def create_search_detector():

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=HAND_MODEL
        ),

        running_mode=RunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=(
            SEARCH_DETECTION_CONF
        ),

        min_hand_presence_confidence=(
            SEARCH_PRESENCE_CONF
        ),
    )

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(
    detector,
    image
):

    if (
        image is None
        or image.size == 0
    ):
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    try:

        result = detector.detect(
            mp_image
        )

    except Exception:

        return None

    if not result.hand_landmarks:

        return None

    lm = result.hand_landmarks[0]

    h, w = image.shape[:2]

    if not valid_landmarks(
        lm,
        w,
        h
    ):

        return None

    return lm


# ============================================================
# CROP
# ============================================================

def make_crop(
    frame,
    cx,
    cy,
    crop_w,
    crop_h,
    scale
):

    h, w = frame.shape[:2]

    half_w = (
        crop_w
        /
        (2.0 * scale)
    )

    half_h = (
        crop_h
        /
        (2.0 * scale)
    )

    x1 = int(
        cx - half_w
    )

    y1 = int(
        cy - half_h
    )

    x2 = int(
        cx + half_w
    )

    y2 = int(
        cy + half_h
    )

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        w,
        x2
    )

    y2 = min(
        h,
        y2
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return None, None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None, None

    return crop, (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# REMAP LANDMARKS
# ============================================================

def remap_landmarks(
    landmarks,
    crop_box,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = crop_box

    crop_width = (
        x2 - x1
    )

    crop_height = (
        y2 - y1
    )

    result = []

    for p in landmarks:

        class P:
            pass

        q = P()

        px = (
            x1
            +
            p.x * crop_width
        )

        py = (
            y1
            +
            p.y * crop_height
        )

        q.x = (
            px
            /
            frame_width
        )

        q.y = (
            py
            /
            frame_height
        )

        q.z = p.z

        result.append(q)

    return result


# ============================================================
# REACQUISITION WORKER
# ============================================================

class ReacquisitionWorker:

    def __init__(self):

        self.detector = (
            create_search_detector()
        )

        self.lock = (
            threading.Lock()
        )

        self.latest_frame = None

        self.last_hand = None

        self.result = None
        self.result_time = 0.0

        self.running = True

        self.last_search = 0.0

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )

        self.thread.start()

    # --------------------------------------------------------
    # Give worker latest frame + previous hand.
    # --------------------------------------------------------

    def submit_frame(
        self,
        frame,
        previous_hand
    ):

        with self.lock:

            self.latest_frame = (
                frame.copy()
            )

            if previous_hand is not None:

                self.last_hand = (
                    previous_hand
                )

    # --------------------------------------------------------
    # Get newest valid result.
    # --------------------------------------------------------

    def get_result(self):

        with self.lock:

            if self.result is None:

                return None

            age = (
                time.time()
                -
                self.result_time
            )

            if age > MAX_RESULT_AGE:

                self.result = None

                return None

            result = self.result

            self.result = None

            return result

    # --------------------------------------------------------

    def publish(
        self,
        landmarks,
        source
    ):

        with self.lock:

            self.result = HandResult(
                landmarks=landmarks,
                timestamp=time.time(),
                source=source
            )

            self.result_time = (
                time.time()
            )

    # ========================================================
    # MAIN SEARCH THREAD
    # ========================================================

    def run(self):

        # Full-frame search locations.
        regions = [

            (0.50, 0.50),

            (0.30, 0.35),

            (0.70, 0.35),

            (0.30, 0.65),

            (0.70, 0.65),
        ]

        region_index = 0

        while self.running:

            now = time.time()

            if (
                now
                -
                self.last_search
                <
                REACQUISITION_INTERVAL
            ):

                time.sleep(
                    0.005
                )

                continue

            self.last_search = now

            with self.lock:

                if self.latest_frame is None:

                    continue

                frame = (
                    self.latest_frame.copy()
                )

                previous_hand = (
                    self.last_hand
                )

            h, w = frame.shape[:2]

            found = False

            # =================================================
            # 1. LOCAL SEARCH
            #
            # If we know where the hand was,
            # look there FIRST.
            # =================================================

            if previous_hand is not None:

                px, py = (
                    hand_center(
                        previous_hand
                    )
                )

                cx = int(
                    px * w
                )

                cy = int(
                    py * h
                )

                crop_w = int(
                    w
                    *
                    LOCAL_SEARCH_RADIUS_X
                    *
                    2.0
                )

                crop_h = int(
                    h
                    *
                    LOCAL_SEARCH_RADIUS_Y
                    *
                    2.0
                )

                # Two scales.
                for scale in (
                    1.45,
                    1.85
                ):

                    crop, box = (
                        make_crop(
                            frame,
                            cx,
                            cy,
                            crop_w,
                            crop_h,
                            scale
                        )
                    )

                    if crop is None:
                        continue

                    # Original first.
                    lm = detect_image(
                        self.detector,
                        crop
                    )

                    if lm is None:

                        # Contrast fallback.
                        enhanced = (
                            increase_contrast(
                                crop
                            )
                        )

                        lm = detect_image(
                            self.detector,
                            enhanced
                        )

                    if lm is not None:

                        full_lm = (
                            remap_landmarks(
                                lm,
                                box,
                                w,
                                h
                            )
                        )

                        if (
                            valid_landmarks(
                                full_lm,
                                w,
                                h
                            )
                            and
                            spatially_plausible(
                                full_lm,
                                previous_hand
                            )
                        ):

                            self.publish(
                                full_lm,
                                "LOCAL"
                            )

                            found = True

                            break

                if found:
                    continue

            # =================================================
            # 2. FULL-FRAME SEARCH
            # =================================================

            lm = detect_image(
                self.detector,
                frame
            )

            if lm is not None:

                if (
                    previous_hand is None
                    or
                    spatially_plausible(
                        lm,
                        previous_hand
                    )
                ):

                    self.publish(
                        lm,
                        "FULL"
                    )

                    continue

            # =================================================
            # 3. ROTATING OVERLAPPING DISTANT SEARCH
            # =================================================

            cx_norm, cy_norm = (
                regions[
                    region_index
                    %
                    len(regions)
                ]
            )

            region_index += 1

            cx = int(
                cx_norm * w
            )

            cy = int(
                cy_norm * h
            )

            crop_w = int(
                w * 0.62
            )

            crop_h = int(
                h * 0.70
            )

            for scale in (
                1.45,
                1.90
            ):

                crop, box = (
                    make_crop(
                        frame,
                        cx,
                        cy,
                        crop_w,
                        crop_h,
                        scale
                    )
                )

                if crop is None:
                    continue

                versions = [
                    crop,
                    increase_contrast(
                        crop
                    )
                ]

                for version in versions:

                    lm = detect_image(
                        self.detector,
                        version
                    )

                    if lm is None:
                        continue

                    full_lm = (
                        remap_landmarks(
                            lm,
                            box,
                            w,
                            h
                        )
                    )

                    if not valid_landmarks(
                        full_lm,
                        w,
                        h
                    ):
                        continue

                    if (
                        previous_hand is not None
                        and
                        not spatially_plausible(
                            full_lm,
                            previous_hand
                        )
                    ):
                        continue

                    self.publish(
                        full_lm,
                        "DISTANT"
                    )

                    found = True

                    break

                if found:
                    break

            time.sleep(
                0.001
            )

    def stop(self):

        self.running = False

        if self.thread.is_alive():

            self.thread.join(
                timeout=1.0
            )

        try:

            self.detector.close()

        except Exception:

            pass


# ============================================================
# DRAW HAND
# ============================================================

CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17),
]


def draw_hand(
    frame,
    landmarks
):

    h, w = frame.shape[:2]

    pts = []

    for p in landmarks:

        x = int(
            p.x * w
        )

        y = int(
            p.y * h
        )

        pts.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1
        )

    for a, b in CONNECTIONS:

        cv2.line(
            frame,
            pts[a],
            pts[b],
            (255, 255, 255),
            2
        )


# ============================================================
# LOAD CAMERA FRAME
# ============================================================

def load_frame():

    return cv2.imread(
        FRAME_PATH
    )


# ============================================================
# MOTION GESTURE DETECTION
# ============================================================

def get_palm_direction(lm):
    """
    Estimate whether the palm is tilted upward or downward.

    Uses the wrist, index MCP, and pinky MCP to estimate the
    palm-plane normal. This is intentionally tolerant because
    the camera is not perfectly calibrated to gravity.
    """

    if lm is None or len(lm) != 21:
        return "UNKNOWN"

    w = lm[0]
    i = lm[5]
    p = lm[17]

    ax = i.x - w.x
    ay = i.y - w.y
    az = i.z - w.z

    bx = p.x - w.x
    by = p.y - w.y
    bz = p.z - w.z

    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx

    magnitude = math.sqrt(
        nx * nx +
        ny * ny +
        nz * nz
    )

    if magnitude < 1e-6:
        return "UNKNOWN"

    ny /= magnitude

    if ny < PALM_UP_THRESHOLD:
        return "UP"

    if ny > PALM_DOWN_THRESHOLD:
        return "DOWN"

    return "UNKNOWN"


class MotionGestureDetector:
    def __init__(self):
        self.history = []
        self.last_command_time = 0.0

    def reset(self):
        self.history.clear()

    def update(self, lm):
        if lm is None or len(lm) != 21:
            self.reset()
            return "NONE"

        now = time.time()
        palm_direction = get_palm_direction(lm)

        self.history.append((
            now,
            lm[0].x,
            lm[0].y,
            palm_direction
        ))

        cutoff = now - MOTION_WINDOW
        self.history = [
            item for item in self.history
            if item[0] >= cutoff
        ]

        if len(self.history) < 3:
            return "NONE"

        old = self.history[0]

        dy = lm[0].y - old[2]

        moved_up = dy < -0.07
        moved_down = dy > 0.07

        print(
            f"DEBUG MOTION: palm={palm_direction} dy={dy:.3f}",
            end="\\r",
            flush=True
        )

        if now - self.last_command_time < POSE_COMMAND_COOLDOWN:
            return "NONE"

        if palm_direction == "UP" and moved_up:
            self.last_command_time = now
            self.reset()
            return "SIT"

        if palm_direction == "DOWN" and moved_down:
            self.last_command_time = now
            self.reset()
            return "LAY_DOWN"

        return "NONE"

# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Starting LiDAR20..."
    )

    print(
        "Sticky tracking + local reacquisition + tolerant gestures"
    )

    print(
        "Press ESC to exit."
    )

    detector = (
        create_video_detector()
    )

    reacquirer = (
        ReacquisitionWorker()
    )

    stabilizer = (
        GestureStabilizer(
            GESTURE_CONFIRMATIONS
        )
    )

    # --------------------------------------------------------

    motion_detector = (
        MotionGestureDetector()
    )

    start_time = time.time()

    current_hand = None

    last_hand_time = 0.0

    current_gesture = "NONE"

    previous_gesture = "NONE"

    last_command = None

    last_command_time = 0.0

    try:

        while True:

            frame = load_frame()

            if frame is None:

                time.sleep(
                    0.01
                )

                continue

            now = time.time()

            h, w = frame.shape[:2]

            # ------------------------------------------------
            # Submit current frame and current hand position.
            # ------------------------------------------------

            reacquirer.submit_frame(
                frame,
                current_hand
            )

            # ------------------------------------------------
            # NORMAL TRACKING
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (
                    now
                    -
                    start_time
                )
                * 1000
            )

            try:

                result = (
                    detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

            except Exception:

                result = None

            hand = None

            if (
                result is not None
                and
                result.hand_landmarks
            ):

                candidate = (
                    result.hand_landmarks[0]
                )

                if valid_landmarks(
                    candidate,
                    w,
                    h
                ):

                    # Don't allow obvious teleportation.
                    if (
                        current_hand is None
                        or
                        spatially_plausible(
                            candidate,
                            current_hand
                        )
                    ):

                        hand = candidate

            # ------------------------------------------------
            # Normal tracking success.
            # ------------------------------------------------

            if hand is not None:

                current_hand = hand

                last_hand_time = now

            # ------------------------------------------------
            # Try local/background reacquisition.
            # ------------------------------------------------

            else:

                reacquired = (
                    reacquirer.get_result()
                )

                if reacquired is not None:

                    candidate = (
                        reacquired.landmarks
                    )

                    if valid_landmarks(
                        candidate,
                        w,
                        h
                    ):

                        if (
                            current_hand is None
                            or
                            spatially_plausible(
                                candidate,
                                current_hand
                            )
                        ):

                            current_hand = (
                                candidate
                            )

                            last_hand_time = (
                                now
                            )

            # ------------------------------------------------
            # Hand timeout.
            # ------------------------------------------------

            if (
                current_hand is not None
                and
                (
                    now
                    -
                    last_hand_time
                )
                >
                HAND_TIMEOUT
            ):

                current_hand = None

                stabilizer.reset()

                current_gesture = "NONE"

                previous_gesture = "NONE"

            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            raw_gesture = "NONE"

            if current_hand is not None:

                raw_gesture = (
                    classify_gesture(
                        current_hand
                    )
                )

            current_gesture = (
                stabilizer.update(
                    raw_gesture
                )
            )

            print(
                f"RAW={raw_gesture:6s}  STABLE={current_gesture:6s}",
                flush=True
            )

            # ------------------------------------------------
            # MOTION GESTURES
            # ------------------------------------------------

            motion_command = "NONE"

            if current_hand is not None:

                motion_command = (
                    motion_detector.update(
                        current_hand
                    )
                )

            else:

                motion_detector.reset()

            # ------------------------------------------------
            # MOTION COMMANDS
            # ------------------------------------------------

            if motion_command == "SIT":

                print("MOTION COMMAND: SIT")

                try:

                    import subprocess

                    result = subprocess.run(
                        ["/home/srge/go2_pose_command", "sit"],
                        capture_output=True,
                        text=True
                    )

                    print(result.stdout, end="")

                    if result.stderr:
                        print(result.stderr, end="")

                except Exception as e:

                    print(
                        "SIT command error:",
                        e
                    )

            elif motion_command == "LAY_DOWN":

                print("MOTION COMMAND: LAY DOWN")

                try:

                    import subprocess

                    result = subprocess.run(
                        ["/home/srge/go2_pose_command", "lay_down"],
                        capture_output=True,
                        text=True
                    )

                    print(result.stdout, end="")

                    if result.stderr:
                        print(result.stderr, end="")

                except Exception as e:

                    print(
                        "LAY DOWN command error:",
                        e
                    )

            # ------------------------------------------------
            # COMMAND
            # ------------------------------------------------

            print(
                f"COMMAND GATE: current={current_gesture} previous={previous_gesture}",
                flush=True
            )

            # FOLLOW:
            # Trigger only when FOLLOW is first detected.
            if (
                current_gesture == "FOLLOW"
                and
                previous_gesture != "FOLLOW"
            ):

                print(
                    "COMMAND: FOLLOW",
                    flush=True
                )

                person_xyz = get_d435i_person_xyz()

                if person_xyz is not None:

                    person_x, person_y, person_z = person_xyz

                    print(
                        f"FOLLOW TARGET: "
                        f"X={person_x:+.2f}m "
                        f"Y={person_y:+.2f}m "
                        f"Z={person_z:.2f}m",
                        flush=True
                    )

                else:

                    print(
                        "FOLLOW TARGET: "
                        "D435i person not detected",
                        flush=True
                    )

            # Other gesture commands
            elif (
                current_gesture != "NONE"
                and
                current_gesture != previous_gesture
            ):

                if current_gesture == "FIST":

                    command = "STAND_UP"

                elif current_gesture == "POINT":

                    command = "INVESTIGATE"

                else:

                    # OPEN = relaxed / no command
                    command = None

                if command == "STAND_UP":

                    print(
                        "COMMAND: STAND UP",
                        flush=True
                    )

                    try:

                        import subprocess

                        result = subprocess.run(
                            [
                                "/home/srge/go2_pose_command",
                                "stand_up"
                            ],
                            capture_output=True,
                            text=True,
                            env={
                                **__import__("os").environ,
                                "LD_LIBRARY_PATH":
                                    "/usr/local/lib:"
                                    + __import__("os").environ.get(
                                        "LD_LIBRARY_PATH",
                                        ""
                                    )
                            }
                        )

                        print(
                            result.stdout,
                            end=""
                        )

                        if result.stderr:

                            print(
                                result.stderr,
                                end=""
                            )

                    except Exception as e:

                        print(
                            "STAND UP command error:",
                            e
                        )

                if command is not None:

                    if (
                        command
                        !=
                        last_command
                        or
                        (
                            now
                            -
                            last_command_time
                        )
                        >
                        COMMAND_COOLDOWN
                    ):

                        print(
                            "COMMAND:",
                            command
                        )

                        last_command = command
                        last_command_time = now

            # Always update AFTER command processing.
            previous_gesture = current_gesture

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = frame.copy()

            if current_hand is not None:

                draw_hand(
                    display,
                    current_hand
                )

                x1, y1, x2, y2 = (
                    landmark_bbox(
                        current_hand,
                        w,
                        h
                    )
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if current_gesture == "FIST":

                label = "STAND"

            elif current_gesture == "POINT":

                label = "INVESTIGATE"

            elif current_gesture == "OPEN":

                label = "NO COMMAND"

            elif motion_command == "SIT":

                label = "SIT"

            elif motion_command == "LAY_DOWN":

                label = "LAY DOWN"

            else:

                label = "NO COMMAND"

            cv2.putText(
                display,
                label,
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                "LiDAR20",
                (40, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if current_hand is not None:

                status = "HAND TRACKED"

            else:

                status = "SEARCHING"

            cv2.putText(
                display,
                status,
                (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1)

            if key == 27:
                break

            time.sleep(
                0.001
            )

    finally:

        print(
            "Stopping LiDAR20..."
        )

        reacquirer.stop()

        try:

            detector.close()

        except Exception:

            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
