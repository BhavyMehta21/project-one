
import cv2
import mediapipe as mp
import math
import time
import winsound
import subprocess
import tkinter as tk

from pathlib import Path
from collections import deque
from tkinter import ttk
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "DRIVER DROWSINESS DETECTION SYSTEM"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"
CAMERA_INDEX = 0

EAR_THRESHOLD = 0.20
EYE_CLOSED_SECONDS = 1.5

# Mouth opening ratio threshold for yawning
YAWN_THRESHOLD = 0.070
YAWN_MIN_SECONDS = 0.6

# Drowsiness alarm settings
BEEP_DURATION_MS = 70

LEVEL_1_FREQUENCY = 1400
LEVEL_2_FREQUENCY = 1800
LEVEL_3_FREQUENCY = 2200

LEVEL_1_INTERVAL = 0.70
LEVEL_2_INTERVAL = 0.40
LEVEL_3_INTERVAL = 0.20

# Different sound for yawning
YAWN_FREQUENCY = 700
YAWN_SOUND_DURATION_MS = 250

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


# ============================================================
# GLOBAL VARIABLES
# ============================================================

camera = None
face_landmarker = None

camera_running = False
session_start_time = time.time()

eye_closed_since = None
closure_duration = 0.0

alert_count = 0
alarm_level = 1
last_alarm_time = 0.0

mouth_open_since = None
yawn_active = False
yawn_count = 0

last_yawn_sound_time = 0.0
last_frame_time_ms = 0

fps_history = deque(maxlen=20)

current_status = "INITIALIZING"
current_yawn_status = "NOT DETECTED"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_distance(point_a, point_b):
    return math.dist(point_a, point_b)


def calculate_ear(points):
    """
    Calculate Eye Aspect Ratio from six eye landmarks.

    EAR = (vertical distance 1 + vertical distance 2)
          / (2 * horizontal distance)
    """

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])
    horizontal = calculate_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_landmark_points(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    return points


def calculate_mouth_ratio(landmarks, width, height):
    """
    Calculate mouth opening relative to mouth width.
    """

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical_distance = math.dist(
        (top.x * width, top.y * height),
        (bottom.x * width, bottom.y * height)
    )

    horizontal_distance = math.dist(
        (left.x * width, left.y * height),
        (right.x * width, right.y * height)
    )

    if horizontal_distance == 0:
        return 0.0

    return vertical_distance / horizontal_distance


def get_alarm_settings(count):
    """
    Increase alarm intensity as alert beeps accumulate.
    """

    if count <= 5:
        return 1, LEVEL_1_FREQUENCY, LEVEL_1_INTERVAL

    elif count <= 10:
        return 2, LEVEL_2_FREQUENCY, LEVEL_2_INTERVAL

    return 3, LEVEL_3_FREQUENCY, LEVEL_3_INTERVAL


def play_drowsiness_beep(frequency):
    """
    Play a short drowsiness warning beep.
    """

    try:
        winsound.Beep(frequency, BEEP_DURATION_MS)
    except RuntimeError:
        pass


def play_yawn_sound():
    """
    Play a different sound for a detected yawn.
    """

    try:
        winsound.Beep(YAWN_FREQUENCY, YAWN_SOUND_DURATION_MS)
    except RuntimeError:
        pass


def speak_yawn_reset_message():
    """Speak a reset message using Windows built-in text-to-speech."""
    try:
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice.Speak('Yawn system reseted'); "
            "$voice.Dispose()"
        )
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", command],
            startupinfo=startup_info,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        # Keep the camera/dashboard running if speech is unavailable.
        pass


def draw_eye_points(frame, points):
    for point in points:
        cv2.circle(frame, point, 3, (255, 255, 0), -1)


def draw_mouth_points(frame, landmarks, width, height):
    mouth_indices = [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 78, 13, 14
    ]

    for index in mouth_indices:
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(frame, (x, y), 2, (0, 165, 255), -1)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

def initialize_face_landmarker():
    global face_landmarker

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n\n"
            "Keep face_landmarker.task beside dashboard.py."
        )

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(
        options
    )


# ============================================================
# DASHBOARD WINDOW
# ============================================================

root = tk.Tk()
root.title(APP_TITLE)
root.geometry("1250x800")
root.minsize(1000, 700)
root.configure(bg="#101820")

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TLabel",
    background="#17232f",
    foreground="#eaf2f8",
    font=("Segoe UI", 11)
)

style.configure(
    "TFrame",
    background="#17232f"
)

style.configure(
    "TButton",
    font=("Segoe UI", 11, "bold"),
    padding=8
)


# Main heading
header = tk.Frame(root, bg="#101820")
header.pack(fill="x", padx=20, pady=(15, 10))

tk.Label(
    header,
    text="DRIVER DROWSINESS",
    font=("Segoe UI", 23, "bold"),
    bg="#101820",
    fg="#ffffff"
).pack(anchor="w")

tk.Label(
    header,
    text="REAL-TIME DRIVER SAFETY MONITORING",
    font=("Segoe UI", 10),
    bg="#101820",
    fg="#8ca3b8"
).pack(anchor="w", pady=(2, 0))


# Main layout
main_frame = tk.Frame(root, bg="#101820")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

camera_panel = tk.Frame(main_frame, bg="#17232f")
camera_panel.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)

dashboard_panel = tk.Frame(
    main_frame,
    bg="#17232f",
    width=330
)
dashboard_panel.pack(
    side="right",
    fill="y",
    padx=(10, 0)
)
dashboard_panel.pack_propagate(False)


# Camera panel
tk.Label(
    camera_panel,
    text="LIVE CAMERA FEED",
    font=("Segoe UI", 13, "bold"),
    bg="#17232f",
    fg="#ffffff"
).pack(anchor="w", padx=15, pady=12)

video_label = tk.Label(
    camera_panel,
    bg="#080d12",
    text="Initializing camera...",
    fg="#8ca3b8",
    font=("Segoe UI", 12)
)
video_label.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)


# Dashboard title
tk.Label(
    dashboard_panel,
    text="LIVE MONITORING",
    font=("Segoe UI", 14, "bold"),
    bg="#17232f",
    fg="#ffffff"
).pack(anchor="w", padx=15, pady=(15, 10))


def create_metric(parent, title, initial_value="--"):
    card = tk.Frame(parent, bg="#223242")
    card.pack(fill="x", padx=12, pady=5)

    tk.Label(
        card,
        text=title,
        font=("Segoe UI", 9),
        bg="#223242",
        fg="#9db0c2"
    ).pack(anchor="w", padx=10, pady=(8, 0))

    value_label = tk.Label(
        card,
        text=initial_value,
        font=("Segoe UI", 17, "bold"),
        bg="#223242",
        fg="#ffffff"
    )
    value_label.pack(anchor="w", padx=10, pady=(2, 8))

    return value_label


status_label = create_metric(
    dashboard_panel, "DRIVER STATUS", "INITIALIZING"
)

ear_label = create_metric(
    dashboard_panel, "AVERAGE EYE ASPECT RATIO", "--"
)

left_ear_label = create_metric(
    dashboard_panel, "LEFT EYE EAR", "--"
)

right_ear_label = create_metric(
    dashboard_panel, "RIGHT EYE EAR", "--"
)

closure_label = create_metric(
    dashboard_panel, "EYE CLOSURE DURATION", "0.0 s"
)

alert_label = create_metric(
    dashboard_panel, "DROWSINESS ALERT BEEPS", "0"
)

alarm_level_label = create_metric(
    dashboard_panel, "ALARM INTENSITY", "LEVEL 1"
)

yawn_label = create_metric(
    dashboard_panel, "TOTAL YAWNS", "0"
)

yawn_status_label = create_metric(
    dashboard_panel, "YAWN STATUS", "NOT DETECTED"
)

session_label = create_metric(
    dashboard_panel, "SESSION TIME", "00:00:00"
)


# Footer
footer = tk.Label(
    root,
    text="Safety monitoring prototype | Always remain alert while driving",
    font=("Segoe UI", 9),
    bg="#101820",
    fg="#8ca3b8"
)
footer.pack(pady=(0, 10))


# ============================================================
# RESET DROWSINESS ALARM
# ============================================================

def reset_drowsiness_alarm():
    """
    Reset alarm escalation whenever the eyes reopen.
    """

    global eye_closed_since
    global closure_duration
    global alert_count
    global alarm_level
    global last_alarm_time

    eye_closed_since = None
    closure_duration = 0.0

    # Reset the escalating beep count and alarm level.
    alert_count = 0
    alarm_level = 1
    last_alarm_time = 0.0

    alert_label.config(text="0")

    alarm_level_label.config(
        text="LEVEL 1",
        fg="#36d399"
    )

    closure_label.config(text="0.0 s")


# ============================================================
# CAMERA STARTUP
# ============================================================

def start_camera():
    global camera
    global camera_running
    global session_start_time

    if camera_running:
        return

    try:
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            status_label.config(
                text="CAMERA ERROR",
                fg="#ff5c5c"
            )
            video_label.config(
                text="Unable to open webcam.\n"
                     "Close other apps using the camera."
            )
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        camera_running = True
        session_start_time = time.time()

        status_label.config(
            text="CAMERA ACTIVE",
            fg="#36d399"
        )

        update_frame()

    except Exception as error:
        status_label.config(
            text="STARTUP ERROR",
            fg="#ff5c5c"
        )

        video_label.config(text=str(error))


# ============================================================
# MAIN FRAME UPDATE
# ============================================================

def update_frame():
    global camera_running
    global eye_closed_since
    global closure_duration
    global alert_count
    global alarm_level
    global last_alarm_time
    global mouth_open_since
    global yawn_active
    global yawn_count
    global last_yawn_sound_time
    global last_frame_time_ms
    global current_status
    global current_yawn_status

    if not camera_running or camera is None:
        return

    success, frame = camera.read()

    if not success:
        status_label.config(
            text="CAMERA SIGNAL LOST",
            fg="#ff5c5c"
        )
        root.after(100, update_frame)
        return

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # MediaPipe VIDEO mode requires increasing timestamps.
    timestamp_ms = int(time.monotonic() * 1000)

    if timestamp_ms <= last_frame_time_ms:
        timestamp_ms = last_frame_time_ms + 1

    last_frame_time_ms = timestamp_ms

    try:
        result = face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    except Exception as error:
        print("Face detection error:", error)
        root.after(20, update_frame)
        return

    face_detected = bool(result.face_landmarks)

    if face_detected:
        landmarks = result.face_landmarks[0]

        # ----------------------------------------------------
        # EYE DETECTION
        # ----------------------------------------------------

        left_points = get_landmark_points(
            landmarks, LEFT_EYE, width, height
        )

        right_points = get_landmark_points(
            landmarks, RIGHT_EYE, width, height
        )

        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        average_ear = (left_ear + right_ear) / 2.0

        draw_eye_points(frame, left_points)
        draw_eye_points(frame, right_points)

        eyes_closed = average_ear < EAR_THRESHOLD

        if eyes_closed:
            if eye_closed_since is None:
                eye_closed_since = time.time()

            closure_duration = time.time() - eye_closed_since

            if closure_duration >= EYE_CLOSED_SECONDS:
                current_status = "DROWSINESS ALERT"
            else:
                current_status = "EYES CLOSED"

            # Trigger repeating beeps once the closure threshold
            # has been reached.
            if closure_duration >= EYE_CLOSED_SECONDS:
                now = time.time()

                # Count alert beeps and increase intensity.
                if now - last_alarm_time >= get_alarm_settings(
                    alert_count + 1
                )[2]:
                    alert_count += 1

                    alarm_level, frequency, interval = (
                        get_alarm_settings(alert_count)
                    )

                    last_alarm_time = now

                    play_drowsiness_beep(frequency)

        else:
            # Eyes have reopened: stop escalation and reset.
            reset_drowsiness_alarm()
            current_status = "ACTIVE"

        # ----------------------------------------------------
        # YAWNING DETECTION
        # ----------------------------------------------------

        mouth_ratio = calculate_mouth_ratio(
            landmarks, width, height
        )

        draw_mouth_points(frame, landmarks, width, height)

        mouth_open = mouth_ratio > YAWN_THRESHOLD

        if mouth_open:
            if mouth_open_since is None:
                mouth_open_since = time.time()

            mouth_open_duration = time.time() - mouth_open_since

            if mouth_open_duration >= YAWN_MIN_SECONDS:
                if not yawn_active:
                    yawn_active = True
                    yawn_count += 1
                    current_yawn_status = "YAWNING"

                    # Reset the yawn counter after every 10 detected yawns.
                    if yawn_count >= 10:
                        yawn_count = 0
                        speak_yawn_reset_message()

                    # A different sound from the drowsiness alarm.
                    now = time.time()

                    if now - last_yawn_sound_time > 1.0:
                        last_yawn_sound_time = now
                        play_yawn_sound()

        else:
            mouth_open_since = None
            yawn_active = False
            current_yawn_status = "NOT DETECTED"

        # ----------------------------------------------------
        # DRAW CAMERA STATUS
        # ----------------------------------------------------

        if current_status == "DROWSINESS ALERT":
            status_color = (0, 0, 255)
        elif current_status == "EYES CLOSED":
            status_color = (0, 165, 255)
        else:
            status_color = (0, 230, 130)

        cv2.putText(
            frame,
            current_status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Yawns: {yawn_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        # Update dashboard metrics.
        ear_label.config(text=f"{average_ear:.3f}")
        left_ear_label.config(text=f"{left_ear:.3f}")
        right_ear_label.config(text=f"{right_ear:.3f}")

        closure_label.config(
            text=f"{closure_duration:.1f} s"
        )

        alert_label.config(text=str(alert_count))

        alarm_level_label.config(
            text=f"LEVEL {alarm_level}",
            fg=(
                "#36d399"
                if alarm_level == 1
                else "#ffb84d"
                if alarm_level == 2
                else "#ff5c5c"
            )
        )

        yawn_label.config(text=str(yawn_count))
        yawn_status_label.config(
            text=current_yawn_status,
            fg=(
                "#ffb84d"
                if current_yawn_status == "YAWNING"
                else "#36d399"
            )
        )

        status_label.config(
            text=current_status,
            fg=(
                "#ff5c5c"
                if current_status == "DROWSINESS ALERT"
                else "#ffb84d"
                if current_status == "EYES CLOSED"
                else "#36d399"
            )
        )

    else:
        # No face detected. Do not treat this as eyes reopening.
        # Preserve the last eye status and show a tracking warning.
        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        status_label.config(
            text="FACE NOT DETECTED",
            fg="#ffb84d"
        )

    # --------------------------------------------------------
    # SESSION TIME
    # --------------------------------------------------------

    elapsed = time.time() - session_start_time

    session_label.config(
        text=format_duration(elapsed)
    )

    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    display_width = max(video_label.winfo_width(), 640)
    display_height = max(video_label.winfo_height(), 360)

    image.thumbnail(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(image=image)

    video_label.config(image=photo, text="")
    video_label.image = photo

    # Schedule next update.
    root.after(10, update_frame)


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def close_application():
    global camera_running
    global camera
    global face_landmarker

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    if face_landmarker is not None:
        face_landmarker.close()
        face_landmarker = None

    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_application)


# ============================================================
# AUTOMATIC STARTUP
# ============================================================

def initialize_and_start():
    try:
        initialize_face_landmarker()
        start_camera()

    except Exception as error:
        status_label.config(
            text="INITIALIZATION ERROR",
            fg="#ff5c5c"
        )

        video_label.config(
            text=f"Could not initialize system:\n{error}"
        )

        print("Initialization error:", error)


print("Driver Drowsiness Detection System")
print("Initializing MediaPipe and webcam...")

root.after(500, initialize_and_start)
root.mainloop()
