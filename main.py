import cv2
import mediapipe as mp
import math

print("Driver Drowsiness Detection System")
print("Starting camera...")

# ==============================
# MediaPipe Tasks
# ==============================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# ==============================
# EYE Landmark Indexes
# ==============================

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]


# ==============================
# EYE Calculation Function
# ==============================

def calculate_eye(points):

    # Vertical distances
    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])

    # Horizontal distance
    horizontal = math.dist(points[0], points[3])

    # Eye Aspect Ratio
    eye = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return eye


# ==============================
# Open Webcam
# ==============================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ==============================
# Start Face Landmarker
# ==============================

with FaceLandmarker.create_from_options(options) as landmarker:

    frame_timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Get frame size
        h, w, _ = frame.shape

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        frame_timestamp += 1

        # Detect face landmarks
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # ==============================
        # If Face Detected
        # ==============================

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            # --------------------------------
            # LEFT EYE POINTS
            # --------------------------------

            left_points = []

            for index in LEFT_EYE:

                landmark = face_landmarks[index]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            # --------------------------------
            # RIGHT EYE POINTS
            # --------------------------------

            right_points = []

            for index in RIGHT_EYE:

                landmark = face_landmarks[index]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            # ==============================
            # Calculate EYE
            # ==============================

            left_eye = calculate_eye(left_points)
            right_eye = calculate_eye(right_points)


            # Average EYE
            average_eye = (
                left_eye + right_eye
            ) / 2.0

            # ==============================
            # Display EYE
            # ==============================

            cv2.putText(
                frame,
                f"Left EYE: {left_eye:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right EYE: {right_eye:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Average EYE: {average_eye:.2f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ==============================
        # Show Camera
        # ==============================

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==============================
# Cleanup
# ==============================

camera.release()

cv2.destroyAllWindows()

print("System stopped.")