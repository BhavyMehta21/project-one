# Driver Drowsiness Detection System 🚗

A real-time driver monitoring system built with Python, OpenCV, and MediaPipe to detect signs of drowsiness and yawning.

## Features

- Real-time face and eye detection using a webcam
- Eye Aspect Ratio (EAR) monitoring
- Repeating audio alerts for drowsiness
- Yawning detection and a yawn counter
- Separate warning sounds for drowsiness and yawning
- Automatic reset of the yawn counter after 10 detections
- Voice announcement when the yawn counter resets
- Live monitoring dashboard

## Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- Tkinter
- Pillow

## Installation and Usage

1. Install Python.
2. Install the required packages:

   ```bash
   pip install opencv-python mediapipe pillow
   ```

3. Ensure `face_landmarker.task` is in the project folder.
4. Run the application:

   ```bash
   python dashboard.py
   ```

Make sure your webcam is connected and available.

## Project Objective

To demonstrate how computer vision can help identify driver fatigue and provide timely audio warnings.

## Disclaimer

This is an educational project and is not a certified vehicle safety system.

## Author

Bhavy Mehta
