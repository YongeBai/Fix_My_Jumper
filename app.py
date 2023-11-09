import cv2 as cv
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template
from mediapipe.framework.formats import landmark_pb2
import time

model_path = "./pose_landmarker_lite.task"

LANDMARKS = [
    ("Right Shoulder", 12),
    ("Right Elbow", 14),
    ("Right Wrist", 16),
    ("Right Index Finger", 20),
]

prev_shot_time = 0
total_shots = 0


def draw_landmarks(
    rgb_image: np.ndarray, detection_result: mp.tasks.vision.PoseLandmarkerResult
):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmarks = pose_landmarks_list[0]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    for _, idx in LANDMARKS:
        pose_landmarks_proto.landmark.append(
            landmark_pb2.NormalizedLandmark(
                x=pose_landmarks[idx].x,
                y=pose_landmarks[idx].y,
                z=pose_landmarks[idx].z,
            )
        )
    connections = set([(0, 1), (1, 2), (2, 3)])
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        connections,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )

    return annotated_image


def count_shots(detection_result: mp.tasks.vision.PoseLandmarkerResult) -> int:
    global prev_shot_time

    current_time = time.time()
    shots = 0
    if current_time > prev_shot_time + 5:
        try:
            pose_landmarks_list = detection_result.pose_landmarks
            pose_landmarks = pose_landmarks_list[0]
            if (
                pose_landmarks[20].visibility < 0.7
            ):
                return shots
            if (
                pose_landmarks[12].y < pose_landmarks[14].y
                and pose_landmarks[14].y < pose_landmarks[16].y
                and pose_landmarks[16].y < pose_landmarks[20].y
            ):
                # TODO:ADD ANGLE CALCULATION HERE AND FIGURE OUT WHEN U WANT TO MEASURE THE ANGLES
                print("Shot")
                shots += 1
                prev_shot_time = time.time()
        except IndexError:
            pass

    return shots


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence = 0.7,
    min_pose_presence_confidence = 0.7,
    min_tracking_confidence = 0.7,
)

app = Flask(__name__)

camera = cv.VideoCapture(0)


def generate_frames():
    global total_shots

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame.flags.writeable = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int(camera.get(cv.CAP_PROP_POS_MSEC))

            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # track shots
            shot = count_shots(results)
            if shot:
                total_shots += shot
                print(total_shots)

            # draw landmarks
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            if results.pose_landmarks:
                frame = draw_landmarks(frame, results)

            _, buffer = cv.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
