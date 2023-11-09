import cv2 as cv
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template
from mediapipe.framework.formats import landmark_pb2

model_path = "./pose_landmarker_lite.task"

LANDMARKS = [
    ("Right Shoulder", 12),
    ("Right Elbow", 14),
    ("Right Wrist", 16),
    ("Right Index Finger", 20),
]


def print_result(
    result: mp.tasks.vision.PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
):
    try:
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for _, idx in LANDMARKS:
                if landmarks[idx].visibility < 0.5:
                    return
                if landmarks[idx].presence < 0.5:
                    return
            if (
                landmarks[12].x < landmarks[14].x
                and landmarks[14].x < landmarks[16].x
                and landmarks[16].x < landmarks[20].x
            ):
                print("Shot")

    except:
        pass


def draw_landmarks(
    rgb_image: np.ndarray, detection_result: mp.tasks.vision.PoseLandmarkerResult
):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmarks = pose_landmarks_list[0]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ]
    )
    # for _, idx in LANDMARKS:
    #     pose_landmarks_proto.landmark.append(
    #         landmark_pb2.NormalizedLandmark(
    #             x=pose_landmarks[idx].x,
    #             y=pose_landmarks[idx].y,
    #             z=pose_landmarks[idx].z,
    #         )
    #     )
    connections = set([(12, 14), (14, 16), (16, 20)])
    print(pose_landmarks_proto)
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        connections,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )

    return annotated_image


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

app = Flask(__name__)

camera = cv.VideoCapture(0)


def generate_frames():
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

            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            if results.pose_landmarks:
                frame = draw_landmarks(frame, results)

            ret, buffer = cv.imencode(".jpg", frame)
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
