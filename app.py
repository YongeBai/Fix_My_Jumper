from flask import Flask, render_template, Response
import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import defaultdict

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./pose_landmarker_lite.task"

LANDMARKS = [("Right Shoulder", 12), ("Right Elbow", 14), ("Right Wrist", 16), ("Right Index Finger", 20)]

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for name, idx in LANDMARKS:
                if landmarks[idx].visibility < 0.5:
                    return
                if landmarks[idx].presence < 0.5:
                    return
            if landmarks[12].x < landmarks[14].x and landmarks[14].x < landmarks[16].x and landmarks[16].x < landmarks[20].x:
                print("Shot")

    except:        
        pass
    #     mp.solutions.drawing_utils.draw_landmarks(
    #        image=output_image,
    #        landmark_list=result.pose_landmarks,
    #        connections=[
    #            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    #            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    #            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    #            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    #        ],
    #        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    #        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    #    )
            
        

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

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

            results = landmarker.detect_async(mp_image, frame_timestamp_ms)    
            
            frame.flags.writeable = True                            
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)    
    