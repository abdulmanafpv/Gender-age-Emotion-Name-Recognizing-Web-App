from flask import Flask, render_template, Response
import cv2
import f_Face_info
import cv2
import time
import imutils
import argparse
import face_recognition
import numpy as np
app=Flask(__name__)


# ----------------------------- webcam -----------------------------

cam = cv2.VideoCapture(0)


    # ----------------------------- webcam -----------------------------

def gen_frames():
    while True:
        star_time = time.time()
        ret, frame = cam.read()
        if not ret:
            break
        else:
            frame = imutils.resize(frame, width=720)

            # obtenego info del frame
            out = f_Face_info.get_face_info(frame)
            # pintar imagen
            frame = f_Face_info.bounding_box(out, frame)

            end_time = time.time() - star_time
            FPS = 1 / end_time
            cv2.putText(frame, f"FPS: {round(FPS, 3)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)