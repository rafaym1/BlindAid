from flask import Flask, Response

# Required to run the YOLOv8 model
import cv2

from running_yolo_webcam.YOLO_Video import video_detection
app = Flask(__name__)

@app.route("/")
def lil():
    return "hello this is testing "
def generate_frames(video=0):

    yolo_output = video_detection(video, model1="runs/detect/train2/weights/best.pt", model2="github/model@1535470106.json", weights2="github/model@1535470106.h5", csvfile_path="rafay.csv")
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        # Any Flask application requires the encoded image to be converted into bytes
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
