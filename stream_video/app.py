from flask import Flask, render_template, Response
import cv2
import datetime

app = Flask(__name__)

# Paths to the MP4 video files
VIDEO_PATH1 = 'videos/sun_glare_condition.avi'
VIDEO_PATH2 = 0
VIDEO_PATH3 = 'videos/rain_condition.avi'
def get_current_datetime():
    # Get the current date and time 
    current_datetime = datetime.datetime.now()
    return current_datetime

def generate_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()

        if not success:
            # If the video ends, reset the video stream to the beginning, 
            # internal pointer points to the first frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        current_datetime = get_current_datetime()
        # Convert current_datetime into a string with specific format
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, formatted_datetime, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
        # Convert the frame to a byte string
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        cv2.waitKey(25)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed1')
def video_feed1():
    # Return the response with the video1 streaming content
    return Response(generate_frames(VIDEO_PATH1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    # Return the response with the video2 streaming content
    return Response(generate_frames(VIDEO_PATH2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    # Return the response with the video3 streaming content
    return Response(generate_frames(VIDEO_PATH3), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5000, debug=True)
    