from flask import Flask, render_template, Response, jsonify
import cv2
import pygame

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# Initialize pygame for sound playback
pygame.mixer.init()

# Specify path to the music file (adjust this path as per your project structure)
music_path = r"static\music\my_baby_love.mp3"
face_detected = False  # Flag to signal face detection

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face():
    global face_detected
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0 and not face_detected:
                face_detected = True
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_detected')
def face_detected():
    global face_detected
    if face_detected:
        return jsonify(face_detected=True)
    else:
        return jsonify(face_detected=False)

@app.route('/play_song')
def play_song():
    play_song_and_show_message()
    return '', 204

def play_song_and_show_message():
    # Play the song
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()
    
    # This print statement is for debugging purposes
    print("I deeply love you and you're my life")

if __name__ == '__main__':
    app.run(debug=False)
