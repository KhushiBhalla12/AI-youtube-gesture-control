import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, Response, render_template, jsonify, request
import pyautogui
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

pyautogui.FAILSAFE = False

app = Flask(__name__)

# --- YOUTUBE API CONFIGURATION ---
YOUTUBE_API_KEY = "AIzaSyCQ65LN57QjMsk3bdcfHIXVwn--5JqTLtU"

def search_youtube_videos(query):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=4,
            type='video'
        )
        response = request.execute()
        videos = []
        for item in response.get('items', []):
            videos.append({
                'title': item['snippet']['title'],
                'videoId': item['id']['videoId'],
                'thumbnail': item['snippet']['thumbnails']['high']['url']
            })
        return videos
    except HttpError as e:
        print(f"[YOUTUBE API HTTP ERROR]: {e}")
        return []
    except Exception as e:
        print(f"[GENERAL SEARCH ERROR]: {e}")
        return []

# --- GESTURE & WEBCAM SETUP ---
current_gesture = "WAITING"
camera_active = False

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,  # Increased confidence to reduce flickering
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        
    (0, 5), (5, 6), (6, 7), (7, 8),        
    (0, 9), (9, 10), (10, 11), (11, 12),  
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20) 
]

def draw_landmarks_manually(image, hand_landmarks_list):
    annotated_image = image.copy()
    h, w, _ = annotated_image.shape
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated_image, pts[a], pts[b], (56, 189, 248), 2)
        for pt in pts:
            cv2.circle(annotated_image, pt, 4, (14, 165, 233), -1)
    return annotated_image

cap = cv2.VideoCapture(0)
last_detected_gesture = "NONE"
gesture_start_time = 0
HOLD_DURATION = 1.2  # Slightly optimized hold duration for stability
last_action_time = 0
cooldown = 1.2

def generate_frames():
    global current_gesture, camera_active, last_detected_gesture, gesture_start_time, last_action_time
    while True:
        if not camera_active:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA OFF", (210, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 180, 200), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        current_time = time.time()
        raw_gesture = "NO HAND"
        
        if detection_result.hand_landmarks:
            frame = draw_landmarks_manually(frame, detection_result.hand_landmarks)
            num_hands = len(detection_result.hand_landmarks)

            # Strict check: only process volume if TWO distinct hands are clearly detected
            if num_hands >= 2:
                h1, h2 = detection_result.hand_landmarks[0], detection_result.hand_landmarks[1]
                h1_open = all([h1[tip].y < h1[tip - 2].y for tip in [8, 12, 16, 20]])
                h2_open = all([h2[tip].y < h2[tip - 2].y for tip in [8, 12, 16, 20]])
                h1_thumb_up = h1[4].y < h1[3].y and h1[4].y < h1[2].y and not h1_open
                h2_thumb_up = h2[4].y < h2[3].y and h2[4].y < h2[2].y and not h2_open

                if h1_open and h2_open:
                    raw_gesture = "VOLUME_UP"
                elif h1_thumb_up and h2_thumb_up:
                    raw_gesture = "VOLUME_DOWN"
                else:
                    raw_gesture = "DUAL TRACKING"
            elif num_hands == 1:
                first_hand = detection_result.hand_landmarks[0]
                is_back_of_hand = first_hand[5].x > first_hand[17].x
                fingers_extended = all([first_hand[tip].y < first_hand[tip - 2].y for tip in [8, 12, 16, 20]])
                fist_closed = all([first_hand[tip].y > first_hand[tip - 2].y for tip in [8, 12, 16, 20]])

                if is_back_of_hand and fingers_extended:
                    raw_gesture = "BACK_HAND_CLICK"
                elif fingers_extended:
                    raw_gesture = "PLAY"
                elif fist_closed:
                    raw_gesture = "PAUSE"
                else:
                    raw_gesture = "HAND ACTIVE"

        # Action execution mapping
        if raw_gesture in ["VOLUME_UP", "VOLUME_DOWN", "BACK_HAND_CLICK"]:
            if current_time - last_action_time > cooldown:
                if raw_gesture == "VOLUME_UP":
                    pyautogui.press('volumeup')
                    current_gesture = "🔊 VOLUME UP (✌️✌️)"
                elif raw_gesture == "VOLUME_DOWN":
                    pyautogui.press('volumedown')
                    current_gesture = "🔉 VOLUME DOWN (👍👍)"
                elif raw_gesture == "BACK_HAND_CLICK":
                    pyautogui.click()
                    current_gesture = "🖱️ CLICK (✋ Back-Hand)"
                last_action_time = current_time
        elif raw_gesture in ["PLAY", "PAUSE"]:
            if raw_gesture == last_detected_gesture:
                if current_time - gesture_start_time >= HOLD_DURATION:
                    if current_time - last_action_time > cooldown:
                        if raw_gesture == "PLAY":
                            pyautogui.press('playpause')
                            current_gesture = "▶️ PLAY (🖐️ Open Palm)"
                        elif raw_gesture == "PAUSE":
                            pyautogui.press('playpause')
                            current_gesture = "⏸️ PAUSE (✊ Closed Fist)"
                        last_action_time = current_time
                        gesture_start_time = current_time
            else:
                last_detected_gesture = raw_gesture
                gesture_start_time = current_time
                current_gesture = f"HOLDING {raw_gesture}..."
        else:
            last_detected_gesture = "NONE"
            current_gesture = raw_gesture

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_cam', methods=['POST'])
def toggle_cam():
    global camera_active
    camera_active = not camera_active
    return jsonify({'camera_active': camera_active})

@app.route('/get_gesture')
def get_gesture():
    return jsonify({'gesture': current_gesture if camera_active else "CAMERA OFF"})

@app.route('/search_api', methods=['POST'])
def search_api():
    data = request.get_json()
    query = data.get('query', 'Lofi Chill')
    results = search_youtube_videos(query)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)