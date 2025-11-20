import os
import cv2
import mediapipe as mp
import numpy as np
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_LANDMARKS = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_WRIST,
    mp_holistic.PoseLandmark.RIGHT_WRIST,
]

FACE_LANDMARKS = [
    1,
    61,
    291,
    52,
    282,
]

NUM_HAND_LANDMARKS = 21

POSE_VEC_SIZE = len(POSE_LANDMARKS) * 3
FACE_VEC_SIZE = len(FACE_LANDMARKS) * 3
HAND_VEC_SIZE = NUM_HAND_LANDMARKS * 3
FEATURE_VECTOR_SIZE = POSE_VEC_SIZE + FACE_VEC_SIZE + HAND_VEC_SIZE + HAND_VEC_SIZE

def get_landmark_coords(landmarks, landmark_list):
    if landmarks is None:
        return np.array([np.nan, np.nan, np.nan])
    
    point = landmarks.landmark[landmark_list]
    return np.array([point.x, point.y, point.z])

def extract_keypoints(results):
    keypoints = np.full(FEATURE_VECTOR_SIZE, np.nan, dtype=np.float32)
    
    pose_lm = results.pose_landmarks
    face_lm = results.face_landmarks
    left_hand_lm = results.left_hand_landmarks
    right_hand_lm = results.right_hand_landmarks
    
    if pose_lm:
        left_shoulder = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.RIGHT_SHOULDER)
        
        if np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any():
            return keypoints
        
        anchor = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        
        if scale < 1e-6:
            return keypoints
        
        pose_vecs = []
        for landmark_id in POSE_LANDMARKS:
            point = get_landmark_coords(pose_lm, landmark_id)
            norm_vec = (point - anchor) / scale
            pose_vecs.extend(norm_vec)
        keypoints[0:POSE_VEC_SIZE] = pose_vecs
        
        left_wrist_anchor = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.LEFT_WRIST)
        right_wrist_anchor = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.RIGHT_WRIST)
        
        if face_lm:
            face_vecs = []
            for landmark_id in FACE_LANDMARKS:
                point = get_landmark_coords(face_lm, landmark_id)
                norm_vec = (point - anchor) / scale
                face_vecs.extend(norm_vec)
            keypoints[POSE_VEC_SIZE : POSE_VEC_SIZE + FACE_VEC_SIZE] = face_vecs
            
        if left_hand_lm and not np.isnan(left_wrist_anchor).any():
            left_hand_vecs = []
            for landmark in left_hand_lm.landmark:
                point = np.array([landmark.x, landmark.y, landmark.z])
                norm_vec = (point - left_wrist_anchor) / scale
                left_hand_vecs.extend(norm_vec)
            start_idx = POSE_VEC_SIZE + FACE_VEC_SIZE
            keypoints[start_idx : start_idx + HAND_VEC_SIZE] = left_hand_vecs
            
        if right_hand_lm and not np.isnan(right_wrist_anchor).any():
            right_hand_vecs = []
            for landmark in right_hand_lm.landmark:
                point = np.array([landmark.x, landmark.y, landmark.z])
                norm_vec = (point - right_wrist_anchor) / scale
                right_hand_vecs.extend(norm_vec)
            start_idx = POSE_VEC_SIZE + FACE_VEC_SIZE + HAND_VEC_SIZE
            keypoints[start_idx : start_idx + HAND_VEC_SIZE] = right_hand_vecs
        
    return keypoints

def _find_max_sample_num(folder_path):
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return None
    sample_numbers = []
    for f in files:
        if f.endswith(".npy"):
            try:
                num = int(f.split(".")[0])
                sample_numbers.append(num)
            except ValueError:
                continue

    if not sample_numbers:
        return 0
    return max(sample_numbers)

def get_next_sample_num(folder_path):
    max_num = _find_max_sample_num(folder_path)
    if max_num is None:
        return 0
    return max_num + 1

def get_last_sample_num(folder_path):
    return _find_max_sample_num(folder_path)

DATA_PATH = os.path.join("data")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

sign_name = input("Enter sign name: ")
sign_path = os.path.join(DATA_PATH, sign_name)
if not os.path.exists(sign_path):
    os.makedirs(sign_path)
    print(f"Created folder for: {sign_name}")
    
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0:
    FPS = 30
    
COUNTDOWN_SECONDS = 0
MIN_RECORD_SECONDS = 1.25
MAX_RECORD_SECONDS = 1.5

COUNTDOWN_FRAMES = int(COUNTDOWN_SECONDS * FPS)
MIN_RECORD_FRAMES = int(MIN_RECORD_SECONDS * FPS)
MAX_RECORD_FRAMES = int(MAX_RECORD_SECONDS * FPS)

STATE = "IDLE"
sequence = []
countdown_timer = 0
record_timer = 0

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.resize(frame, (1280,720))
        frame = cv2.flip(frame,1)
        
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1),
        )
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1),
        )
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1),
        )
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        
        if key == ord(' ') and STATE == "IDLE":
            STATE = "COUNTDOWN"
            countdown_timer = COUNTDOWN_FRAMES
            sequence = []
            print("Starting countdown...")
            
        if key == ord('c') and STATE == "IDLE":
            print("Attempting to delete last sample...")
            last_sample_num = get_last_sample_num(sign_path)
            
            if last_sample_num is None:
                print("No samples found in directory to delete...")
            else:
                file_to_delete = os.path.join(sign_path, f"{last_sample_num}.npy")
                try:
                    os.remove(file_to_delete)
                    print(f"Successfully deleted: {file_to_delete}")
                except OSError as e:
                    print(f"Error deleting file {file_to_delete}: {e}")
            
        if STATE == "IDLE":
            cv2.putText(
                image, "PRESS [SPACE] TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
        elif STATE == "COUNTDOWN":
            countdown_val = int(np.ceil(countdown_timer / FPS))
            cv2.putText(image, f"{countdown_val}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            countdown_timer -=1
            if countdown_timer <= 0:
                STATE = "RECORDING"
                record_timer = np.random.randint(MIN_RECORD_FRAMES, MAX_RECORD_FRAMES + 1)
                print(f"RECORDING for {record_timer} frames...")
        elif STATE == "RECORDING":
            cv2.putText(image, "RECORDING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            record_timer -= 1
            if record_timer <= 0:
                STATE = "IDLE"
                print("...Recording finished.")
                
                if len(sequence) > 0:
                    sample_num = get_next_sample_num(sign_path)
                    save_path = os.path.join(sign_path, f"{sample_num}.npy")
                    np.save(save_path, np.array(sequence))
                    print(f"Saved sequence as {save_path}")
                    sequence = []
                    
        cv2.putText(
            image, f"Sign: {sign_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imshow("Sign Language Data Collection", image)
        
cap.release()
cv2.destroyAllWindows()