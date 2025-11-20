import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    MultiHeadAttention,
    LayerNormalization,
)
import threading
import queue
import time
import pyttsx3
import collections
from statistics import Counter


class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(embedding_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        x_zeroed = tf.where(tf.math.is_nan(x), 0.0, x)

        seq_len = tf.shape(x)[1]
        return x_zeroed + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


custom_objects = {
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

try:
    with custom_object_scope(custom_objects):
        model = tf.keras.models.load_model("sign_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model 'sign_model.h5': {e}")
    exit()

try:
    import config

    print("Configuration loaded.")
except ImportError:
    print("CRITICAL ERROR: config.py not found.")
    exit()

try:
    labels = np.load("labels.npy")
    print(f"Labels loaded successfully: {labels}")
except Exception as e:
    print(f"Error loading 'labels.npy': {e}")
    exit()

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
FACE_LANDMARKS = [1, 61, 291, 52, 282]
NUM_HAND_LANDMARKS = 21

POSE_VEC_SIZE = len(POSE_LANDMARKS) * 3
FACE_VEC_SIZE = len(FACE_LANDMARKS) * 3
HAND_VEC_SIZE = NUM_HAND_LANDMARKS * 3

FEATURE_VECTOR_SIZE_CHECK = POSE_VEC_SIZE + FACE_VEC_SIZE + HAND_VEC_SIZE + HAND_VEC_SIZE

if FEATURE_VECTOR_SIZE_CHECK != config.FEATURE_VECTOR_SIZE:
    print("FATAL ERROR: Feature size mismatch!")
    print(f"config.py says {config.FEATURE_VECTOR_SIZE}")
    print(f"app.py calculates {FEATURE_VECTOR_SIZE_CHECK}")
    print("Please check your landmark lists in both files.")
    exit()


def get_landmark_coords(landmarks, landmark_id):
    if landmarks is None:
        return np.array([np.nan, np.nan, np.nan])
    point = landmarks.landmark[landmark_id]
    return np.array([point.x, point.y, point.z])


def extract_keypoints(results):
    keypoints = np.full(config.FEATURE_VECTOR_SIZE, np.nan, dtype=np.float32)
    pose_lm = results.pose_landmarks
    face_lm = results.face_landmarks
    left_hand_lm = results.left_hand_landmarks
    right_hand_lm = results.right_hand_landmarks

    if pose_lm:
        left_shoulder = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.RIGHT_SHOULDER)

        if not np.isnan(left_shoulder).any() and not np.isnan(right_shoulder).any():
            anchor = (left_shoulder + right_shoulder) / 2.0
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale < 1e-6:
                scale = 1e-6  # Avoid division by zero

            # 1. Pose
            pose_vecs = []
            for landmark_id in POSE_LANDMARKS:
                norm_vec = (get_landmark_coords(pose_lm, landmark_id) - anchor) / scale
                pose_vecs.extend(norm_vec)
            keypoints[0:POSE_VEC_SIZE] = pose_vecs

            left_wrist_anchor = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.LEFT_WRIST)
            right_wrist_anchor = get_landmark_coords(pose_lm, mp_holistic.PoseLandmark.RIGHT_WRIST)

            # 2. Face
            if face_lm:
                face_vecs = []
                for landmark_id in FACE_LANDMARKS:
                    norm_vec = (get_landmark_coords(face_lm, landmark_id) - anchor) / scale
                    face_vecs.extend(norm_vec)
                keypoints[POSE_VEC_SIZE : POSE_VEC_SIZE + FACE_VEC_SIZE] = face_vecs

            # 3. Left Hand
            if left_hand_lm and not np.isnan(left_wrist_anchor).any():
                left_hand_vecs = []
                for landmark in left_hand_lm.landmark:
                    norm_vec = (np.array([landmark.x, landmark.y, landmark.z]) - left_wrist_anchor) / scale
                    left_hand_vecs.extend(norm_vec)
                start_idx = POSE_VEC_SIZE + FACE_VEC_SIZE
                keypoints[start_idx : start_idx + HAND_VEC_SIZE] = left_hand_vecs

            # 4. Right Hand
            if right_hand_lm and not np.isnan(right_wrist_anchor).any():
                right_hand_vecs = []
                for landmark in right_hand_lm.landmark:
                    norm_vec = (np.array([landmark.x, landmark.y, landmark.z]) - right_wrist_anchor) / scale
                    right_hand_vecs.extend(norm_vec)
                start_idx = POSE_VEC_SIZE + FACE_VEC_SIZE + HAND_VEC_SIZE
                keypoints[start_idx : start_idx + HAND_VEC_SIZE] = right_hand_vecs
    return keypoints


def pad_and_prep_sequence(seq_list):
    padded_seq = np.full((config.TARGET_FRAMES, config.FEATURE_VECTOR_SIZE), np.nan, dtype=np.float32)
    seq_len = len(seq_list)

    if seq_len > 0:
        copy_len = min(seq_len, config.TARGET_FRAMES)
        padded_seq[-copy_len:] = seq_list[-copy_len:]

    return np.expand_dims(padded_seq, axis=0)


def tts_worker(tts_queue, stop_event):
    print("TTS thread started.")
    engine = None

    def init_engine():
        """Tries to initialize the TTS engine."""
        try:
            engine = pyttsx3.init()
            print("TTS engine initialized.")
            return engine
        except Exception as e:
            print(f"Failed to initialize pyttsx3: {e}")
            return None

    engine = init_engine()

    while not stop_event.is_set():
        if engine is None:
            time.sleep(0.5)
            engine = init_engine()
            if engine is None:
                continue

        try:
            text = tts_queue.get(timeout=1)
            if text is None:
                break

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            tts_queue.stop()

        except queue.Empty:
            continue

        except Exception as e:
            print(f"Error in TTS engine: {e}. Attempting to restart.")
            try:
                del engine
            except:
                pass
            engine = None
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except queue.Empty:
                    break

    print("TTS thread stopping.")
    if engine:
        engine.stop()


def prediction_worker(sequence_buffer, sequence_lock, ui_queue, tts_queue, clear_sentence_event, stop_event):
    print("Prediction thread started.")

    sentence = []

    prediction_history = collections.deque(maxlen=config.PREDICTION_HISTORY_FRAMES)

    last_added_word = None

    while not stop_event.is_set():
        if clear_sentence_event.is_set():
            sentence = []
            prediction_history.clear()
            last_added_word = None
            clear_sentence_event.clear()

        with sequence_lock:
            local_seq = list(sequence_buffer)

        if not local_seq:
            time.sleep(config.PREDICTION_INTERVAL)
            continue

        X_pred = pad_and_prep_sequence(local_seq)
        predictions = model.predict(X_pred, verbose=0)[0]
        best_pred_index = np.argmax(predictions)
        best_pred_sign = labels[best_pred_index]
        best_pred_confidence = predictions[best_pred_index]

        if best_pred_confidence > config.CONFIDENCE_THRESHOLD:
            prediction_history.append(best_pred_sign)
        else:
            prediction_history.append(config.IDLE_SIGN_NAME)

        display_prediction = ""
        if prediction_history:
            most_common_sign = Counter(prediction_history).most_common(1)[0][0]
            if most_common_sign != config.IDLE_SIGN_NAME:
                display_prediction = most_common_sign.upper()

        if len(prediction_history) == config.PREDICTION_HISTORY_FRAMES:
            data = Counter(prediction_history)
            most_common_sign = Counter(prediction_history).most_common(1)[0][0]
            count = data.most_common(1)[0][1]
            stability = count / config.PREDICTION_HISTORY_FRAMES

            if stability >= config.MIN_PREDICTION_STABILITY:
                if most_common_sign == config.IDLE_SIGN_NAME:
                    if last_added_word is not None:
                        print("User is idle. Ready for next word.")
                        last_added_word = None
                elif most_common_sign != last_added_word:
                    sentence.append(most_common_sign)
                    last_added_word = most_common_sign
                    print(f"Confirmed word: {most_common_sign}")
                    tts_queue.put(most_common_sign)

        ui_data = {"sentence": " ".join(sentence), "prediction": display_prediction}
        try:
            ui_queue.put_nowait(ui_data)
        except queue.Full:
            while not ui_queue.empty():
                try:
                    ui_queue.get_nowait()
                except queue.Empty:
                    break
            ui_queue.put(ui_data)

        time.sleep(config.PREDICTION_INTERVAL)

    print("Prediction thread stopping.")


def main():
    sequence_buffer = collections.deque(maxlen=config.TARGET_FRAMES)
    sequence_lock = threading.Lock()
    ui_queue = queue.Queue(maxsize=1)
    tts_queue = queue.Queue()
    stop_event = threading.Event()
    clear_sentence_event = threading.Event()

    ui_data = {"sentence": "", "prediction": ""}

    predictor_thread = threading.Thread(
        target=prediction_worker,
        args=(sequence_buffer, sequence_lock, ui_queue, tts_queue, clear_sentence_event, stop_event),
    )
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, stop_event))

    predictor_thread.start()
    tts_thread.start()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    print("Main thread started. Press 'q' to quit, 's' to speak **sentence**, 'c' to clear.")

    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frame.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            keypoints = extract_keypoints(results)
            with sequence_lock:
                sequence_buffer.append(keypoints)

            try:
                ui_data = ui_queue.get_nowait()
                ui_queue.task_done()
            except queue.Empty:
                pass

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

            cv2.putText(
                image, ui_data["prediction"], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA
            )

            (w, h), _ = cv2.getTextSize(ui_data["sentence"], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(
                image, (40, config.CAMERA_HEIGHT - 80), (50 + w + 10, config.CAMERA_HEIGHT - 30), (0, 0, 0), -1
            )
            cv2.putText(
                image,
                ui_data["sentence"],
                (50, config.CAMERA_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Sign Language Translator", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                print("Clearing sentence and TTS queue...")
                clear_sentence_event.set()

                while not tts_queue.empty():
                    try:
                        tts_queue.get_nowait()
                        tts_queue.task_done()
                    except queue.Empty:
                        break
            if key == ord("s"):
                if ui_data["sentence"]:
                    print(f"Speaking full sentence: {ui_data['sentence']}")
                    tts_queue.put(ui_data["sentence"])

    print("Shutting down...")
    stop_event.set()
    tts_queue.put(None)
    predictor_thread.join()
    tts_thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
