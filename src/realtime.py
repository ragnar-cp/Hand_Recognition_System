# src/realtime.py
"""
Realtime demo with normalization + smoothing + confidence threshold.
"""

import os, time, argparse, numpy as np, joblib
from tensorflow.keras.models import load_model
from src import preprocess

def safe_load_label_encoder(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print("Label encoder not found:", e)
        return None

def try_import_mediapipe():
    try:
        import mediapipe as mp
        import cv2
        return mp, cv2
    except Exception as e:
        print("mediapipe/opencv not available:", e)
        return None, None

def majority_vote(window):
    if not window:
        return None
    vals = [v for v in window if v is not None]
    if not vals:
        return None
    return max(set(vals), key=vals.count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiments/best_model.h5")
    parser.add_argument("--labelenc", default="experiments/label_encoder.joblib")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--window", type=int, default=8, help="smoothing window (frames)")
    parser.add_argument("--conf-thresh", type=float, default=0.6, help="confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found. Please run: python -m src.train")
        return

    model = load_model(args.model)
    le = safe_load_label_encoder(args.labelenc)
    mp, cv2 = try_import_mediapipe()

    if mp and cv2:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
        cap = cv2.VideoCapture(args.cam)
        window = []
        print("Press q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            display_text = "No hand"
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                vec = []
                for p in lm.landmark:
                    vec += [p.x, p.y, p.z]
                # normalize before predict
                x = preprocess.normalize_landmarks(np.array(vec))
                x = x.reshape(1, -1)
                pred = model.predict(x, verbose=0)[0]
                idx = int(pred.argmax())
                conf = float(pred.max())
                if conf < args.conf_thresh:
                    cur_label = "UNKNOWN"
                else:
                    cur_label = le.classes_[idx] if le else str(idx)
                window.append(cur_label)
                if len(window) > args.window:
                    window.pop(0)
                voted = majority_vote(window)
                display_text = f"{voted} ({conf:.2f})" if voted is not None else "UNKNOWN"
                # draw skeleton
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            # draw label
            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Realtime (smoothed)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print("No MediaPipe; run simulated demo.")
        # fallback: use same smoothing behavior but with fake vectors
        le = le if le else None
        window = []
        import cv2
        cap = cv2.VideoCapture(args.cam)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            x = np.random.normal(0.5, 0.15, size=(1,63))
            pred = model.predict(x)[0]
            idx = int(pred.argmax()); conf = float(pred.max())
            if conf < args.conf_thresh:
                cur_label = "UNKNOWN"
            else:
                cur_label = le.classes_[idx] if le else str(idx)
            window.append(cur_label)
            if len(window) > args.window: window.pop(0)
            voted = majority_vote(window)
            display_text = f"{voted} ({conf:.2f})" if voted is not None else "UNKNOWN"
            cv2.putText(frame, display_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Simulated (smoothed)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        try: cv2.destroyAllWindows()
        except: pass

if __name__ == "__main__":
    main()
