# src/realtime.py
"""
Realtime demo — observe-then-confirm behavior.

Flow:
 - Keep a majority-vote window for smoothing.
 - When a voted label (not UNKNOWN) is stable for `accept_stability` consecutive voted frames,
   begin observing it.
 - If the same voted label remains during the observation window (`--confirm` seconds),
   commit (append) the label to the transcript.
 - If the voted label changes before confirmation, cancel and begin observing the new one.
 - UI remains responsive (no blocking sleeps).
"""

import os
import time
import argparse
from collections import deque
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# try to import preprocess.normalize_landmarks if present
try:
    from src import preprocess
    _HAS_PREPROCESS = True
except Exception:
    _HAS_PREPROCESS = False

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

def draw_transcript(frame, transcript, org=(10,80), font_scale=1.0, thickness=2, max_width=None):
    import cv2
    x, y = org
    lines = [transcript]
    if max_width is not None:
        lines = []
        cur = ""
        for ch in transcript:
            test = cur + ch
            (w, h), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if w > max_width and cur:
                lines.append(cur)
                cur = ch
            else:
                cur = test
        if cur:
            lines.append(cur)
    for i, line in enumerate(lines):
        # dark outline for readability
        cv2.putText(frame, line, (x, y + i * int(30 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,0,0), thickness+2, lineType=cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * int(30 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,200,0), thickness, lineType=cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiments/best_model.h5")
    parser.add_argument("--labelenc", default="experiments/label_encoder.joblib")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--window", type=int, default=8, help="smoothing window (frames)")
    parser.add_argument("--accept_stability", type=int, default=3, help="consecutive voted frames required to start observing")
    parser.add_argument("--confirm", type=float, default=2.0, help="seconds to observe before confirming & printing label")
    parser.add_argument("--conf-thresh", type=float, default=0.6, help="confidence threshold to consider a prediction valid")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found. Please run: python -m src.train")
        return

    model = load_model(args.model)
    le = safe_load_label_encoder(args.labelenc)
    mp, cv2 = try_import_mediapipe()

    # smoothing window and stability counters
    vote_window = deque(maxlen=args.window)
    stability_candidate = None
    stability_count = 0

    # observe/confirm state
    observing_label = None   # label currently being observed for confirmation
    first_seen = 0.0         # timestamp when observing_label was first seen (after stability)
    last_committed = None    # last committed label (to avoid immediate duplicates)

    transcript = ""

    if mp and cv2:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
        cap = cv2.VideoCapture(args.cam)
        print("Press q to quit, c clear transcript, b delete last char, s add space.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            display_text = "No hand"
            pending_text = ""  # show pending confirmation timer

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                vec = []
                for p in lm.landmark:
                    vec += [p.x, p.y, p.z]
                # normalize if available
                try:
                    if _HAS_PREPROCESS:
                        xvec = preprocess.normalize_landmarks(np.array(vec))
                    else:
                        xvec = np.array(vec, dtype='float32')
                except Exception:
                    xvec = np.array(vec, dtype='float32')

                x = xvec.reshape(1, -1)
                pred = model.predict(x, verbose=0)[0]
                conf = float(pred.max())
                idx = int(pred.argmax())
                cur_label = le.classes_[idx] if le else str(idx)

                # threshold
                if conf < args.conf_thresh:
                    cur_label = "UNKNOWN"

                # smoothing: add to vote window and compute voted label
                vote_window.append(cur_label)
                voted = majority_vote(vote_window)
                display_text = f"{voted} ({conf:.2f})" if voted is not None else "UNKNOWN"

                # Stability to start observing
                if voted is not None and voted != "UNKNOWN":
                    if stability_candidate is None or voted != stability_candidate:
                        stability_candidate = voted
                        stability_count = 1
                    else:
                        stability_count += 1

                    # if stability reached, start observing/confirming
                    if stability_count >= args.accept_stability:
                        # if currently observing a different label, reset observation
                        if observing_label is None or observing_label != stability_candidate:
                            observing_label = stability_candidate
                            first_seen = time.time()
                        else:
                            # already observing same label -> check timeout
                            elapsed = time.time() - first_seen
                            remaining = max(0.0, args.confirm - elapsed)
                            pending_text = f"Pending '{observing_label}' ({remaining:.1f}s)"
                            # if observed long enough, commit
                            if elapsed >= args.confirm:
                                # commit action based on label content
                                lab = observing_label.lower().strip()
                                if lab == "space":
                                    transcript += " "
                                elif lab in ("del", "back", "delete"):
                                    transcript = transcript[:-1]
                                elif lab == "clear":
                                    transcript = ""
                                elif len(lab) == 1 and lab.isalpha():
                                    # avoid duplicate immediate commits of same char
                                    if last_committed != lab:
                                        transcript += lab
                                        last_committed = lab
                                else:
                                    # ignore other multi-token labels by default
                                    pass
                                # reset observing state after commit
                                observing_label = None
                                first_seen = 0.0
                                stability_candidate = None
                                stability_count = 0
                else:
                    # voted is UNKNOWN -> reset stability and observation
                    stability_candidate = None
                    stability_count = 0
                    observing_label = None
                    first_seen = 0.0

                # draw skeleton
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            else:
                # no hand
                vote_window.append(None)
                stability_candidate = None
                stability_count = 0
                observing_label = None
                first_seen = 0.0

            # draw texts
            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if observing_label:
                cv2.putText(frame, pending_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

            # draw transcript
            h, w = frame.shape[:2]
            draw_transcript(frame, transcript, org=(10, 110), font_scale=0.9, thickness=2, max_width=w-20)

            cv2.imshow("Realtime (observe->confirm)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                transcript = ""
            elif key == ord('b'):
                transcript = transcript[:-1]
            elif key == ord('s'):
                transcript += " "

        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

    else:
        # fallback simulated demo
        print("No MediaPipe; simulated demo. Press q to quit.")
        import cv2
        cap = cv2.VideoCapture(args.cam)
        vote_window = deque(maxlen=args.window)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            x = np.random.normal(0.5, 0.15, size=(1,63))
            pred = model.predict(x)[0]
            conf = float(pred.max())
            idx = int(pred.argmax())
            cur_label = le.classes_[idx] if le else str(idx)
            if conf < args.conf_thresh:
                cur_label = "UNKNOWN"
            vote_window.append(cur_label)
            voted = majority_vote(vote_window)
            display_text = f"{voted} ({conf:.2f})" if voted is not None else "UNKNOWN"

            # simplified observe-confirm: accept immediately after confirm seconds if persists
            # for simulated demo we accept when voted != last_committed and random condition
            cv2.putText(frame, display_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            draw_transcript(frame, transcript, org=(10,80), font_scale=0.9, thickness=2, max_width=frame.shape[1]-20)
            cv2.imshow("Simulated (observe->confirm)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()
