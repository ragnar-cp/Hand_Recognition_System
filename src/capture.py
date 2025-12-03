"""
capture.py
Capture hand landmarks using MediaPipe and save to CSV.
Draws a skeleton overlay during capture (can be disabled with --no-skeleton).
If MediaPipe/OpenCV are not available, generates synthetic landmark CSV data as fallback.
"""

import argparse
import os
import csv
import time
import numpy as np


def write_synthetic(label, samples, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"label_{label}.csv")
    print(f"[Synthetic] Creating dataset: {path} ({samples} samples)")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(samples):
            lm = np.random.normal(loc=0.5, scale=0.15, size=(63,))
            row = lm.tolist() + [label]
            writer.writerow(row)
    print("[Synthetic] Done.")


def try_capture_with_mediapipe(label, samples, outdir, cam, draw_skeleton=True):
    """
    Capture landmarks using MediaPipe. Returns True if capture executed (even partial),
    False if MediaPipe/OpenCV is not available.
    """
    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        print("MediaPipe/OpenCV capture not available in this environment:", e)
        return False

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Cannot open camera (index {}).".format(cam))
        return False

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"label_{label}.csv")
    print(f"[Capture] Saving to: {path}")
    collected = 0

    # Open CSV in append mode so you can add more samples later
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        print("Starting capture. Press 'q' to quit early.")
        while collected < samples:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available, exiting.")
                break

            # Mirror for natural front-camera feeling
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]

                # Draw skeleton overlay if requested
                if draw_skeleton:
                    try:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2)
                        )
                    except Exception:
                        # non-fatal if drawing fails
                        pass

                # Build landmark vector: 21 * (x,y,z)
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]

                # Append label and write to CSV
                writer.writerow(row + [label])
                collected += 1
                cv2.putText(frame, f"Saved {collected}/{samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # no hand found
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow(f"Capture - {label} (q to quit)", frame)
            # press q to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture aborted by user.")
                break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    print(f"Capture completed: {collected}/{samples} samples saved to {path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label name for this sign (e.g., hello)")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
    parser.add_argument("--outdir", default="data/landmarks", help="Output directory for CSVs")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable skeleton overlay during capture")
    args = parser.parse_args()

    # try mediapipe capture; if fails, create synthetic
    ok = try_capture_with_mediapipe(args.label, args.samples, args.outdir, args.cam, draw_skeleton=not args.no_skeleton)
    if not ok:
        print("[FALLBACK] Creating synthetic data instead.")
        write_synthetic(args.label, args.samples, args.outdir)


if __name__ == "__main__":
    main()
