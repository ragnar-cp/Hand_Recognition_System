✋ Sign Language Interpreter – Starter Repository

A minimal, ready-to-run Sign Language Interpreter starter project using hand landmarks, a lightweight MLP model, and optional MediaPipe Hands for realtime detection.
This repo is built to run without errors even if MediaPipe is not installed — it automatically falls back to synthetic/random data for training & demo.

Perfect for beginners, students, prototypes, and anyone who wants a clean base project to extend into a full gesture-recognition system.

#Project Structure

project/
│
├── src/
│   ├── capture.py          # Capture hand landmarks for any label
│   ├── train.py            # Train a simple MLP classifier
│   ├── realtime.py         # Realtime gesture detection demo
│   ├── models/
│   │   └── mlp.py          # Simple MLP model builder
│   ├── utils.py            # Helper functions (saving/loading encoder, etc.)
│   └── voice.py            # Text-to-Speech using pyttsx3 (optional)
│
├── experiments/
│   └── best_model.h5       # Saved model after training (auto-created)
│
├── requirements.txt
└── README.md

🚀 Features

✔ Works even if MediaPipe is NOT installed. (recommended to use MediaPipe)
✔ Automatically switches to synthetic/fallback data
✔ Lightweight MLP classifier for fast training
✔ Realtime demo with optional text-to-speech
✔ Clean modular structure
✔ Easy to extend to CNN, LSTM, or full TF models

🏁 Getting Started

1️⃣ Create Virtual Environment & Install Requirements
*windows

	 python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt

*Linux/macOS

	 python -m venv venv
 	source venv/bin/activate
 	pip install -r requirements.txt
	python -m textblob.download_corpora

2️⃣ Capture Gesture Samples
	
Use this script to collect your own gesture dataset such as:
hello, yes, no, stop, ok, thanks, etc.
Example: 

	python -m src.capture --label hello --samples 200
General format:

	python -m src.capture.py --label <label_name> --samples <count>

The more samples you record, the better the model performs.

Captured data is automatically saved under:

data/<label_name>/

🧠 3️⃣ Train the Model

After collecting several gestures, train your classifier:

      python -m src.train --epochs 15 --augment-times 1


This generates:

experiments/best_model.h5
experiments/label_encoder.pkl

## OPTIONAL:-- Check the accuracy of the trained model....
	python -m src.evaluate


4️⃣ Run Realtime Interpreter

To run realtime gesture recognition:

      python -m src.realtime --conf-thresh 0.6 --window 8


Behavior:

If MediaPipe Hands is installed → realtime webcam detection

If NOT → simulated detection demo (no errors)


🔊 Optional: Enable Voice Output

voice.py uses pyttsx3 to convert predictions into speech.

If installed:

       pip install pyttsx3




