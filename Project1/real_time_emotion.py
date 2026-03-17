import os
import sys
import cv2
import pickle
import numpy as np
from training import ModifiedPerceptron
from data_processing import preprocess_image

MODEL_PATH = "weights.pkl"
# i keep model path here so i can load saved model easily
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# i get script folder so i can build correct paths for files
# try to find face cascade in common places then use local haarcascade_frontalface_default folder
if getattr(cv2, "data", None) is not None and hasattr(cv2.data, "haarcascades"):
    HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
else:
    HAAR_PATH = os.path.join(SCRIPT_DIR, "haarcascade_frontalface_default", "haarcascade_frontalface_default.xml")
# i choose final haar path so i always have face detector file

def _gui_available():
    # i test here if opencv window can open on this system
    try:
        test = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("_test_", test)
        cv2.destroyWindow("_test_")
        return True
    except cv2.error:
        return False

USE_GUI = _gui_available()
# i store gui flag so later i decide to show window or save video
if not USE_GUI:
    print("Note: OpenCV GUI not available. Will save to video file instead.")

if not os.path.isfile(MODEL_PATH):
    print("Run app.py first to train and save emotion_model.pkl")
    sys.exit(1)

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    save_dict = pickle.load(f)

# i rebuild same model here so i can use weights from training file
model = ModifiedPerceptron(
    input_size=save_dict["model_params"]["input_size"],
    hidden_sizes=tuple(save_dict["model_params"]["hidden_sizes"]),
    output_size=save_dict["model_params"]["output_size"],
    dropout=save_dict["model_params"].get("dropout", 0.5),
)
model.set_params_dict(save_dict["model_params"])
CLASSES = save_dict["CLASSES"]
# i keep class names so i can turn index prediction into emotion word

if not os.path.isfile(HAAR_PATH):
    print(f"Haar cascade not found at: {HAAR_PATH}")
    sys.exit(1)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    print("Could not load Haar cascade. Check OpenCV installation.")
    sys.exit(1)


def preprocess_face(face_img):
    """Preprocess one face like in training."""
    # i call same preprocess that i used in training so data stays consistent
    return preprocess_image(face_img, use_clahe=True)


def main():
    # i open webcam here so i can read frames one by one
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    writer = None
    if USE_GUI:
        print("Real-time emotion recognition. Press 'q' in the window to quit.")
    else:
        out_path = "realtime_output.avi"
        # i choose fps and size from camera so saved video looks same as input
        fps = min(20.0, cap.get(cv2.CAP_PROP_FPS) or 20.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        print(f"Saving to {out_path}. Press Ctrl+C to stop (or run for ~30 sec).")

    frame_count = 0
    try:
        max_frames_no_gui = int(30 * (cap.get(cv2.CAP_PROP_FPS) or 20))  # ~30 sec
        # i limit frames without gui so script does not run forever when saving video
        while True:
            # i grab one frame from webcam for each loop
            ret, frame = cap.read()
            if not ret:
                break
            # i convert frame to gray so face detector works better
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                # i crop face region so model only sees face area
                face_roi = frame[y : y + h, x : x + w]
                x_vec = preprocess_face(face_roi).reshape(1, -1)
                # i get prediction probabilities from model for this one face
                probs = model.predict_proba(x_vec)[0]
                pred_idx = int(np.argmax(probs))
                emotion = CLASSES[pred_idx]
                conf = float(np.max(probs))
                # i change color based on confidence so i can feel trust in prediction
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{emotion} ({conf:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if USE_GUI:
                cv2.imshow("Real-Time Emotion Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # i write processed frame to video file when gui is not allowed
                writer.write(frame)
                frame_count += 1
                if frame_count >= max_frames_no_gui:
                    print(f"Saved {frame_count} frames to {out_path}")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if USE_GUI:
            cv2.destroyAllWindows()
        elif writer is not None:
            writer.release()
            # i show how many frames i saved so i know recording length
            print(f"Saved {frame_count} frames to realtime_output.avi")


if __name__ == "__main__":
    main()
