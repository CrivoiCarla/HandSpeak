import sys
import os
import cv2
import mediapipe as mp
import datetime
import time
import pickle
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFrame, QLineEdit, QRadioButton
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt


# ============================================================
# 2-Layer NN helpers (LOADS params dict: weight1,bias1,weight2,bias2)
# ============================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nn_forward(X, params):
    W1, b1 = params["weight1"], params["bias1"]
    W2, b2 = params["weight2"], params["bias2"]

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # If row-wise, transpose to (n_features, m)
    if X.shape[0] != W1.shape[1] and X.shape[1] == W1.shape[1]:
        X = X.T

    if X.shape[0] != W1.shape[1]:
        raise ValueError(
            f"Feature mismatch: model expects {W1.shape[1]} features, got {X.shape[0]}."
        )

    Z1 = (W1 @ X) + b1
    A1 = np.tanh(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = _sigmoid(Z2)
    return A2


def nn_predict(params, X_rowwise):
    X_rowwise = np.asarray(X_rowwise)
    if X_rowwise.ndim == 1:
        X_rowwise = X_rowwise.reshape(1, -1)

    X = X_rowwise.T  # (n_features, m)
    A2 = nn_forward(X, params)

    if A2.shape[0] == 1:
        return (A2[0] > 0.5).astype(int)

    return np.argmax(A2, axis=0).astype(int)


# ============================================================
# HandProcessor (MediaPipe)
# ============================================================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        clean_frame = frame.copy()
        display_frame = frame.copy()

        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_landmarks_list = []
        hand_detected = False
        bbox = None

        h, w = clean_frame.shape[:2]

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                xs, ys = [], []
                for lm in hand_landmarks.landmark:
                    hand_landmarks_list.extend([lm.x, lm.y, lm.z])
                    xs.append(lm.x)
                    ys.append(lm.y)

                x1 = int(max(0, min(xs) * w))
                x2 = int(min(w - 1, max(xs) * w))
                y1 = int(max(0, min(ys) * h))
                y2 = int(min(h - 1, max(ys) * h))

                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)

                bbox = (x1, y1, x2, y2)
                break

        return display_frame, clean_frame, hand_detected, hand_landmarks_list, bbox


# ============================================================
# Main UI
# ============================================================
class SIMPAC_Module1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIMPAC 2025 - Multi-Model Hand Detection")
        self.setFixedSize(1200, 850)
        self.setStyleSheet("background-color: #f8f9fa;")

        self.processor = HandProcessor()
        self.cap = cv2.VideoCapture(0)

        self.full_sentence = ""
        self.last_detected_char = ""
        self.selected_model = "Letters"

        # Debounce / cooldown
        self.last_append_time = 0.0
        self.append_cooldown_sec = 0.6

        # -----------------------------
        # PATHS
        # -----------------------------
        self.base_dir = os.path.join(os.path.dirname(__file__), "..")
        self.save_dir = os.path.join(self.base_dir, "saved_frames")
        os.makedirs(self.save_dir, exist_ok=True)

        # -----------------------------
        # LOAD MODELS
        # -----------------------------
        self.letters_model = None  # placeholder

        self.numbers_model = None
        self.numbers_model_path = os.path.join(
            self.base_dir, "SignLanguageDigits", "two_layer_nn_model.pkl"
        )

        try:
            with open(self.numbers_model_path, "rb") as f:
                self.numbers_model = pickle.load(f)

            if self.numbers_model is None:
                raise ValueError("Loaded object is None. Your .pkl likely saved None instead of parameters.")

            required = {"weight1", "bias1", "weight2", "bias2"}
            if (not isinstance(self.numbers_model, dict)) or (not required.issubset(self.numbers_model.keys())):
                raise ValueError(f"Loaded object is not a valid NN params dict. Expected keys: {required}")

            print("[OK] Loaded numbers model:", self.numbers_model_path)
            print("Keys:", self.numbers_model.keys())
            print("weight1 shape:", self.numbers_model["weight1"].shape)
            print("weight2 shape:", self.numbers_model["weight2"].shape)

        except Exception as e:
            self.numbers_model = None
            print(f"[ERROR] Unable to load numbers model from: {self.numbers_model_path}")
            print("        Details:", e)

        # -----------------------------
        # VIEWER STATE
        # -----------------------------
        self.viewer_mode = False
        self.saved_images = []
        self.viewer_index = 0

        self.init_ui()

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.run_logic)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_clock)
        self.ui_timer.start(1000)

    # -----------------------------
    # UI
    # -----------------------------
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header = QFrame()
        header.setStyleSheet("background-color: white; border-bottom: 4px solid #000080;")
        header.setFixedHeight(80)
        h_layout = QHBoxLayout(header)
        title = QLabel("SIGN LANGUAGE TRANSLATION - PHASE 1")
        title.setFont(QFont("Georgia", 18, QFont.Bold))
        title.setStyleSheet("color: #000080; border: none;")
        h_layout.addWidget(title, alignment=Qt.AlignCenter)
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()

        self.video_label = QLabel()
        self.video_label.setFixedSize(850, 500)
        self.video_label.setStyleSheet("background-color: black; border: 3px solid #000080; border-radius: 5px;")
        content_layout.addWidget(self.video_label)

        sidebar = QVBoxLayout()

        model_group_box = QFrame()
        model_group_box.setStyleSheet("background-color: #E8E8E8; border-radius: 10px; padding: 5px;")
        model_layout = QVBoxLayout(model_group_box)

        model_title = QLabel("SELECT MODEL")
        model_title.setFont(QFont("Arial", 10, QFont.Bold))
        model_layout.addWidget(model_title)

        self.model_letters_btn = QRadioButton("Model Letters (placeholder)")
        self.model_numbers_btn = QRadioButton("Model Digits (2-Layer NN)")
        self.model_letters_btn.setChecked(True)
        self.model_letters_btn.toggled.connect(lambda: self.change_model("Letters"))
        self.model_numbers_btn.toggled.connect(lambda: self.change_model("Numbers"))

        model_layout.addWidget(self.model_letters_btn)
        model_layout.addWidget(self.model_numbers_btn)
        sidebar.addWidget(model_group_box)

        # Buttons
        self.start_btn = QPushButton("START SCANNER")
        self.stop_btn = QPushButton("STOP")
        self.clear_btn = QPushButton("CLEAR TEXT")

        self.save_frame_btn = QPushButton("SAVE ROI (64x64 CLEAN)")
        self.view_saved_btn = QPushButton("VIEW SAVED")
        self.prev_btn = QPushButton("PREV")
        self.next_btn = QPushButton("NEXT")

        for btn in [self.start_btn, self.stop_btn, self.clear_btn,
                    self.save_frame_btn, self.view_saved_btn, self.prev_btn, self.next_btn]:
            btn.setFixedSize(220, 45)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)

        self.start_btn.setStyleSheet("background-color: #FFA500; border-radius: 8px; border: 1px solid black;")
        self.stop_btn.setStyleSheet("background-color: #E0E0E0; border-radius: 8px; border: 1px solid black;")
        self.clear_btn.setStyleSheet("background-color: #ff4d4d; color: white; border-radius: 8px;")

        self.save_frame_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 8px;")
        self.view_saved_btn.setStyleSheet("background-color: #2196F3; color: white; border-radius: 8px;")
        self.prev_btn.setStyleSheet("background-color: #dddddd; border-radius: 8px;")
        self.next_btn.setStyleSheet("background-color: #dddddd; border-radius: 8px;")

        # IMPORTANT: START exits viewer mode and returns to camera
        self.start_btn.clicked.connect(self.start_camera_mode)
        self.stop_btn.clicked.connect(lambda: self.video_timer.stop())
        self.clear_btn.clicked.connect(self.clear_text)

        self.save_frame_btn.clicked.connect(self.save_current_roi_frame)
        self.view_saved_btn.clicked.connect(self.toggle_viewer_mode)
        self.prev_btn.clicked.connect(self.viewer_prev)
        self.next_btn.clicked.connect(self.viewer_next)

        sidebar.addWidget(self.start_btn)
        sidebar.addWidget(self.stop_btn)
        sidebar.addWidget(self.clear_btn)

        sidebar.addSpacing(10)
        sidebar.addWidget(self.save_frame_btn)
        sidebar.addWidget(self.view_saved_btn)
        sidebar.addWidget(self.prev_btn)
        sidebar.addWidget(self.next_btn)

        self.clock_lbl = QLabel()
        self.date_lbl = QLabel()
        for lbl in [self.clock_lbl, self.date_lbl]:
            lbl.setStyleSheet("background-color: white; border: 1px solid gray; padding: 5px; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            sidebar.addWidget(lbl)

        sidebar.addStretch()
        content_layout.addLayout(sidebar)
        main_layout.addLayout(content_layout)

        sentence_frame = QFrame()
        sentence_frame.setStyleSheet("background-color: white; border: 2px solid #000080; border-radius: 10px;")
        sentence_layout = QVBoxLayout(sentence_frame)

        self.sentence_display = QLineEdit()
        self.sentence_display.setReadOnly(True)
        self.sentence_display.setFont(QFont("Arial", 20, QFont.Bold))
        self.sentence_display.setPlaceholderText("Detected text will appear here...")
        self.sentence_display.setStyleSheet("border: none; padding: 10px; color: #333;")
        sentence_layout.addWidget(self.sentence_display)

        main_layout.addWidget(sentence_frame)

        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.status_label.setStyleSheet("color: #000080; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    # -----------------------------
    # NEW: Start camera mode (exit viewer/preview)
    # -----------------------------
    def start_camera_mode(self):
        """
        If we are in preview/viewer mode, exit it and return to live camera mode.
        Then start the camera timer.
        """
        if self.viewer_mode:
            self.viewer_mode = False
            self.status_label.setText("CAMERA MODE")
            self.status_label.setStyleSheet("color: #000080;")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

        self.video_timer.start(30)

    # -----------------------------
    # BASIC ACTIONS
    # -----------------------------
    def change_model(self, model_name):
        self.selected_model = model_name
        self.status_label.setText(f"SWITCHED TO: {model_name.upper()} MODEL")

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S"))
        self.date_lbl.setText(now.strftime("%d %b %Y"))

    def clear_text(self):
        self.full_sentence = ""
        self.last_detected_char = ""
        self.sentence_display.setText("")

    # -----------------------------
    # ROI + FEATURES
    # -----------------------------
    def build_roi_64(self, clean_frame_bgr, bbox):
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        roi = clean_frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        return gray_64

    def build_features_from_image(self, gray_64):
        if gray_64 is None:
            return None
        return gray_64.flatten().reshape(1, -1).astype(np.float32)

    def predict_numbers(self, features):
        if self.numbers_model is None or features is None:
            return "?"

        try:
            pred_class = nn_predict(self.numbers_model, features)[0]
            return str(int(pred_class))
        except Exception as e:
            print("[ERROR] numbers predict failed:", e)
            return "?"

    # -----------------------------
    # SAVE / VIEWER (saved frames)
    # -----------------------------
    def refresh_saved_list(self):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        if not os.path.isdir(self.save_dir):
            self.saved_images = []
            return
        files = [f for f in os.listdir(self.save_dir) if f.lower().endswith(exts)]
        files.sort()
        self.saved_images = [os.path.join(self.save_dir, f) for f in files]

    def save_current_roi_frame(self):
        if self.viewer_mode:
            self.status_label.setText("EXIT VIEWER TO SAVE")
            self.status_label.setStyleSheet("color: #000080;")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("CAMERA READ ERROR")
            self.status_label.setStyleSheet("color: #000080;")
            return

        display_frame, clean_frame, hand_detected, _, bbox = self.processor.process_frame(frame)
        if not hand_detected:
            self.status_label.setText("NO HAND TO SAVE")
            self.status_label.setStyleSheet("color: #000080;")
            return

        gray_64 = self.build_roi_64(clean_frame, bbox)
        if gray_64 is None:
            self.status_label.setText("ROI ERROR")
            self.status_label.setStyleSheet("color: #000080;")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"roi64_{ts}.png"
        path = os.path.join(self.save_dir, filename)

        cv2.imwrite(path, gray_64)
        self.status_label.setText(f"SAVED CLEAN ROI: {filename}")
        self.status_label.setStyleSheet("color: green;")

        self.refresh_saved_list()

    def toggle_viewer_mode(self):
        self.viewer_mode = not self.viewer_mode

        if self.viewer_mode:
            self.refresh_saved_list()
            self.viewer_index = 0
            self.video_timer.stop()
            self.status_label.setText(f"VIEWER MODE ({len(self.saved_images)} images)")
            self.status_label.setStyleSheet("color: #000080;")

            self.prev_btn.setEnabled(len(self.saved_images) > 1)
            self.next_btn.setEnabled(len(self.saved_images) > 1)

            self.show_saved_image()
        else:
            self.status_label.setText("CAMERA MODE")
            self.status_label.setStyleSheet("color: #000080;")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def show_saved_image(self):
        if not self.saved_images:
            blank = np.zeros((500, 850, 3), dtype=np.uint8)
            self.display_bgr(blank)
            self.status_label.setText("VIEWER MODE - NO IMAGES FOUND")
            self.status_label.setStyleSheet("color: #000080;")
            return

        path = self.saved_images[self.viewer_index]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.status_label.setText("ERROR READING IMAGE")
            self.status_label.setStyleSheet("color: #000080;")
            return

        disp = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_NEAREST)
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        fname = os.path.basename(path)
        self.status_label.setText(f"VIEWING {self.viewer_index+1}/{len(self.saved_images)}: {fname}")
        self.status_label.setStyleSheet("color: #000080;")
        self.display_bgr(disp_bgr)

    def viewer_next(self):
        if not self.viewer_mode or not self.saved_images:
            return
        self.viewer_index = (self.viewer_index + 1) % len(self.saved_images)
        self.show_saved_image()

    def viewer_prev(self):
        if not self.viewer_mode or not self.saved_images:
            return
        self.viewer_index = (self.viewer_index - 1) % len(self.saved_images)
        self.show_saved_image()

    # -----------------------------
    # DISPLAY
    # -----------------------------
    def display_bgr(self, bgr_img):
        h, w, ch = bgr_img.shape
        qt_img = QImage(bgr_img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(850, 500, Qt.KeepAspectRatio))

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run_logic(self):
        if self.viewer_mode:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame, clean_frame, hand_detected, _, bbox = self.processor.process_frame(frame)

        if hand_detected:
            self.status_label.setText(f"SCANNING GESTURE ({self.selected_model})")
            self.status_label.setStyleSheet("color: green;")

            if self.selected_model == "Numbers":
                gray_64 = self.build_roi_64(clean_frame, bbox)
                features = self.build_features_from_image(gray_64)
                predicted_char = self.predict_numbers(features)
            else:
                predicted_char = "X"

            now_t = time.time()
            cooldown_ok = (now_t - self.last_append_time) >= self.append_cooldown_sec

            if predicted_char != self.last_detected_char and cooldown_ok:
                self.full_sentence += predicted_char
                self.sentence_display.setText(self.full_sentence)
                self.last_detected_char = predicted_char
                self.last_append_time = now_t
        else:
            self.status_label.setText("NO HAND DETECTED")
            self.status_label.setStyleSheet("color: #000080;")
            self.last_detected_char = ""

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.display_bgr(display_frame)

    def closeEvent(self, event):
        self.video_timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIMPAC_Module1()
    window.show()
    sys.exit(app.exec_())
