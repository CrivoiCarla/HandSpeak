import sys
import os
import cv2
import mediapipe as mp
import datetime
import time
import numpy as np

from collections import deque

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import load_model

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFrame, QLineEdit, QRadioButton
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt


# ============================================================
# DIGITS CNN
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# HandProcessor
# ============================================================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.2
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        clean_frame = frame.copy()
        display_frame = frame.copy()

        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_detected = False
        bbox = None
        h, w = clean_frame.shape[:2]

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # draw only on display_frame
                self.mp_draw.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                xs, ys = [], []
                for lm in hand_landmarks.landmark:
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

        return display_frame, clean_frame, hand_detected, bbox


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

        # Debounce / cooldown (scriere text)
        self.last_append_time = 0.0
        self.append_cooldown_sec = 0.6

        # -----------------------------
        # PATHS
        # -----------------------------
        self.base_dir = os.path.join(os.path.dirname(__file__), "..")  # with ".."
        self.save_dir = os.path.join(self.base_dir, "saved_frames")
        os.makedirs(self.save_dir, exist_ok=True)

        # -----------------------------
        # LABEL MAP (LETTERS) - 29 clase
        # -----------------------------
        self.letters_classes = [
            'A','B','C','D','E','F','G','H','I','J','K','L',
            'M','N','O','P','Q','R','S','T','U','V','W','X',
            'Y','Z','Blank','Space','Del'
        ]  # 29
        if len(self.letters_classes) != 29:
            raise ValueError("letters_classes must have 29 entries to match Dense(29).")

        # -----------------------------
        # LOAD DIGITS MODEL (PyTorch)
        # -----------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.digits_cnn_path = os.path.join(self.base_dir, "SignLanguageDigits", "cnn_digits_0_5_aug.pt")

        self.digits_cnn = None
        self.digits_num_classes = 6  # fallback
        try:
            ckpt = torch.load(self.digits_cnn_path, map_location=self.device, weights_only=False)
            if not isinstance(ckpt, dict) or "model_state" not in ckpt:
                raise ValueError("Digits checkpoint must be dict with key 'model_state'.")

            self.digits_num_classes = int(ckpt.get("num_new_classes", 10))
            model = SmallCNN(num_classes=self.digits_num_classes)
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device)
            model.eval()
            self.digits_cnn = model

            print("[OK] Loaded DIGITS CNN:", self.digits_cnn_path, "| device:", self.device,
                  "| num_classes:", self.digits_num_classes)
        except Exception as e:
            self.digits_cnn = None
            print("[ERROR] Digits model load failed:", e)

        # -----------------------------
        # LOAD LETTERS MODEL (Keras)
        # -----------------------------
        self.letters_model = None
        self.letters_model_path = os.path.join(self.base_dir, "SignLanguageLetter", "letters_model.h5")  # change if needed

        try:
            self.letters_model = load_model(self.letters_model_path)
            print("[OK] Loaded LETTERS model:", self.letters_model_path)
        except Exception as e:
            self.letters_model = None
            print("[ERROR] Letters model load failed:", e)

        # -----------------------------
        # SOFTMAX CONFIDENCE SMOOTHING (buffers)
        # -----------------------------
        self.digits_prob_buffer = deque(maxlen=12)
        self.letters_prob_buffer = deque(maxlen=20)

        # thresholds (tune)
        self.digits_conf_threshold = 0.55
        self.digits_margin_threshold = 0.1

        self.letters_conf_threshold = 0.45
        self.letters_margin_threshold = 0.1

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

        self.model_letters_btn = QRadioButton("Model Letters (Keras)")
        self.model_numbers_btn = QRadioButton("Model Digits (PyTorch)")
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
    # Camera mode
    # -----------------------------
    def start_camera_mode(self):
        if self.viewer_mode:
            self.viewer_mode = False
            self.status_label.setText("CAMERA MODE")
            self.status_label.setStyleSheet("color: #000080;")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        self.video_timer.start(30)

    # -----------------------------
    # Basic actions
    # -----------------------------
    def change_model(self, model_name):
        self.selected_model = model_name
        self.status_label.setText(f"SWITCHED TO: {model_name.upper()} MODEL")
        self.digits_prob_buffer.clear()
        self.letters_prob_buffer.clear()

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S"))
        self.date_lbl.setText(now.strftime("%d %b %Y"))

    def clear_text(self):
        self.full_sentence = ""
        self.last_detected_char = ""
        self.sentence_display.setText("")
        self.digits_prob_buffer.clear()
        self.letters_prob_buffer.clear()

    # -----------------------------
    # ROI: 64x64 grayscale normalized [0,1]
    # -----------------------------
    # def build_roi_64_gray_norm(self, clean_frame_bgr, bbox):
    #     if bbox is None:
    #         return None
    #
    #     x1, y1, x2, y2 = bbox
    #     roi = clean_frame_bgr[y1:y2, x1:x2]
    #     if roi.size == 0:
    #         return None
    #
    #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     gray_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    #     x = gray_64.astype(np.float32) / 255.0  # normalized
    #     return x  # shape (64,64) float32 in [0,1]

    def expand_bbox(self, bbox, pad, frame_shape):
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return x1, y1, x2, y2

    def build_roi_64_gray_norm(self, clean_frame_bgr, bbox):
        if bbox is None:
            return None

        pad = 40
        x1, y1, x2, y2 = self.expand_bbox(bbox, pad, clean_frame_bgr.shape)

        roi = clean_frame_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # 1) Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2) CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # 3) Normalize [0,255]
        gray_normalized = cv2.normalize(gray_enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # 4) Resize la 52x52 (ca 6px padding pe fiecare latură => 64x64 final)
        inner = cv2.resize(gray_normalized, (56, 56), interpolation=cv2.INTER_AREA)

        # 5) Padding 6px cu pixeli din margine
        # a) margine “copiată” (edge padding):
        gray_64 = cv2.copyMakeBorder(
            inner, 4, 4, 4, 4,
            borderType=cv2.BORDER_REPLICATE
        )

        # Alternativ, mai “smooth”:
        # gray_64 = cv2.copyMakeBorder(inner, 6, 6, 6, 6, cv2.BORDER_REFLECT_101)

        # 6) Normalizare finală [0,1]
        x = gray_64.astype(np.float32) / 255.0
        return x

    # -----------------------------
    # DIGITS: per-frame softmax probs (PyTorch)
    # -----------------------------
    def digits_probs(self, gray_norm_64):

        if self.digits_cnn is None or gray_norm_64 is None:
            return None

        x = torch.from_numpy(gray_norm_64).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,64,64)
        with torch.no_grad():
            logits = self.digits_cnn(x)  # (1,C)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (C,)

        print(probs)
        return probs.detach().cpu().numpy()

    # -----------------------------
    # LETTERS: per-frame probs
    # -----------------------------
    def letters_probs(self, gray_norm_64):
        if self.letters_model is None or gray_norm_64 is None:
            return None

        # Keras expects (1,64,64,1)
        x = gray_norm_64[..., np.newaxis][np.newaxis, ...]  # (1,64,64,1)
        probs = self.letters_model.predict(x, verbose=0)[0]  # (29,)
        print(np.argmax(probs))
        return probs.astype(np.float32)

    # -----------------------------
    # Decision from buffer: mean probs + threshold + margin
    # -----------------------------
    @staticmethod
    def decide_from_buffer(prob_buffer, conf_th, margin_th):
        if len(prob_buffer) < prob_buffer.maxlen:
            return None, None, None

        avg_probs = np.mean(np.stack(prob_buffer, axis=0), axis=0)
        top1 = int(np.argmax(avg_probs))

        sorted_probs = np.sort(avg_probs)
        top1_conf = float(sorted_probs[-1])
        top2_conf = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
        margin = top1_conf - top2_conf

        if top1_conf >= conf_th and margin >= margin_th:
            return top1, top1_conf, margin

        return None, top1_conf, margin

    # -----------------------------
    # SAVE / VIEWER
    # -----------------------------
    def refresh_saved_list(self):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
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

        display_frame, clean_frame, hand_detected, bbox = self.processor.process_frame(frame)
        if not hand_detected:
            self.status_label.setText("NO HAND TO SAVE")
            self.status_label.setStyleSheet("color: #000080;")
            return

        gray_norm = self.build_roi_64_gray_norm(clean_frame, bbox)
        if gray_norm is None:
            self.status_label.setText("ROI ERROR")
            self.status_label.setStyleSheet("color: #000080;")
            return

        # save as grayscale image (0..255)
        gray_u8 = (gray_norm * 255.0).clip(0, 255).astype(np.uint8)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"roi64_{ts}.png"
        path = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, gray_u8)

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
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        ))
g
    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run_logic(self):
        if self.viewer_mode:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame, clean_frame, hand_detected, bbox = self.processor.process_frame(frame)

        predicted_char = None
        info = ""

        if hand_detected:
            self.status_label.setStyleSheet("color: green;")

            gray_norm = self.build_roi_64_gray_norm(clean_frame, bbox)

            if self.selected_model == "Numbers":
                probs = self.digits_probs(gray_norm)
                if probs is not None:
                    self.digits_prob_buffer.append(probs)

                cls, conf, margin = self.decide_from_buffer(
                    self.digits_prob_buffer,
                    self.digits_conf_threshold,
                    self.digits_margin_threshold
                )

                if conf is not None:
                    info = f" conf={conf:.2f} margin={margin:.2f} buf={len(self.digits_prob_buffer)}/{self.digits_prob_buffer.maxlen}"
                self.status_label.setText(f"SCANNING DIGITS (softmax){info}")

                if cls is not None:
                    predicted_char = str(cls)
                    self.digits_prob_buffer.clear()

            else:
                probs = self.letters_probs(gray_norm)
                if probs is not None:
                    self.letters_prob_buffer.append(probs)

                cls, conf, margin = self.decide_from_buffer(
                    self.letters_prob_buffer,
                    self.letters_conf_threshold,
                    self.letters_margin_threshold
                )

                if conf is not None:
                    info = f" conf={conf:.2f} margin={margin:.2f} buf={len(self.letters_prob_buffer)}/{self.letters_prob_buffer.maxlen}"
                self.status_label.setText(f"SCANNING LETTERS (softmax){info}")

                if cls is not None:
                    predicted_char = self.letters_classes[cls]  # map index -> label
                    self.letters_prob_buffer.clear()

            # Append only if stable char decided
            if predicted_char is not None:
                now_t = time.time()
                cooldown_ok = (now_t - self.last_append_time) >= self.append_cooldown_sec

                if predicted_char != self.last_detected_char and cooldown_ok:
                    # Optional: if predicted_char is 'Blank' ignore (or handle differently)
                    if predicted_char == "Blank":
                        pass
                    else:
                        self.full_sentence += predicted_char
                        self.sentence_display.setText(self.full_sentence)

                    self.last_detected_char = predicted_char
                    self.last_append_time = now_t

        else:
            self.status_label.setText("NO HAND DETECTED")
            self.status_label.setStyleSheet("color: #000080;")
            self.last_detected_char = ""
            self.digits_prob_buffer.clear()
            self.letters_prob_buffer.clear()

        # bbox for display
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
