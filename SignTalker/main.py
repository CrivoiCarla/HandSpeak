import sys
import cv2
import mediapipe as mp
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame, QLineEdit, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt


class HandProcessor:
    """Class dedicated to MediaPipe logic"""

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_landmarks_list = []
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton landmarks on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Collect all 21 landmarks (x, y, z)
                for lm in hand_landmarks.landmark:
                    hand_landmarks_list.extend([lm.x, lm.y, lm.z])

        return frame, hand_detected, hand_landmarks_list


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
        self.selected_model = "Letters"  # Default model

        self.init_ui()

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.run_logic)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_clock)
        self.ui_timer.start(1000)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Header Area
        header = QFrame()
        header.setStyleSheet("background-color: white; border-bottom: 4px solid #000080;")
        header.setFixedHeight(80)
        h_layout = QHBoxLayout(header)
        title = QLabel("SIGN LANGUAGE TRANSLATION - PHASE 1")
        title.setFont(QFont("Georgia", 18, QFont.Bold))
        title.setStyleSheet("color: #000080; border: none;")
        h_layout.addWidget(title, alignment=Qt.AlignCenter)
        main_layout.addWidget(header)

        # 2. Main Content
        content_layout = QHBoxLayout()

        # Video Section
        self.video_label = QLabel()
        self.video_label.setFixedSize(850, 500)
        self.video_label.setStyleSheet("background-color: black; border: 3px solid #000080; border-radius: 5px;")
        content_layout.addWidget(self.video_label)

        # Sidebar Section
        sidebar = QVBoxLayout()

        # Model Selection Group
        model_group_box = QFrame()
        model_group_box.setStyleSheet("background-color: #E8E8E8; border-radius: 10px; padding: 5px;")
        model_layout = QVBoxLayout(model_group_box)

        model_title = QLabel("SELECT MODEL")
        model_title.setFont(QFont("Arial", 10, QFont.Bold))
        model_layout.addWidget(model_title)

        self.model_letters_btn = QRadioButton("Model Litere (SVM)")
        self.model_numbers_btn = QRadioButton("Model Cifre (SVM)")
        self.model_letters_btn.setChecked(True)
        self.model_letters_btn.toggled.connect(lambda: self.change_model("Letters"))
        self.model_numbers_btn.toggled.connect(lambda: self.change_model("Numbers"))

        model_layout.addWidget(self.model_letters_btn)
        model_layout.addWidget(self.model_numbers_btn)
        sidebar.addWidget(model_group_box)

        # Action Buttons
        self.start_btn = QPushButton("START SCANNER")
        self.stop_btn = QPushButton("STOP")
        self.clear_btn = QPushButton("CLEAR TEXT")

        for btn in [self.start_btn, self.stop_btn, self.clear_btn]:
            btn.setFixedSize(220, 50)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)

        self.start_btn.setStyleSheet("background-color: #FFA500; border-radius: 8px; border: 1px solid black;")
        self.stop_btn.setStyleSheet("background-color: #E0E0E0; border-radius: 8px; border: 1px solid black;")
        self.clear_btn.setStyleSheet("background-color: #ff4d4d; color: white; border-radius: 8px;")

        self.start_btn.clicked.connect(lambda: self.video_timer.start(30))
        self.stop_btn.clicked.connect(lambda: self.video_timer.stop())
        self.clear_btn.clicked.connect(self.clear_text)

        sidebar.addWidget(self.start_btn)
        sidebar.addWidget(self.stop_btn)
        sidebar.addWidget(self.clear_btn)

        self.clock_lbl = QLabel()
        self.date_lbl = QLabel()
        for lbl in [self.clock_lbl, self.date_lbl]:
            lbl.setStyleSheet("background-color: white; border: 1px solid gray; padding: 5px; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            sidebar.addWidget(lbl)

        sidebar.addStretch()
        content_layout.addLayout(sidebar)
        main_layout.addLayout(content_layout)

        # 3. Sentence Display Area
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

        # 4. Status Footer
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.status_label.setStyleSheet("color: #000080; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

    def change_model(self, model_name):
        self.selected_model = model_name
        self.status_label.setText(f"SWITHCED TO: {model_name.upper()} MODEL")

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S"))
        self.date_lbl.setText(now.strftime("%d %b %Y"))

    def clear_text(self):
        self.full_sentence = ""
        self.last_detected_char = ""
        self.sentence_display.setText("")

    def run_logic(self):
        ret, frame = self.cap.read()
        if ret:
            processed_frame, hand_detected, raw_data = self.processor.process_frame(frame)

            if hand_detected:
                self.status_label.setText(f"SCANNING GESTURE ({self.selected_model})")
                self.status_label.setStyleSheet("color: green;")

                # After model created: here you will call
                # predicted_char = self.active_model.predict(raw_data)
                simulated_char = "X" if self.selected_model == "Letters" else "5"

                if simulated_char != self.last_detected_char:
                    self.full_sentence += simulated_char
                    self.sentence_display.setText(self.full_sentence)
                    self.last_detected_char = simulated_char
            else:
                self.status_label.setText("NO HAND DETECTED")
                self.status_label.setStyleSheet("color: #000080;")
                self.last_detected_char = ""

            h, w, ch = processed_frame.shape
            qt_img = QImage(processed_frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(850, 500, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIMPAC_Module1()
    window.show()
    sys.exit(app.exec_())