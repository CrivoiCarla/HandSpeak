import sys
import cv2
import mediapipe as mp
import numpy as np
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal


class HandProcessor:
    """Class dedicated to MediaPipe logic - Easy for teammates to extend"""

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
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_landmarks_list = []
        is_raised = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton like in the reference project
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extract landmarks for future ML training
                # Teammates will use this 'hand_landmarks_list' later
                for lm in hand_landmarks.landmark:
                    hand_landmarks_list.extend([lm.x, lm.y, lm.z])

                # Current Logic: Check if Index Finger Tip (ID 8) is raised
                if hand_landmarks.landmark[8].y < 0.3:
                    is_raised = True

        return frame, is_raised, hand_landmarks_list


class SIMPAC_Module1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIMPAC 2025 - Phase 1: Hand Detection")
        self.setFixedSize(1200, 800)
        self.setStyleSheet("background-color: #f8f9fa;")

        self.processor = HandProcessor()
        self.cap = cv2.VideoCapture(0)

        self.init_ui()

        # Timers for UI and Video
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.run_logic)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_clock)
        self.ui_timer.start(1000)

    def init_ui(self):
        """UI structured exactly like the reference project"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header - Professional Branding
        header = QFrame()
        header.setStyleSheet("background-color: white; border-bottom: 4px solid #000080;")
        header.setFixedHeight(100)
        h_layout = QHBoxLayout(header)
        title = QLabel("SIGN LANGUAGE SYSTEM - DATA ACQUISITION PHASE")
        title.setFont(QFont("Georgia", 18, QFont.Bold))
        title.setStyleSheet("color: #000080; border: none;")
        h_layout.addWidget(title, alignment=Qt.AlignCenter)
        main_layout.addWidget(header)

        # Main Content
        content_layout = QHBoxLayout()

        # Video Display Area
        self.video_label = QLabel()
        self.video_label.setFixedSize(850, 550)
        self.video_label.setStyleSheet("background-color: black; border: 3px solid #000080; border-radius: 5px;")
        content_layout.addWidget(self.video_label)

        # Sidebar Controls
        sidebar = QVBoxLayout()

        self.start_btn = QPushButton("START SCANNER")
        self.stop_btn = QPushButton("STOP")

        for btn in [self.start_btn, self.stop_btn]:
            btn.setFixedSize(220, 60)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)

        self.start_btn.setStyleSheet("background-color: #FFA500; border-radius: 8px; border: 1px solid black;")
        self.stop_btn.setStyleSheet("background-color: #E0E0E0; border-radius: 8px; border: 1px solid black;")

        self.start_btn.clicked.connect(lambda: self.video_timer.start(30))
        self.stop_btn.clicked.connect(self.stop_all)

        sidebar.addWidget(self.start_btn)
        sidebar.addWidget(self.stop_btn)

        # Clock & Date like in reference
        self.clock_lbl = QLabel()
        self.date_lbl = QLabel()
        for lbl in [self.clock_lbl, self.date_lbl]:
            lbl.setStyleSheet("background-color: white; border: 1px solid gray; padding: 5px;")
            lbl.setAlignment(Qt.AlignCenter)
            sidebar.addWidget(lbl)

        sidebar.addStretch()
        content_layout.addLayout(sidebar)
        main_layout.addLayout(content_layout)

        # Status Footer
        self.status_label = QLabel("SYSTEM IDLE")
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setStyleSheet("color: #000080; background-color: white; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S"))
        self.date_lbl.setText(now.strftime("%d %b %Y"))

    def stop_all(self):
        self.video_timer.stop()
        self.status_label.setText("SYSTEM STOPPED")

    def run_logic(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame through our modular class
            processed_frame, is_raised, raw_data = self.processor.process_frame(frame)

            # Phase 1 Logic: Just detect if raised
            if is_raised:
                self.status_label.setText("PHASE 1: HAND DETECTED (READY FOR ML)")
                self.status_label.setStyleSheet("color: green; background-color: white; padding: 10px;")
            else:
                self.status_label.setText("PHASE 1: SCANNING...")
                self.status_label.setStyleSheet("color: red; background-color: white; padding: 10px;")

            # Display logic
            h, w, ch = processed_frame.shape
            qt_img = QImage(processed_frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(850, 550, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIMPAC_Module1()
    window.show()
    sys.exit(app.exec_())