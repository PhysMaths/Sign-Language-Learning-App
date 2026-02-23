import sys
import os
import cv2
from datetime import date, timedelta
import json
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import mediapipe as mp
import pickle
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QMessageBox,
    QPushButton
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle.load(open('./1to10.p', 'rb'))
model = model_dict['model']

labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}

class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")

        title = QLabel("Sign Language Practise")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24))

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_session)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.webcam_window = None

    
    def start_session(self):
        self.webcam_window = WebcamWindow()
        self.webcam_window.resize(800, 600)
        if not self.webcam_window.closed:
            self.webcam_window.show()
        self.close()


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Preview")

        self.closed = False

        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.index = 0

        self.progress = {}
        self.path = "progress.json"

        if not os.path.exists(self.path):
            today = date.today().isoformat()

            self.progress = {
                number : {"interval": 1, "next_due": today, "streak": 0}
                for number in self.numbers
            }

            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.progress =  json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        
        # get the first valid letter
        while not self.is_valid(self.index):
            self.index += 1
            if self.index >= len(self.progress):
                self.go_home()
                self.closed = True
                return
        
        self.question = QLabel(self.numbers[self.index])
        self.question.setAlignment(Qt.AlignCenter)
        self.question.setFont(QFont("Arial", 24))

        self.label = QLabel("Starting camera…")
        self.label.setScaledContents(True)

        self.capture_button = QPushButton("Take Picture")
        self.capture_button.clicked.connect(self.analyse_picture)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 24))

        layout = QVBoxLayout()
        layout.addWidget(self.question)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label.setText("Could not open webcam.")
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.latest_frame = None

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        self.latest_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      

        data_aux = []
            
        results = hands.process(frame)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                
                
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Display prediction on frame
            cv2.putText(frame,
                        f'Prediction: {predicted_character}',
                        (50, 50),                      # Position (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,      # Font
                        1.5,                           # Font scale
                        (0, 255, 0),                   # Color (BGR)
                        3,                             # Thickness
                        cv2.LINE_AA)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))

    def analyse_picture(self):
        if self.latest_frame is None:
            return
        
        self.result_label.setText("")
        
        data_aux = []

        frame = self.latest_frame

        frame_rgb = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)
         
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])

        labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}

        sign_prediction = str(labels_dict[int(prediction[0])])
        
        if sign_prediction == self.numbers[self.index]:
            self.result_label.setText("Correct!")
            self.progress[self.numbers[self.index]]["interval"] = min(self.progress[self.numbers[self.index]]["interval"] * 2, 30)
        else:
            print("got here to the wrong part")
            self.result_label.setText("Wrong")
            self.progress[self.numbers[self.index]]["interval"] = 1
            # self.show_wrong_popup()
        
        interval = self.progress[self.numbers[self.index]]["interval"]
        next_due = self.progress[self.numbers[self.index]]["next_due"]
        d = date.fromisoformat(next_due)
        d += timedelta(days=interval)
        self.progress[self.numbers[self.index]]["next_due"] = d.isoformat()

        while not self.is_valid(self.index):
            self.index += 1
            if self.index >= len(self.progress):
                self.go_home()  
                return      

        self.question.setText(self.numbers[self.index])
                
    def is_valid(self, index):
        if self.progress[self.numbers[index]]["next_due"] == date.today().isoformat():
            return True
        return False

    def closeEvent(self, event):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2)

        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().closeEvent(event)

    def show_wrong_popup(self):
        img_path = "images/" + self.letters[self.index] + ".png"
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            message = QMessageBox(self)
            message.setText("")
            if not pixmap.isNull():
                message.setIconPixmap(
                    pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            message.exec_()
    
    def go_home(self):
        QMessageBox.information(self, "All done", "All cards for today are finished.")
        self.home = HomeWindow()
        self.home.resize(800, 600)
        self.home.show()
        self.close()
    
def main():

    app = QApplication(sys.argv)
    window = HomeWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
