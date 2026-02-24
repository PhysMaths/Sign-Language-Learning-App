import sys
import os
import cv2
from datetime import date, timedelta, datetime
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

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

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

        # makes a new progress dictionary every time change for the final implementation
        if not os.path.exists(self.path) or os.path.exists(self.path):
            now = datetime.now().isoformat()

            self.progress = {
                number : {
                    "repetitions" : 0,
                    "interval" : 0,
                    "ease_factor" : 2.5,
                    "due" : now,
                    "lapses": 0

                }
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

        self.answer_button = QPushButton("See Answer")
        self.answer_button.clicked.connect(self.see_answer)

        layout = QVBoxLayout()
        layout.addWidget(self.question)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.answer_button)

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

    def see_answer(self):
        img_path = "pictures/" + self.numbers[self.index] + ".png"
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            message = QMessageBox(self)
            message.setText("")
            if not pixmap.isNull():
                message.setIconPixmap(
                    pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            message.exec_()

            self.sm2_update(1)

            while not self.is_valid(self.index):
                self.index += 1
                if self.index >= len(self.progress):
                    self.go_home()  
                    return   

            self.question.setText(self.numbers[self.index])

        return

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

            sign_prediction = labels_dict[int(prediction[0])]

            if sign_prediction == self.numbers[self.index]:
                quality = self.choose_difficulty()

                self.sm2_update(quality)

                while not self.is_valid(self.index):
                    self.index += 1
                    if self.index >= len(self.progress):
                        self.go_home()  
                        return      

                self.question.setText(self.numbers[self.index])

            
            # Display prediction on frame
            cv2.putText(frame,
                        f'Prediction: {sign_prediction}',
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


    from datetime import datetime, timedelta

    def sm2_update(self, quality):
        ef = self.progress[self.numbers[self.index]]["ease_factor"]
        reps = self.progress[self.numbers[self.index]]["repetitions"]
        interval = self.progress[self.numbers[self.index]]["interval"]

        if quality < 3:
            reps = 0
            interval = 1
            self.progress[self.numbers[self.index]]["lapses"] += 1
        else:
            reps += 1

            if reps == 1:
                interval = 1
            elif reps == 2:
                interval = 6
            else:
                interval = round(interval * ef)

        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ef = max(1.3, ef)

        self.progress[self.numbers[self.index]]["ease_factor"] = ef
        self.progress[self.numbers[self.index]]["repetitions"] = reps
        self.progress[self.numbers[self.index]]["interval"] = interval
        self.progress[self.numbers[self.index]]["due"] = (datetime.now() + timedelta(days=interval)).isoformat()

    def choose_difficulty(self):
        msg = QMessageBox()
        msg.setWindowTitle("Difficulty")
        msg.setText("Well Done! How hard was it to recall?:")

        buttons = {}

        for quality in ['AGAIN', 'HARD', 'GOOD', 'EASY']:
            button = msg.addButton(quality, QMessageBox.ActionRole)
            buttons[button] = quality

        msg.exec()

        quality_to_number = {
            'AGAIN' : 1,
            'HARD' : 3,
            'GOOD' : 4, 
            'EASY' : 5
        }

        clicked_button = msg.clickedButton()

        if clicked_button in buttons:
            chosen_quality = buttons[clicked_button]
            chosen_number = quality_to_number[chosen_quality]
            return chosen_number

        return 1

                
    def is_valid(self, index):
        if datetime.fromisoformat(self.progress[self.numbers[index]]["due"]) <= datetime.now():
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
