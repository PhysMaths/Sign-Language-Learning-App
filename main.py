import sys
import os
import cv2
from datetime import timedelta, datetime
import json
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import mediapipe as mp
import pickle
import numpy as np
import random
from collections import Counter, defaultdict


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

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


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

        self.analytics_button = QPushButton("See Analytics")
        self.analytics_button.clicked.connect(self.show_analytics)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.start_button)
        layout.addWidget(self.analytics_button)

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

    def compute_analytics(self, path="reviews.jsonl"):
        total = 0
        correct = 0
        quality_counts = Counter()
        reviews_per_day = defaultdict(int)

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    total += 1
                    if ev.get("correct"):
                        correct += 1
                    q = int(ev.get("quality", 0))
                    quality_counts[q] += 1
                    day = ev["ts"][:10]  # YYYY-MM-DD
                    reviews_per_day[day] += 1
        except FileNotFoundError:
            pass

        accuracy = (correct / total * 100) if total else 0.0

        return {
            "total_reviews": total,
            "accuracy_pct": round(accuracy, 1),
            "quality_counts": dict(quality_counts),
            "reviews_per_day": dict(sorted(reviews_per_day.items())),
        }

    def show_analytics(self):
        a = self.compute_analytics()
        text = (
            f"Total reviews: {a['total_reviews']}\n"
            f"Accuracy: {a['accuracy_pct']}%\n\n"
            f"Quality counts: {a['quality_counts']}\n"
        )
        QMessageBox.information(self, "Analytics", text)


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Preview")

        self.closed = False

        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        self.progress = {}
        self.path = "progress.json"

        self.due_queue = []

        # makes a new progress dictionary every time change for the final implementation
        if not os.path.exists(self.path):
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

        self.current_number = self.next_number()
        if self.current_number is None:
            return
        
        
        self.question = QLabel(self.current_number)
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
    
    def build_due_queue(self):
        now = datetime.now()
        due = [n for n in self.numbers if datetime.fromisoformat(self.progress[n]["due"]) <= now]
        random.shuffle(due)
        self.due_queue = due
    
    def next_number(self):
        if not self.due_queue:
            self.build_due_queue()
            if not self.due_queue:
                self.go_home()
                self.closed = True 
                return None
        n = self.due_queue.pop()
        return n


    def see_answer(self):
        img_path = "pictures/" + self.current_number + ".png"
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

            self.current_number = self.next_number()
            if self.current_number is None:
                return
            self.question.setText(self.current_number)

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
                
                
            probs = model.predict_proba([np.asarray(data_aux)])[0]

            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])

            print(confidence)

            if confidence >= 0.8:

                sign_prediction = labels_dict[int(model.classes_[best_idx])]

                if sign_prediction == self.current_number:
                    quality = self.choose_difficulty()

                    self.sm2_update(quality)

                    self.current_number = self.next_number()   

                    self.question.setText(self.current_number)


        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))

    
    def log_review(self, number, quality, before, after, correct=True):
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "card_id": number,
            "quality": quality,
            "correct": correct,
            "before": before,
            "after": after,
        }
        with open("reviews.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    
    def save_progress(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2)

    def sm2_update(self, quality):
        key = self.current_number

        before = {
            "repetitions": self.progress[key]["repetitions"],
            "interval": self.progress[key]["interval"],
            "ease_factor": self.progress[key]["ease_factor"],
            "due": self.progress[key]["due"],
            "lapses": self.progress[key]["lapses"],
        }

        ef = before["ease_factor"]
        reps = before["repetitions"]
        interval = before["interval"]

        if quality < 3:
            reps = 0
            interval = 1
            self.progress[key]["lapses"] += 1
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

        self.progress[key]["ease_factor"] = ef
        self.progress[key]["repetitions"] = reps
        self.progress[key]["interval"] = interval
        self.progress[key]["due"] = (datetime.now() + timedelta(days=interval)).isoformat()

        after = {
            "repetitions": reps,
            "interval": interval,
            "ease_factor": ef,
            "due": self.progress[key]["due"],
            "lapses": self.progress[key]["lapses"],
        }

        self.log_review(key, quality, before, after, correct=(quality >= 3))

        self.save_progress()


    def choose_difficulty(self):
        msg = QMessageBox(self)
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


    def closeEvent(self, event):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
    
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
