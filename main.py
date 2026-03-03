import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
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

model_dict = pickle.load(open("./1to10.p", "rb"))
model = model_dict["model"]

labels_dict = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",
    9: "10",
}

APP_STYLE = """
QMainWindow {
    background: #f2eee5;
}
QWidget#root {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #f8f4ea,
        stop: 0.55 #efe4d0,
        stop: 1 #dce9da
    );
}
QFrame#panel {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(78, 96, 84, 0.14);
    border-radius: 28px;
}
QLabel#heroTitle {
    color: #203226;
    font-size: 34px;
    font-weight: 700;
}
QLabel#heroSubtitle {
    color: #536356;
    font-size: 15px;
}
QLabel#sectionLabel {
    color: #3a5642;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
QLabel#numberBadge {
    background: #203a2a;
    color: #ffffff;
    border-radius: 22px;
    padding: 14px 28px;
    font-size: 30px;
    font-weight: 700;
}
QLabel#cameraLabel {
    background: #faf7f0;
    border: 1px solid #d9d2c6;
    border-radius: 22px;
    color: #5c675d;
    padding: 12px;
}
QLabel#statCard {
    background: #faf7f0;
    color: #2b4332;
    border: 1px solid #e2dbcf;
    border-radius: 18px;
    padding: 16px;
    font-size: 15px;
}
QPushButton {
    min-height: 52px;
    border-radius: 16px;
    padding: 0 18px;
    font-size: 15px;
    font-weight: 700;
}
QPushButton#primaryButton {
    background: #2e5b40;
    color: #ffffff;
    border: none;
}
QPushButton#primaryButton:hover {
    background: #274d36;
}
QPushButton#secondaryButton {
    background: transparent;
    color: #2e5b40;
    border: 2px solid #2e5b40;
}
QPushButton#secondaryButton:hover {
    background: rgba(46, 91, 64, 0.08);
}
QMessageBox {
    background: #f7f2e8;
}
QMessageBox QLabel {
    color: #243126;
    font-size: 14px;
}
QMessageBox QPushButton {
    min-width: 96px;
}
"""


def build_root_widget():
    root = QWidget()
    root.setObjectName("root")
    return root


def build_panel():
    panel = QFrame()
    panel.setObjectName("panel")
    return panel


def apply_app_style(app):
    app.setStyleSheet(APP_STYLE)


class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Practice")
        self.resize(900, 640)

        title = QLabel("Sign Language Practise")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("heroTitle")
        title.setFont(QFont("Avenir Next", 24))

        # subtitle = QLabel(
        #     "Short webcam drills with spaced repetition so practice feels focused instead of repetitive."
        # )
        # subtitle.setAlignment(Qt.AlignCenter)
        # subtitle.setWordWrap(True)
        # subtitle.setObjectName("heroSubtitle")

        self.start_button = QPushButton("Start Practice")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.clicked.connect(self.start_session)

        self.analytics_button = QPushButton("View Analytics")
        self.analytics_button.setObjectName("secondaryButton")
        self.analytics_button.setCursor(Qt.PointingHandCursor)
        self.analytics_button.clicked.connect(self.show_analytics)

        analytics = self.compute_analytics()
        summary = QLabel(
            f"Reviews completed: {analytics['total_reviews']}   •   Accuracy: {analytics['accuracy_pct']}%"
        )
        summary.setAlignment(Qt.AlignCenter)
        summary.setObjectName("statCard")

        button_row = QHBoxLayout()
        button_row.setSpacing(14)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.analytics_button)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(42, 44, 42, 44)
        panel_layout.setSpacing(18)
        panel_layout.addStretch()
        panel_layout.addWidget(title)
        # panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(summary)
        panel_layout.addSpacing(4)
        panel_layout.addLayout(button_row)
        panel_layout.addStretch()

        panel = build_panel()
        panel.setLayout(panel_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(48, 40, 48, 40)
        layout.addWidget(panel)

        container = build_root_widget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.webcam_window = None

    def start_session(self):
        self.webcam_window = WebcamWindow()
        self.webcam_window.resize(980, 720)
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
                    day = ev["ts"][:10]
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
        analytics = self.compute_analytics()
        quality_labels = {1: "Again", 3: "Hard", 4: "Good", 5: "Easy"}
        quality_lines = []
        for key in [1, 3, 4, 5]:
            quality_lines.append(
                f"{quality_labels[key]}: {analytics['quality_counts'].get(key, 0)}"
            )

        busiest_day = "No activity yet"
        if analytics["reviews_per_day"]:
            day, count = max(
                analytics["reviews_per_day"].items(),
                key=lambda item: item[1],
            )
            busiest_day = f"{day} ({count} reviews)"

        text = (
            f"Total reviews: {analytics['total_reviews']}\n"
            f"Accuracy: {analytics['accuracy_pct']}%\n"
            f"Busiest day: {busiest_day}\n\n"
            f"{chr(10).join(quality_lines)}"
        )
        QMessageBox.information(self, "Analytics", text)


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practice Session")
        self.resize(1040, 780)

        self.closed = False
        self.numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.progress = {}
        self.path = "progress.json"
        self.due_queue = []

        if not os.path.exists(self.path):
            now = datetime.now().isoformat()
            self.progress = {
                number: {
                    "repetitions": 0,
                    "interval": 0,
                    "ease_factor": 2.5,
                    "due": now,
                    "lapses": 0,
                }
                for number in self.numbers
            }

            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.progress = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.progress = {}

        if not self.progress:
            now = datetime.now().isoformat()
            self.progress = {
                number: {
                    "repetitions": 0,
                    "interval": 0,
                    "ease_factor": 2.5,
                    "due": now,
                    "lapses": 0,
                }
                for number in self.numbers
            }

        self.current_number = self.next_number()
        if self.current_number is None:
            return

        section_label = QLabel("Current Prompt")
        section_label.setObjectName("sectionLabel")
        section_label.setAlignment(Qt.AlignCenter)

        prompt_hint = QLabel("Show the matching sign to move to the next card.")
        prompt_hint.setObjectName("heroSubtitle")
        prompt_hint.setAlignment(Qt.AlignCenter)

        self.question = QLabel(self.current_number)
        self.question.setAlignment(Qt.AlignCenter)
        self.question.setObjectName("numberBadge")
        self.question.setFont(QFont("Avenir Next", 28))

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("statCard")

        self.label = QLabel("Starting camera...")
        self.label.setObjectName("cameraLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(500)

        self.answer_button = QPushButton("See Answer")
        self.answer_button.setObjectName("secondaryButton")
        self.answer_button.setCursor(Qt.PointingHandCursor)
        self.answer_button.clicked.connect(self.see_answer)

        self.refresh_status_label()

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(34, 34, 34, 34)
        panel_layout.setSpacing(16)
        panel_layout.addWidget(section_label)
        panel_layout.addWidget(self.question)
        panel_layout.addWidget(prompt_hint)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.label, stretch=1)
        panel_layout.addWidget(self.answer_button)

        panel = build_panel()
        panel.setLayout(panel_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(36, 28, 36, 28)
        layout.addWidget(panel)

        container = build_root_widget()
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
        due = [
            number
            for number in self.numbers
            if datetime.fromisoformat(self.progress[number]["due"]) <= now
        ]
        random.shuffle(due)
        self.due_queue = due

    def next_number(self):
        if not self.due_queue:
            self.build_due_queue()
            if not self.due_queue:
                self.go_home()
                self.closed = True
                return None
        return self.due_queue.pop()

    def refresh_status_label(self):
        remaining = len(self.due_queue) + (1 if self.current_number else 0)
        self.status_label.setText(
            f"Cards left today: {remaining}   •   Use 'See Answer' when you need a quick reminder."
        )

    def see_answer(self):
        img_path = "pictures/" + self.current_number + ".png"
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            message = QMessageBox(self)
            message.setWindowTitle("Answer")
            message.setText("")
            if not pixmap.isNull():
                message.setIconPixmap(
                    pixmap.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            message.exec_()

            self.sm2_update(1)
            self.current_number = self.next_number()
            if self.current_number is None:
                return
            self.question.setText(self.current_number)
            self.refresh_status_label()

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
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

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
                    if self.current_number is None:
                        return
                    self.question.setText(self.current_number)
                    self.refresh_status_label()

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
        msg.setText("Well done. How hard was it to recall?")

        buttons = {}
        labels = ["AGAIN", "HARD", "GOOD", "EASY"]
        for quality in labels:
            button = msg.addButton(quality, QMessageBox.ActionRole)
            buttons[button] = quality

        msg.exec()

        quality_to_number = {
            "AGAIN": 1,
            "HARD": 3,
            "GOOD": 4,
            "EASY": 5,
        }

        clicked_button = msg.clickedButton()
        if clicked_button in buttons:
            chosen_quality = buttons[clicked_button]
            return quality_to_number[chosen_quality]

        return 1

    def closeEvent(self, event):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().closeEvent(event)

    def go_home(self):
        QMessageBox.information(self, "All done", "All cards for today are finished.")
        self.home = HomeWindow()
        self.home.resize(900, 640)
        self.home.show()
        self.close()


def main():
    app = QApplication(sys.argv)
    apply_app_style(app)
    window = HomeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
