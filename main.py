import sys
import os
import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

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


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Preview")

        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.index = 0

        self.question = QLabel(self.letters[self.index])
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

        self.next_button = QPushButton("Next")
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self.next_action)

        layout = QVBoxLayout()
        layout.addWidget(self.question)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.next_button)

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
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))

    def analyse_picture(self):
        if self.latest_frame is None:
            return
        
        MODEL_ID = "sign-language-dn6dl/2"

        config = InferenceConfiguration(confidence_threshold=0.5, iou_threshold=0.5)

        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key="4O4prTCPkawUqMfWnLI5",
        )
        client.configure(config)
        client.select_model(MODEL_ID)

        predictions = client.infer(self.latest_frame)
        prediction_items = predictions.get("predictions", [])
        if len(prediction_items) > 0:
            sign_prediction = prediction_items[0].get("class", 'No sign detected')
        else:
            sign_prediction = "No sign detected"
        
        if sign_prediction == self.letters[self.index]:
            self.result_label.setText("Correct!")
            self.next_button.setVisible(True)
        else:
            self.result_label.setText("Wrong, please try again")
            # Set this back to false
            self.next_button.setVisible(True)
            self.show_wrong_popup()
        
        # self.result_label.setText(f"Sign: {sign_prediction}")
        

    def closeEvent(self, event):
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
    
    def next_action(self):
        self.index += 1
        self.index %= 26
        self.question.setText(self.letters[self.index])
        self.next_button.setVisible(False)
        self.result_label.setText("")




def main():
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
