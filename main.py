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
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
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

model_dict = pickle.load(open("./modelextended2.p", "rb"))
model = model_dict["model"]
WINDOW_REGISTRY = []

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
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E"
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
QFrame#metricCard {
    background: rgba(255, 251, 245, 0.96);
    border: 1px solid rgba(52, 77, 61, 0.10);
    border-radius: 22px;
}
QLabel#metricValue {
    color: #173324;
    font-size: 29px;
    font-weight: 700;
}
QLabel#metricLabel {
    color: #617261;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
QLabel#metricDetail {
    color: #516053;
    font-size: 14px;
}
QFrame#chartCard {
    background: rgba(255, 255, 255, 0.93);
    border: 1px solid rgba(50, 73, 59, 0.12);
    border-radius: 24px;
}
QLabel#chartTitle {
    color: #213427;
    font-size: 20px;
    font-weight: 700;
}
QLabel#chartSubtitle {
    color: #5d6b60;
    font-size: 14px;
}
QLabel#insightPill {
    background: #eef5ee;
    color: #2f5a40;
    border-radius: 14px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 600;
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


def keep_window_reference(window):
    WINDOW_REGISTRY.append(window)
    return window


def load_review_events(path="reviews.jsonl"):
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return events


def compute_analytics_summary(path="reviews.jsonl"):
    total = 0
    correct = 0
    quality_counts = Counter()
    reviews_per_day = defaultdict(int)
    card_totals = Counter()

    for ev in load_review_events(path):
        total += 1
        if ev.get("correct"):
            correct += 1
        quality = int(ev.get("quality", 0))
        quality_counts[quality] += 1
        timestamp = ev.get("ts", "")
        if len(timestamp) >= 10:
            reviews_per_day[timestamp[:10]] += 1
        card_id = ev.get("card_id")
        if card_id:
            card_totals[card_id] += 1

    accuracy = (correct / total * 100) if total else 0.0
    ordered_days = dict(sorted(reviews_per_day.items()))
    streak_days = 0
    if ordered_days:
        active_days = {
            datetime.fromisoformat(day).date() for day in ordered_days.keys()
        }
        cursor = max(active_days)
        while cursor in active_days:
            streak_days += 1
            cursor -= timedelta(days=1)

    best_day = None
    if ordered_days:
        best_day = max(ordered_days.items(), key=lambda item: item[1])

    return {
        "total_reviews": total,
        "accuracy_pct": round(accuracy, 1),
        "quality_counts": dict(quality_counts),
        "reviews_per_day": ordered_days,
        "card_totals": dict(card_totals),
        "streak_days": streak_days,
        "best_day": best_day,
    }


class MetricCard(QFrame):
    def __init__(self, value, label, detail, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")

        value_label = QLabel(value)
        value_label.setObjectName("metricValue")

        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")

        detail_label = QLabel(detail)
        detail_label.setObjectName("metricDetail")
        detail_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        layout.addWidget(label_widget)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        layout.addStretch()
        self.setLayout(layout)


class QualityBarChart(QWidget):
    def __init__(self, quality_counts, parent=None):
        super().__init__(parent)
        self.quality_counts = quality_counts
        self.labels = [
            ("Again", 1, QColor("#d96c5f")),
            ("Hard", 3, QColor("#d9a441")),
            ("Good", 4, QColor("#4f8a5b")),
            ("Easy", 5, QColor("#2c5f46")),
        ]
        self.setMinimumHeight(170)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffaf4"))

        chart_rect = self.rect().adjusted(28, 18, -28, -40)
        track_rect = chart_rect.adjusted(0, 8, 0, -24)
        max_value = max([self.quality_counts.get(key, 0) for _, key, _ in self.labels] + [1])
        bar_space = chart_rect.width() / max(len(self.labels), 1)

        painter.setPen(QPen(QColor("#d9d3c8"), 1))
        painter.drawLine(track_rect.bottomLeft(), track_rect.bottomRight())

        for index, (label, key, color) in enumerate(self.labels):
            value = self.quality_counts.get(key, 0)
            bar_width = min(42, bar_space * 0.58)
            x = chart_rect.left() + index * bar_space + (bar_space - bar_width) / 2
            full_bar_rect = QRectF(x, track_rect.top(), bar_width, track_rect.height())
            fill_height = 0 if max_value == 0 else (value / max_value) * track_rect.height()
            fill_rect = QRectF(x, track_rect.bottom() - fill_height, bar_width, fill_height)
            radius = min(10.0, bar_width / 2, max(fill_rect.height() / 2, 0.0))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#ebe4d9"))
            painter.drawRoundedRect(full_bar_rect, 10, 10)

            if fill_rect.height() > 0:
                painter.setBrush(color)
                painter.drawRoundedRect(fill_rect, radius, radius)

            painter.setPen(QColor("#2c392f"))
            painter.setFont(QFont("Avenir Next", 10, QFont.DemiBold))
            painter.drawText(QRectF(x - 12, chart_rect.bottom() + 6, bar_width + 24, 16), Qt.AlignCenter, label)
            painter.drawText(QRectF(x - 12, track_rect.top() - 20, bar_width + 24, 18), Qt.AlignCenter, str(value))

        painter.end()


class ActivityLineChart(QWidget):
    def __init__(self, reviews_per_day, parent=None):
        super().__init__(parent)
        self.reviews_per_day = reviews_per_day
        self.setMinimumHeight(170)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffaf4"))

        chart_rect = self.rect().adjusted(32, 20, -20, -34)
        values = list(self.reviews_per_day.values())[-7:]
        labels = list(self.reviews_per_day.keys())[-7:]

        painter.setPen(QPen(QColor("#ddd4c7"), 1))
        for step in range(4):
            y = chart_rect.top() + step * chart_rect.height() / 3
            painter.drawLine(chart_rect.left(), int(y), chart_rect.right(), int(y))

        if not values:
            painter.setPen(QColor("#6a756b"))
            painter.setFont(QFont("Avenir Next", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "Complete a few review sessions to populate this chart.")
            painter.end()
            return

        max_value = max(values + [1])
        if len(values) == 1:
            points = [QPointF(chart_rect.center().x(), chart_rect.bottom() - (values[0] / max_value) * chart_rect.height())]
        else:
            points = []
            step_x = chart_rect.width() / (len(values) - 1)
            for index, value in enumerate(values):
                x = chart_rect.left() + index * step_x
                y = chart_rect.bottom() - (value / max_value) * (chart_rect.height() - 10)
                points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)

        fill_path = QPainterPath(path)
        fill_path.lineTo(chart_rect.bottomRight())
        fill_path.lineTo(chart_rect.bottomLeft())
        fill_path.closeSubpath()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(122, 163, 127, 55))
        painter.drawPath(fill_path)

        painter.setPen(QPen(QColor("#3d6d4d"), 3))
        painter.drawPath(path)

        painter.setBrush(QColor("#244e35"))
        for point, value in zip(points, values):
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(point, 4, 4)
            painter.setPen(QColor("#2b372e"))
            painter.setFont(QFont("Avenir Next", 10, QFont.DemiBold))
            painter.drawText(QRectF(point.x() - 14, point.y() - 24, 28, 16), Qt.AlignCenter, str(value))

        painter.setPen(QColor("#59665b"))
        painter.setFont(QFont("Avenir Next", 9))
        for point, label in zip(points, labels):
            painter.drawText(QRectF(point.x() - 24, chart_rect.bottom() + 8, 48, 14), Qt.AlignCenter, label[5:])

        painter.end()


class AnalyticsWindow(QMainWindow):
    def __init__(self, home_window=None, analytics_path="reviews.jsonl"):
        super().__init__()
        self.home_window = home_window
        self.analytics_path = analytics_path
        self.setWindowTitle("Practice Analytics")
        self.resize(1080, 760)

        analytics = compute_analytics_summary(self.analytics_path)

        header_label = QLabel("Analytics")
        header_label.setObjectName("heroTitle")

        subtitle = QLabel(
            "A clearer read on your recall quality, session rhythm, and which signs are taking the most reps."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        back_button = QPushButton("Back Home")
        back_button.setObjectName("secondaryButton")
        back_button.setCursor(Qt.PointingHandCursor)
        back_button.clicked.connect(self.go_home)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("primaryButton")
        refresh_button.setCursor(Qt.PointingHandCursor)
        refresh_button.clicked.connect(self.refresh_page)

        header_buttons = QHBoxLayout()
        header_buttons.setSpacing(12)
        header_buttons.addWidget(back_button)
        header_buttons.addWidget(refresh_button)

        top_practiced = sorted(
            analytics["card_totals"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        top_practiced_text = " • ".join(
            f"{card}: {count}" for card, count in top_practiced
        ) or "No review history yet"

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(14)
        metrics_grid.setVerticalSpacing(14)
        metrics_grid.addWidget(
            MetricCard(
                str(analytics["total_reviews"]),
                "Total Reviews",
                "Every scored recall event captured across sessions.",
            ),
            0,
            0,
        )
        metrics_grid.addWidget(
            MetricCard(
                f"{analytics['accuracy_pct']}%",
                "Accuracy",
                "Share of reviews marked correct after recognition or answer reveal.",
            ),
            0,
            1,
        )
        metrics_grid.addWidget(
            MetricCard(
                str(analytics["streak_days"]),
                "Active Streak",
                "Consecutive active review days based on your log history.",
            ),
            0,
            2,
        )
        metrics_grid.addWidget(
            MetricCard(
                top_practiced[0][0] if top_practiced else "--",
                "Most Practiced",
                top_practiced_text,
            ),
            0,
            3,
        )

        quality_chart = self.build_chart_card(
            "Recall Quality",
            "How each review was rated.",
            QualityBarChart(analytics["quality_counts"]),
        )
        activity_chart = self.build_chart_card(
            "Recent Activity",
            "Daily review volume over your latest sessions.",
            ActivityLineChart(analytics["reviews_per_day"]),
        )

        dashboard_row = QHBoxLayout()
        dashboard_row.setSpacing(14)
        dashboard_row.addWidget(quality_chart, 1)
        dashboard_row.addWidget(activity_chart, 1)

        hero_layout = QHBoxLayout()
        hero_layout.setSpacing(16)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(8)
        hero_text.addWidget(header_label)
        hero_text.addWidget(subtitle)
        hero_text.addStretch()
        hero_layout.addLayout(hero_text, 1)
        hero_layout.addLayout(header_buttons)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(24, 22, 24, 22)
        panel_layout.setSpacing(14)
        panel_layout.addLayout(hero_layout)
        panel_layout.addLayout(metrics_grid)
        panel_layout.addLayout(dashboard_row, 1)

        panel = build_panel()
        panel.setLayout(panel_layout)

        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(22, 18, 22, 18)
        page_layout.addWidget(panel)

        container = build_root_widget()
        container.setLayout(page_layout)
        self.setCentralWidget(container)

    def build_chart_card(self, title, subtitle, chart_widget):
        title_label = QLabel(title)
        title_label.setObjectName("chartTitle")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("chartSubtitle")
        subtitle_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(chart_widget, 1)

        card = QFrame()
        card.setObjectName("chartCard")
        card.setLayout(layout)
        return card

    def refresh_page(self):
        refreshed = keep_window_reference(
            AnalyticsWindow(self.home_window, self.analytics_path)
        )
        refreshed.resize(self.size())
        refreshed.show()
        self.close()

    def go_home(self):
        if self.home_window is None:
            self.home_window = keep_window_reference(HomeWindow())
            self.home_window.resize(900, 640)
        self.home_window.show()
        self.close()


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
        self.webcam_window = keep_window_reference(WebcamWindow())
        self.webcam_window.resize(980, 720)
        if not self.webcam_window.closed:
            self.webcam_window.show()
        self.close()

    def compute_analytics(self, path="reviews.jsonl"):
        analytics = compute_analytics_summary(path)
        return {
            "total_reviews": analytics["total_reviews"],
            "accuracy_pct": analytics["accuracy_pct"],
            "quality_counts": analytics["quality_counts"],
            "reviews_per_day": analytics["reviews_per_day"],
        }

    def show_analytics(self):
        self.analytics_window = keep_window_reference(AnalyticsWindow(self))
        self.analytics_window.resize(1080, 760)
        self.analytics_window.show()
        self.hide()


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practice Session")
        self.resize(1040, 780)

        self.closed = False
        self.numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "B", "C", "D", "E"]
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
        self.label.setScaledContents(False)

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
        x_ = []
        y_ = []

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
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            probs = model.predict_proba([np.asarray(data_aux)])[0]
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            sign_prediction = labels_dict[int(model.classes_[best_idx])]

            if confidence >= 0.8 or (sign_prediction == 4 and confidence >= 0.6):
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
        pixmap = QPixmap.fromImage(image).scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label.setPixmap(pixmap)

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
        self.home = keep_window_reference(HomeWindow())
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
