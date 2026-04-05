# Sign Language Practice

Desktop practice app for reviewing a small set of sign language cards with webcam-based recognition, spaced repetition scheduling, and lightweight analytics.

The current app is a single-file PyQt5 application in [main.py](/Users/allenjollymathew/project/main.py). It recognizes the signs for `1` to `10` and `A` to `E`, schedules future reviews with an SM-2 style algorithm, and logs each review to a JSONL file so progress can be tracked over time.

## What It Does

- Shows one due card at a time.
- Uses the webcam plus MediaPipe hand landmarks to classify the sign you perform.
- Lets you rate recall difficulty as `Again`, `Hard`, `Good`, or `Easy`.
- Updates each card's interval, ease factor, repetitions, due date, and lapse count.
- Shows a home screen summary and a separate analytics view for review history.
- Lets you reveal a reference image from `pictures/` when you need a reminder.

## Project Files

- [main.py](/Users/allenjollymathew/project/main.py): application entry point and all UI / scheduling logic.
- [progress.json](/Users/allenjollymathew/project/progress.json): current spaced repetition state for each card.
- [reviews.jsonl](/Users/allenjollymathew/project/reviews.jsonl): append-only review event log.
- [pictures/](/Users/allenjollymathew/project/pictures): answer images used by the "See Answer" flow.
- `modelextended2.p`: pickled recognition model loaded at startup.

## Requirements

At runtime, the app depends on:

- Python 3
- PyQt5
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- A working webcam


## Setup
Create your own virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Running The App

Start from the project root:

```bash
python main.py
```

Session flow:

1. Open the home screen.
2. Click `Start Practice`.
3. Perform the sign shown on screen.
4. If the model matches the prompt, rate the recall difficulty.
5. If you need help, click `See Answer` to open the reference image. This counts as a failed recall and schedules the card sooner.
6. When no cards are due, the app returns to the home screen and shows a completion message.

## Data Files

`progress.json` stores one object per card, for example:

```json
{
  "A": {
    "repetitions": 1,
    "interval": 1,
    "ease_factor": 2.5,
    "due": "2026-03-19T16:04:45.291504",
    "lapses": 0
  }
}
```

`reviews.jsonl` stores one review event per line, including:

- timestamp
- card id
- quality score
- whether the review counted as correct
- the card state before the update
- the card state after the update

## Analytics

The analytics screen is built from `reviews.jsonl` and currently reports:

- total reviews
- accuracy percentage
- quality distribution
- reviews per day
- active streak
- most-practiced cards


## Notes

- The app loads the model from `./modelextended2.p` at import time, so that file must be present.
- Review logs and progress files are written in the project root.
- If the webcam cannot be opened, the practice window stays open but displays `Could not open webcam.`
