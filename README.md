# Sign Language Translator

A real-time Sign Language Translator using computer vision and deep learning. Uses **MediaPipe Holistic** to extract pose, face, and hand keypoints and a **Transformer-based model** to classify gestures into text and speech.

---

## Features

* **Real-time Translation** – Converts sign gestures into text instantly
* **Text-to-Speech (TTS)** – Speaks generated sentences
* **Transformer Architecture** – Custom sequence classification model
* **Custom Data Pipeline** – Record your own sign dataset and retrain
* **Sentence Construction** – Stabilized predictions and sentence formation logic

---

## Prerequisites

* Python **3.10**
* **Webcam**

---

## Installation

### Clone the repository

```bash
git clone https://github.com/machinesr/Sign-Language-Translator.git
cd Sign-Language-Translator
```

### Create a Virtual Environment (recommended)

```bash
python -m venv sign-language-translator-env

# Windows
sign-language-translator-env\Scripts\activate

# macOS/Linux
source sign-language-translator-env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> Note: Ensure TensorFlow version is compatible with your CPU/GPU configuration.

---

## Usage

### Connect a Webcam

The default camera index is `1`. For built-in laptop cameras, edit (VideoCapture):

```python
cap = cv2.VideoCapture(0)
```

### 1. Run the Translator (Quick Start)

Pre-trained files (`sign_model.h5` and `labels.npy`) are included. Run:

```bash
python app.py
```

#### Controls

| Key | Action                 |
| --- | ---------------------- |
| `s` | Speak sentence via TTS |
| `c` | Clear current sentence |
| `q` | Quit application       |

---

### 2. Collect Custom Data

To add new gestures or retrain:

```bash
python data_collection.py
```

#### Recording controls

| Key     | Action                            |
| ------- | --------------------------------- |
| `Space` | Start countdown and record sample |
| `c`     | Delete last sample                |
| `q`     | Quit                              |

> Collect samples for **`idle`** (no gesture) to help the model detect non-signing periods.

---

### 3. Train the Model

```bash
python training.py
```

This generates a new `sign_model.h5` and `labels.npy` based on collected data.

---

## Configuration

Adjust parameters in **`config.py`**:

| Setting                     | Description                              |
| --------------------------- | ---------------------------------------- |
| `CAMERA_INDEX`              | Select camera (default: 1)               |
| `CONFIDENCE_THRESHOLD`      | Minimum probability to accept prediction |
| `PREDICTION_HISTORY_FRAMES` | Frames used for prediction smoothing     |
| `IDLE_SIGN_NAME`            | Label for idle class                     |

---

## Supported Signs / Movements

| Sign Name | Description / Guide                        | Visual Reference                                                      |
| --------- | ------------------------------------------ | ----------------------------------------------------------------------|
| idle      | Standing still, hands relaxed              | -                  |
| me        | Pointing to your chest                     | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=97s)  |
| hello     | Salute extending outward                   | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=79s)  |
| father    | Tap thumb of open hand on forehead.        | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=106s) |
| mother    | Tap thumb of open hand on chin.            | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=116s) |
| yes       | Make a fist and bob it up and down.        | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=123s) |
| no        | Tap index and middle finger against thumb. | [Video Reference](https://www.youtube.com/watch?v=0FcwzMq4iWg&t=131s) |

> Update table to match `labels.npy`.

---

## License

Open-source for personal and educational use.
