# Emote Detector

A real-time hand and face gesture detector using **MediaPipe** and **OpenCV** that recognizes a variety of facial and hand poses, then displays corresponding meme images.

Detects gestures like yawning, thinking, celebrating, crying, or a shocked pose and updates the meme display in real-time.

---

## Features

* Real-time detection of facial landmarks using MediaPipe Face Mesh.
* Real-time detection of hand landmarks using MediaPipe Hands.
* Detects:

  * **Thinking**: Pointer finger next to the corner of the lips.
  * **Yawning**: Hand covering mouth.
  * **Hog (shocked pose)**: Mouth open with hands near jaw.
  * **Celebrate**: Both hands raised in the air.
  * **Cry**: Both hands in fists near the eyes.

---

## Project Structure

```
emote-detector/
├── src/
│   └── main.py
├── images/
│   ├── celebrate.png
│   ├── cry.png
│   ├── hog.png
│   ├── thinking.png
│   ├── yawning.png
│   └── blank.png
├── README.md
└── requirements.txt
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/gesture-meme-detector.git
cd gesture-meme-detector
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Dependencies:

* `opencv-python`
* `mediapipe`

---

## Usage

1. Make sure all emote images are in the `images/` folder.
2. Run the main script:

```bash
python src/main.py
```

3. Webcam feed will open in a window labeled `tspmo`.
4. Meme output will appear in a separate window labeled `emote`.
5. Press **`q`** to quit.

---

## How It Works

* **Face detection:** Uses MediaPipe Face Mesh to extract 468 facial landmarks.
* **Hand detection:** Uses MediaPipe Hands to extract hand landmarks and check for gestures.
* **Gesture logic:** Checks for various gestures based on hand positions, finger states (fist, extended), and distances to the face or mouth.
* **Emote display:** Shows the corresponding emote for the detected gesture.

## 📄 License

MIT License (or your preferred license)

---
* [MediaPipe](https://google.github.io/mediapipe/) for landmark detection
* [OpenCV](https://opencv.org/) for real-time video processing
