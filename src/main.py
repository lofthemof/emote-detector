import cv2
import mediapipe as mp
import math

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

CELEBRATE = "celebrate"
CRY = "cry"
HOG = "hog"
THINKING = "thinking"
YAWNING = "yawning"
BLANK = "blank"


def detect_emotions(frame):
    pass


def load_images():
    images = {}
    image_files = {
        "celebrate": "celebrate.png",
        "cry": "cry.png",
        "hog": "hog.png",
        "thinking": "thinking.png",
        "yawning": "yawning.png",
        "blank": "white.png",
    }
    for name, filename in image_files.items():
        img = cv2.imread(filename)
        if img is None:
            print(f"Could not load image '{filename}'")
            return
        images[name] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    return images


def main():
    images = load_images()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        cv2.imshow("tspmo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
