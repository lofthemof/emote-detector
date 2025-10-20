import cv2
import mediapipe as mp
import math

WINDOW_WIDTH = 756
WINDOW_HEIGHT = 491

CELEBRATE = "celebrate"
CRY = "cry"
HOG = "hog"
THINKING = "thinking"
YAWNING = "yawning"
BLANK = "blank"


def detect_emote(frame):
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape

    detected_emotion = BLANK

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        face_results = face_mesh.process(rgb_frame)

        face_center_x, face_center_y = None, None
        if face_results.multi_face_landmarks:
            nose_tip = face_results.multi_face_landmarks[0].landmark[1]
            face_center_x = int(nose_tip.x * w)
            face_center_y = int(nose_tip.y * h)

    # hand detection
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        hand_results = hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks and face_center_x is not None:
            hands_data = []

            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                hand_x = int(wrist.x * w)
                hand_y = int(wrist.y * h)

                is_fist = check_fist(hand_landmarks, w, h)

                hands_data.append(
                    {
                        "x": hand_x,
                        "y": hand_y,
                        "is_fist": is_fist,
                        "landmarks": hand_landmarks,
                    }
                )

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            # THINKING: check if pointer finger is touching corner of lip (highest priority)
            if face_results.multi_face_landmarks:
                if check_thinking_gesture(
                    hands_data, face_results.multi_face_landmarks[0], w, h
                ):
                    detected_emotion = THINKING

            # YAWN: check if exactly one hand is covering the face (second priority)
            if detected_emotion == BLANK:
                hands_covering_face = 0
                for hand in hands_data:
                    fingers_near_mouth = check_fingers_near_mouth(
                        hand["landmarks"], face_center_x, face_center_y, w, h
                    )
                    only_index_finger = check_only_index_finger_extended(
                        hand["landmarks"], w, h
                    )
                    if fingers_near_mouth and not only_index_finger:
                        hands_covering_face += 1
                if hands_covering_face == 1:
                    detected_emotion = YAWNING

            # HOG: Check if mouth is open and hands are open near jaw (third priority)
            if detected_emotion == BLANK and face_results.multi_face_landmarks:
                mouth_open = check_mouth_open(face_results.multi_face_landmarks[0], h)
                hands_open_near_jaw = check_hands_open_near_jaw(
                    hands_data, face_center_x, face_center_y
                )

                if mouth_open and hands_open_near_jaw:
                    detected_emotion = HOG

            # Only check other emotions if yawn, hog, and thinking are not detected
            if detected_emotion == BLANK and len(hands_data) == 2:
                # CELEBRATE: Both hands up with open hands (more lenient)
                if (
                    not hands_data[0]["is_fist"]
                    and not hands_data[1]["is_fist"]
                    and hands_data[0]["y"] < face_center_y - 100
                    and hands_data[1]["y"] < face_center_y - 100
                ):
                    detected_emotion = CELEBRATE

                # CRY: Both hands in fists (more lenient position)
                elif (
                    hands_data[0]["is_fist"]
                    and hands_data[1]["is_fist"]
                    and hands_data[0]["y"] < face_center_y + 150
                    and hands_data[1]["y"] < face_center_y + 150
                ):
                    detected_emotion = CRY

        cv2.putText(
            frame,
            f"Emotion: {detected_emotion.upper()}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if detected_emotion != BLANK else (255, 255, 255),
            2,
        )

    return frame, detected_emotion


def check_fist(hand_landmarks, w, h):
    """Check if hand is in a fist position by checking if fingertips are between knuckles and wrist"""
    fingertips = [4, 8, 12, 16, 20]  # fingertip landmarks
    knuckles = [3, 6, 10, 14, 18]  # knuckle landmarks
    wrist = 0  # wrist landmark

    fingers_curled = 0
    for tip, knuckle in zip(fingertips, knuckles):
        tip_x = hand_landmarks.landmark[tip].x * w
        tip_y = hand_landmarks.landmark[tip].y * h
        knuckle_x = hand_landmarks.landmark[knuckle].x * w
        knuckle_y = hand_landmarks.landmark[knuckle].y * h
        wrist_x = hand_landmarks.landmark[wrist].x * w
        wrist_y = hand_landmarks.landmark[wrist].y * h

        wrist_to_knuckle = math.sqrt(
            (knuckle_x - wrist_x) ** 2 + (knuckle_y - wrist_y) ** 2
        )
        wrist_to_tip = math.sqrt((tip_x - wrist_x) ** 2 + (tip_y - wrist_y) ** 2)

        # finger is curled if fingertip is closer to wrist than the knuckle
        if wrist_to_tip < wrist_to_knuckle:
            fingers_curled += 1

    # consider a fist if 4+ fingers are curled
    return fingers_curled >= 4


def check_fingers_near_mouth(hand_landmarks, face_center_x, face_center_y, w, h):
    """Check if any fingertips, knuckles, or palms are near the mouth/face center (for yawning)"""
    fingertips = [4, 8, 12, 16, 20]  # fingertip landmarks
    knuckles = [3, 6, 10, 14, 18]  # knuckle landmarks
    palm_points = [0, 5, 9, 13, 17]  # palm/wrist landmarks

    all_points = fingertips + knuckles + palm_points

    for point_idx in all_points:
        point = hand_landmarks.landmark[point_idx]
        point_x = int(point.x * w)
        point_y = int(point.y * h)

        distance_x = abs(point_x - face_center_x)
        distance_y = abs(point_y - face_center_y)

        if distance_x < 100 and distance_y < 60:
            return True

    return False


def check_only_index_finger_extended(hand_landmarks, w, h):
    """Check if only the index finger is extended by checking if fingertips are between knuckles and wrist"""
    fingertips = [8, 12, 16, 20]  # non-thumb fingertip landmarks
    knuckles = [6, 10, 14, 18]  # non-thumb knuckle landmarks
    wrist = 0  # wrist landmark

    extended_fingers = 0
    index_finger_extended = False

    for i, (tip, knuckle) in enumerate(zip(fingertips, knuckles)):
        tip_x = hand_landmarks.landmark[tip].x * w
        tip_y = hand_landmarks.landmark[tip].y * h
        knuckle_x = hand_landmarks.landmark[knuckle].x * w
        knuckle_y = hand_landmarks.landmark[knuckle].y * h
        wrist_x = hand_landmarks.landmark[wrist].x * w
        wrist_y = hand_landmarks.landmark[wrist].y * h

        wrist_to_knuckle = math.sqrt(
            (knuckle_x - wrist_x) ** 2 + (knuckle_y - wrist_y) ** 2
        )
        wrist_to_tip = math.sqrt((tip_x - wrist_x) ** 2 + (tip_y - wrist_y) ** 2)

        if wrist_to_tip > wrist_to_knuckle + 15:
            extended_fingers += 1
            if i == 0:  # index finger
                index_finger_extended = True

    return extended_fingers == 1 and index_finger_extended


def check_mouth_open(face_landmarks, h):
    """Check if mouth is open by measuring distance between upper and lower lip"""
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    lip_distance = abs(upper_lip.y - lower_lip.y) * h

    return lip_distance > 8


def check_hands_open_near_jaw(hands_data, face_center_x, face_center_y):
    """Check if both hands are open and near the jaw area"""
    if len(hands_data) < 2:
        return False

    hands_near_jaw = 0
    jaw_area_y = face_center_y + 250

    for hand in hands_data:
        if not hand["is_fist"]:
            distance_to_jaw = abs(hand["y"] - jaw_area_y)
            distance_to_face_x = abs(hand["x"] - face_center_x)
            if distance_to_face_x < 250 and distance_to_jaw < 150:
                hands_near_jaw += 1

    return hands_near_jaw >= 2


def check_thinking_gesture(hands_data, face_landmarks, w, h):
    """Check if pointer finger is touching the corner of the lip and only index finger is extended"""
    if not hands_data:
        return False

    left_lip_corner = face_landmarks.landmark[61]
    right_lip_corner = face_landmarks.landmark[291]

    left_lip_x = int(left_lip_corner.x * w)
    left_lip_y = int(left_lip_corner.y * h)
    right_lip_x = int(right_lip_corner.x * w)
    right_lip_y = int(right_lip_corner.y * h)

    for hand in hands_data:
        if not check_only_index_finger_extended(hand["landmarks"], w, h):
            continue

        index_finger = hand["landmarks"].landmark[8]
        index_x = int(index_finger.x * w)
        index_y = int(index_finger.y * h)

        left_distance = math.sqrt(
            (index_x - left_lip_x) ** 2 + (index_y - left_lip_y) ** 2
        )
        if left_distance < 60:
            return True

        right_distance = math.sqrt(
            (index_x - right_lip_x) ** 2 + (index_y - right_lip_y) ** 2
        )
        if right_distance < 60:
            return True

    return False


def load_images():
    images = {}
    image_files = {
        "celebrate": "images/celebrate.png",
        "cry": "images/cry.png",
        "hog": "images/hog.png",
        "thinking": "images/thinking.png",
        "yawning": "images/yawning.png",
        "blank": "images/blank.png",
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
        frame, emotion = detect_emote(frame)
        emotion_image = images[emotion]

        cv2.imshow("tspmo", frame)
        cv2.imshow("emote", emotion_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
