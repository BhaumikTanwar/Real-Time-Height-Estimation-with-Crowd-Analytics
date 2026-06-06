import cv2
import sqlite3
import os

from statistics import median
from ultralytics import YOLO

points = []

def create_database():
    if os.path.exists("crowd.db"):
        os.remove("crowd.db")

    conn = sqlite3.connect("crowd.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            height REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_person(height):

    conn = sqlite3.connect("crowd.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO people(height)
        VALUES (?)
        """,
        (height,)
    )

    conn.commit()
    conn.close()

def click_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Clicked: {x}, {y}")

def calibrate():

    global points

    points = []

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Calibration")

    cv2.setMouseCallback(
        "Calibration",
        click_event
    )

    print("Click TOP of A4 sheet")
    print("Then click BOTTOM of A4 sheet")

    while True:

        ret, frame = cap.read()

        cv2.imshow(
            "Calibration",
            frame
        )

        if len(points) == 2:

            y1 = points[0][1]
            y2 = points[1][1]

            a4_pixels = abs(y2 - y1)

            A4_HEIGHT_CM = 29.7

            cm_per_pixel = (
                A4_HEIGHT_CM /
                a4_pixels
            )

            print(
                f"A4 Pixels: {a4_pixels}"
            )

            print(
                f"Scale: {cm_per_pixel}"
            )

            cap.release()

            cv2.destroyAllWindows()

            return cm_per_pixel

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return None

def estimate_height(
    pixel_height,
    scale
):

    return pixel_height * scale

def detect_people(scale):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    person_heights = {}

    logged_ids = set()

    stable_frames = {}

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if len(results) == 0:
            continue

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if cls != 0:
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            pixel_height = y2 - y1

            height_cm = estimate_height(
                pixel_height,
                scale
            )

            if track_id not in person_heights:
                person_heights[track_id] = []

            if track_id not in stable_frames:
                stable_frames[track_id] = 0

            person_heights[track_id].append(
                height_cm
            )

            person_heights[track_id] = (
                person_heights[track_id][-5:]
            )

            recent = person_heights[track_id]

            status = "Collecting"

            if len(recent) == 5:

                variation = (
                    max(recent)
                    - min(recent)
                )

                if variation < 5:

                    stable_frames[track_id] += 1

                    status = (
                        f"Stable {stable_frames[track_id]}/30"
                    )

                    if (
                        stable_frames[track_id] >= 30
                        and
                        track_id not in logged_ids
                    ):

                        final_height = median(
                            recent
                        )

                        save_person(
                            final_height
                        )

                        logged_ids.add(
                            track_id
                        )

                        status = "Saved"

                        print(
                            f"Saved Height: {final_height:.1f} cm"
                        )

                else:

                    stable_frames[track_id] = 0

                    status = "Collecting"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{height_cm:.1f} cm",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {stable_frames.get(track_id, 0)}",
                (x1, y2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv2.imshow(
            "Height Estimation",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

def main():

    create_database()

    scale = calibrate()

    if scale is None:

        print(
            "Calibration failed"
        )

        return

    print(
        f"Using Scale: {scale}"
    )

    detect_people(scale)


if __name__ == "__main__":

    main()