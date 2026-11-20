from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from mediapipe.python.solutions import drawing_utils as mp_draw  # type: ignore[reportMissingTypeStubs]
from mediapipe.python.solutions import hands as mp_hands  # type: ignore[reportMissingTypeStubs]


class Landmark(Protocol):
    x: float
    y: float
    z: float


class HandLandmarks(Protocol):
    landmark: Sequence[Landmark]


class HandsResults(Protocol):
    multi_hand_landmarks: Sequence[HandLandmarks] | None


FEATURE_COUNT = 63
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CSV_FILE = DATASET_DIR / "help_gesture.csv"
COLUMNS = [f"f{i}" for i in range(FEATURE_COUNT)] + ["label"]


def extract_features(hand: HandLandmarks) -> NDArray[np.float32] | None:
    if len(hand.landmark) != 21:
        return None

    points = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark],
        dtype=np.float32,
    )

    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale < 1e-6:
        return None

    return (points / scale).flatten()


def initialize_dataset() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)


def main() -> None:
    initialize_dataset()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam 0. Check that a camera is connected and available."
        )

    connections = list(
        cast(Iterable[tuple[int, int]], mp_hands.HAND_CONNECTIONS)  # type: ignore[reportUnknownMemberType]
    )

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            print("Press H = HELP")
            print("Press N = NORMAL")
            print("Press Q = Quit")

            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_results = hands.process(rgb_frame)  # type: ignore[reportUnknownMemberType]
                results = cast(HandsResults, raw_results)
                features = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(  # type: ignore[reportUnknownMemberType]
                        frame, hand, connections
                    )
                    features = extract_features(hand)

                    cv2.putText(
                        frame,
                        "HAND DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    "H: HELP | N: NORMAL | Q: QUIT",
                    (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Dataset Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break

                if key in (ord("h"), ord("H"), ord("n"), ord("N")):
                    if features is None:
                        print("No hand detected; sample not saved.")
                        continue

                    label = "HELP" if key in (ord("h"), ord("H")) else "NORMAL"
                    row = list(features) + [label]
                    pd.DataFrame([row], columns=COLUMNS).to_csv(
                        CSV_FILE,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    print(f"Saved: {label}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
