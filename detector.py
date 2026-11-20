from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, cast

import cv2
import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray
from mediapipe.python.solutions import drawing_utils as mp_draw  # type: ignore[reportMissingTypeStubs]
from mediapipe.python.solutions import hands as mp_hands  # type: ignore[reportMissingTypeStubs]
import winsound


class Landmark(Protocol):
    x: float
    y: float
    z: float


class HandLandmarks(Protocol):
    landmark: Sequence[Landmark]


class HandsResults(Protocol):
    multi_hand_landmarks: Sequence[HandLandmarks] | None


class Classifier(Protocol):
    classes_: NDArray[np.generic]

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.generic]: ...

    def predict_proba(
        self, features: NDArray[np.float32]
    ) -> NDArray[np.float64]: ...


class LabelDecoder(Protocol):
    def inverse_transform(
        self, labels: NDArray[np.generic]
    ) -> NDArray[np.str_]: ...


FEATURE_COUNT = 63
MODEL_FILE = Path(__file__).resolve().parent / "models" / "help_gesture_model.pkl"
CONFIDENCE_THRESHOLD = 0.85
REQUIRED_FRAMES = 12
RESET_FRAMES = 10


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

    return (points / scale).flatten().reshape(1, FEATURE_COUNT)


def load_model_bundle() -> tuple[Classifier, LabelDecoder]:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. "
            "Train the gesture model and save the bundle there first."
        )

    bundle = joblib.load(MODEL_FILE)  # type: ignore[reportUnknownMemberType]
    if not isinstance(bundle, dict):
        raise TypeError("The model file must contain a dictionary bundle.")
    if "model" not in bundle or "encoder" not in bundle:
        raise KeyError("The model bundle must contain 'model' and 'encoder'.")

    model = cast(Classifier, bundle["model"])
    encoder = cast(LabelDecoder, bundle["encoder"])

    if not all(
        callable(getattr(model, method, None))
        for method in ("predict", "predict_proba")
    ) or not hasattr(model, "classes_"):
        raise TypeError("The bundled model must support predict and predict_proba.")
    if not callable(getattr(encoder, "inverse_transform", None)):
        raise TypeError("The bundled encoder must support inverse_transform.")

    return model, encoder


def main() -> None:
    model, encoder = load_model_bundle()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam 0. Check that a camera is connected and available."
        )

    help_counter = 0
    reset_counter = 0
    alert_triggered = False
    connections = list(
        cast(Iterable[tuple[int, int]], mp_hands.HAND_CONNECTIONS)  # type: ignore[reportUnknownMemberType]
    )

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_results = hands.process(rgb_frame)  # type: ignore[reportUnknownMemberType]
                results = cast(HandsResults, raw_results)

                status = "NO HAND"
                color = (0, 255, 255)
                features: NDArray[np.float32] | None = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(  # type: ignore[reportUnknownMemberType]
                        frame, hand, connections
                    )
                    features = extract_features(hand)

                if features is not None:
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    class_indices = np.flatnonzero(model.classes_ == prediction)
                    if class_indices.size != 1:
                        raise ValueError(
                            "The model prediction does not match exactly one model class."
                        )

                    confidence = float(probabilities[int(class_indices[0])])
                    predicted_label = str(
                        encoder.inverse_transform(np.asarray([prediction]))[0]
                    )

                    if (
                        predicted_label == "HELP"
                        and confidence >= CONFIDENCE_THRESHOLD
                    ):
                        help_counter += 1
                        reset_counter = 0
                        status = "HELP DETECTED"
                        color = (0, 0, 255)

                        if help_counter >= REQUIRED_FRAMES and not alert_triggered:
                            print("EMERGENCY ALERT!")
                            winsound.MessageBeep(winsound.MB_ICONHAND)
                            alert_triggered = True
                    else:
                        help_counter = 0
                        reset_counter += 1
                        status = predicted_label
                        color = (0, 255, 0)

                else:
                    help_counter = 0
                    reset_counter += 1

                if reset_counter >= RESET_FRAMES:
                    alert_triggered = False

                cv2.putText(
                    frame,
                    f"STATUS: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

                if alert_triggered:
                    cv2.rectangle(frame, (0, 70), (640, 130), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "EMERGENCY! HELP ALERT!",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow("AI Help Gesture Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
