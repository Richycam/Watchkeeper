import cv2
import numpy as np

from src.detection import Detection
from src.base_detector import BaseDetector


class RedCarDetector(BaseDetector):
    RED_LOWER_1 = np.array([0, 60, 60])
    RED_UPPER_1 = np.array([15, 255, 255])
    RED_LOWER_2 = np.array([155, 60, 60])
    RED_UPPER_2 = np.array([180, 255, 255])
    MIN_CONTOUR_AREA = 800
    MIN_ASPECT_RATIO = 0.8
    MAX_ASPECT_RATIO = 6.0
    RED_PIXEL_THRESHOLD = 0.10

    def detector_type(self):
        return "red_car"

    def detect_frame(self, frame, timestamp_seconds):
        return self._detect_colour(frame, timestamp_seconds)

    def _detect_colour(self, frame, timestamp_seconds):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1),
            cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        seen = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue
            if any(abs(x - sx) < 30 and abs(y - sy) < 30 for sx, sy in seen):
                continue
            seen.append((x, y))
            annotated = frame.copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, "Red Vehicle", (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            thumb = self.save_thumbnail(annotated, timestamp_seconds, "_red_car")
            detections.append(Detection(
                feed_id=self.feed_id, detector_type=self.detector_type(),
                timestamp_seconds=timestamp_seconds, label="Red Vehicle",
                confidence=min(area / 20000, 0.95), thumbnail_path=thumb,
                metadata={"bbox": [x, y, x + w, y + h], "area": int(area)},
            ))
        return detections
