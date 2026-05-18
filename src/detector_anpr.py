import cv2
from src.detection import Detection
from src.base_detector import BaseDetector


class ANPRDetector(BaseDetector):
    def __init__(self, feed_id, store, registry, output_dir, target_plate: str):
        super().__init__(feed_id, store, registry, output_dir)
        self._target_plate = self._normalise(target_plate)
        self._reader = self._load_reader()

    def _load_reader(self):
        try:
            import easyocr
            return easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            return None

    def _normalise(self, plate: str):
        return plate.upper().replace(" ", "").replace("-", "")

    def detector_type(self):
        return "anpr"

    def detect_frame(self, frame, timestamp_seconds):
        if self._reader is None:
            return []
        detections = []
        for region, (rx, ry, rw, rh) in self._extract_regions(frame):
            for (_, text, conf) in self._reader.readtext(region):
                norm = self._normalise(text)
                if len(norm) < 4:
                    continue
                is_target = norm == self._target_plate
                partial = self._target_plate in norm or norm in self._target_plate
                if is_target or (partial and conf > 0.4):
                    annotated = frame.copy()
                    cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
                    cv2.putText(annotated, text.upper(), (rx, max(ry - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    thumb = self.save_thumbnail(annotated, timestamp_seconds, "_plate_" + norm)
                    detections.append(Detection(
                        feed_id=self.feed_id, detector_type=self.detector_type(),
                        timestamp_seconds=timestamp_seconds,
                        label="Plate: " + text.upper(),
                        confidence=float(conf), thumbnail_path=thumb,
                        metadata={"plate_text": text.upper(), "normalised": norm,
                                  "target_match": is_target,
                                  "bbox": [rx, ry, rx+rw, ry+rh]},
                    ))
        return detections

    def _extract_regions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        fh, fw = frame.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500 or area > fh * fw * 0.1:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            if 2.0 <= (w / h) <= 6.0 and 20 < h < 120:
                x1, y1 = max(0, x-5), max(0, y-5)
                x2, y2 = min(fw, x+w+5), min(fh, y+h+5)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    regions.append((roi, (x1, y1, x2-x1, y2-y1)))
        return regions[:20]
