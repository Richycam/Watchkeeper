import cv2
import numpy as np

from src.detection import Detection
from src.face_store import FaceStore
from src.base_detector import BaseDetector


class FaceRecognitionDetector(BaseDetector):
    SIMILARITY_THRESHOLD = 0.55
    SKIP_FRAMES = 3

    def __init__(self, feed_id, store, registry, output_dir, face_store: FaceStore):
        super().__init__(feed_id, store, registry, output_dir)
        self._face_store = face_store
        self._deepface = None
        self._haar = None
        self._frame_count = 0
        self._init_backend()

    def _init_backend(self):
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except Exception:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detector_type(self):
        return "face_recognition"

    def get_embedding(self, image_input):
        if self._deepface is None:
            return None
        try:
            result = self._deepface.represent(
                img_path=image_input, model_name="Facenet512",
                enforce_detection=False, detector_backend="opencv",
            )
            return np.array(result[0]["embedding"])
        except Exception:
            return None

    def detect_frame(self, frame, timestamp_seconds):
        self._frame_count += 1
        if self._frame_count % (self.SKIP_FRAMES + 1) != 0:
            return []
        references = self._face_store.all_references()
        if self._deepface:
            return self._detect_deepface(frame, timestamp_seconds, references)
        return self._detect_haar(frame, timestamp_seconds)

    def _detect_deepface(self, frame, timestamp_seconds, references):
        detections = []
        try:
            faces = self._deepface.extract_faces(
                img_path=frame, enforce_detection=False, detector_backend="opencv"
            )
            for face_data in faces:
                region = face_data.get("facial_area", {})
                x, y, w, h = (region.get("x", 0), region.get("y", 0),
                               region.get("w", 0), region.get("h", 0))
                if w < 20 or h < 20:
                    continue
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                embedding = self.get_embedding(roi)
                if embedding is None:
                    continue
                best_name, best_score = self._best_match(embedding, references)
                if best_score >= self.SIMILARITY_THRESHOLD:
                    label = "Match: " + best_name
                    colour = (0, 255, 0)
                else:
                    label = "Unknown Face"
                    colour = (0, 165, 255)
                annotated = frame.copy()
                cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 2)
                cv2.putText(annotated, label, (x, max(y - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                thumb = self.save_thumbnail(annotated, timestamp_seconds, "_face")
                detections.append(Detection(
                    feed_id=self.feed_id, detector_type=self.detector_type(),
                    timestamp_seconds=timestamp_seconds, label=label,
                    confidence=float(best_score), thumbnail_path=thumb,
                    metadata={"bbox": [x, y, x+w, y+h], "matched_reference": best_name},
                ))
        except Exception:
            pass
        return detections

    def _detect_haar(self, frame, timestamp_seconds):
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            annotated = frame.copy()
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 165, 255), 2)
            thumb = self.save_thumbnail(annotated, timestamp_seconds, "_face")
            detections.append(Detection(
                feed_id=self.feed_id, detector_type=self.detector_type(),
                timestamp_seconds=timestamp_seconds, label="Face Detected",
                confidence=0.5, thumbnail_path=thumb,
                metadata={"bbox": [x, y, x+w, y+h]},
            ))
        return detections

    def _best_match(self, embedding, references):
        best_name, best_score = "Unknown", 0.0
        for name, ref_emb in references.items():
            score = self._cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score

    def _cosine_similarity(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
