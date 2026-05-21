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
        self._use_face_recognition = False
        self._haar = None
        self._frame_count = 0
        self._init_backend()

    def _init_backend(self):
        try:
            import face_recognition
            self._use_face_recognition = True
        except Exception:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detector_type(self):
        return "face_recognition"

    def get_embedding(self, image_input):
        if not self._use_face_recognition:
            return None
        try:
            import face_recognition
            # Load image if it's a file path (string), otherwise assume it's already loaded
            if isinstance(image_input, str):
                image = face_recognition.load_image_file(image_input)
            else:
                # Convert BGR to RGB if it's a numpy array from cv2
                image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                return np.array(encodings[0])
            return None
        except Exception:
            return None

    def detect_frame(self, frame, timestamp_seconds):
        self._frame_count += 1
        if self._frame_count % (self.SKIP_FRAMES + 1) != 0:
            return []
        references = self._face_store.all_references()
        if self._use_face_recognition:
            return self._detect_face_recognition(frame, timestamp_seconds, references)
        return self._detect_haar(frame, timestamp_seconds)

    def _detect_face_recognition(self, frame, timestamp_seconds, references):
        detections = []
        try:
            import face_recognition
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # face_locations returns (top, right, bottom, left)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Convert to (x, y, w, h) format
                x, y = left, top
                w, h = right - left, bottom - top
                
                if w < 20 or h < 20:
                    continue
                
                embedding = np.array(encoding)
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
