import os
import threading
from datetime import datetime
from pathlib import Path

import cv2

from src.detection import MotionEvent
from src.feed_registry import FeedRegistry, FeedStatus


class MotionDetector:
    MIN_CONTOUR_AREA = 1500
    POST_MOTION_SECONDS = 2
    PROCESS_EVERY_N = 2

    def __init__(self, feed_id, store, registry, output_dir: Path, rtsp_url: str):
        self.feed_id = feed_id
        self.store = store
        self.registry = registry
        self.output_dir = output_dir
        self.rtsp_url = rtsp_url
        self._stop_event = threading.Event()
        self._thread = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detector_type(self):
        return "motion"

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _suppress_stderr(self):
        devnull = os.open(os.devnull, os.O_WRONLY)
        old = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        return old

    def _restore_stderr(self, old):
        os.dup2(old, 2)
        os.close(old)

    def _run(self):
        old = self._suppress_stderr()
        cap = cv2.VideoCapture(self.rtsp_url)
        self._restore_stderr(old)
        if not cap.isOpened():
            self.registry.update_status(
                self.feed_id, FeedStatus.ERROR, error="Cannot open: " + self.rtsp_url
            )
            return
        self.registry.update_status(self.feed_id, FeedStatus.PROCESSING)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        recording = False
        writer = None
        frames_since_motion = 0
        event_start = None
        clip_path = None
        frame_n = 0
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1
            if frame_n % self.PROCESS_EVERY_N != 0:
                if recording and writer:
                    writer.write(frame)
                continue
            small = cv2.resize(frame, (640, 360))
            fg = bg_sub.apply(small)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = any(cv2.contourArea(c) > self.MIN_CONTOUR_AREA for c in contours)
            if motion:
                frames_since_motion = 0
                if not recording:
                    recording = True
                    event_start = datetime.now()
                    clip_path = self.output_dir / (
                        "motion_" + event_start.strftime("%Y%m%d_%H%M%S") + ".avi"
                    )
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(clip_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
                    )
            else:
                if recording:
                    frames_since_motion += 1
            if recording and writer:
                writer.write(frame)
                if not motion and frames_since_motion > int(fps * self.POST_MOTION_SECONDS):
                    writer.release()
                    self.store.add_motion_event(MotionEvent(
                        feed_id=self.feed_id, start_time=event_start,
                        clip_path=clip_path,
                        duration_seconds=frames_since_motion / fps,
                    ))
                    recording = False
                    writer = None
        if writer:
            writer.release()
            if event_start:
                self.store.add_motion_event(MotionEvent(
                    feed_id=self.feed_id, start_time=event_start,
                    clip_path=clip_path,
                    duration_seconds=frames_since_motion / fps,
                ))
        cap.release()
        if self._stop_event.is_set():
            self.registry.update_status(self.feed_id, FeedStatus.STOPPED)
        else:
            self.registry.update_status(self.feed_id, FeedStatus.COMPLETE)
