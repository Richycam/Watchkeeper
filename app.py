from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input,
    Label, Rule, Select, Static, TabbedContent, TabPane,
)

from src.detection import Detection
from src.detection_store import DetectionStore
from src.face_store import FaceStore
from src.feed_registry import FeedRegistry, FeedStatus, FeedType
from src.orchestrator import FeedOrchestrator


class NotificationBanner(Static):
    def show_alert(self, message: str, severity: str = "warning"):
        self.update("[bold]ALERT:[/bold] " + message)
        self.add_class("severity-" + severity)
        self.set_timer(6, self.hide_alert)

    def hide_alert(self):
        self.update("")
        self.remove_class("severity-warning", "severity-error", "severity-success")


class UploadFaceModal(ModalScreen):
    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Label("Upload Reference Face", id="modal-title")
            yield Rule()
            yield Label("Name / Label for this person")
            yield Input(placeholder="e.g. Obj_B or John_Doe", id="face-name")
            yield Label("Full path to image file (.jpg / .png)")
            yield Input(placeholder="/path/to/face.jpg", id="face-path")
            with Horizontal(id="modal-buttons"):
                yield Button("Upload", id="confirm-face", variant="primary")
                yield Button("Cancel", id="cancel-face", variant="default")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-face":
            self.dismiss(None)
            return
        name = self.query_one("#face-name", Input).value.strip()
        path = self.query_one("#face-path", Input).value.strip()
        if not name or not path:
            return
        self.dismiss({"name": name, "path": path})


class AddFeedModal(ModalScreen):
    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Label("Add New Feed", id="modal-title")
            yield Rule()
            yield Label("Feed ID (unique name, no spaces)")
            yield Input(placeholder="e.g. feed_cam1", id="feed-id")
            yield Label("Source  (file path or rtsp://...)")
            yield Input(placeholder="/path/to/video.mp4  or  rtsp://192.168.1.x/...",
                        id="feed-source")
            yield Label("Feed Type")
            yield Select(
                [("Recorded Video File", "recorded"), ("Live RTSP Stream", "live")],
                id="feed-type", value="recorded",
            )
            yield Label("Detector Type")
            yield Select(
                [
                    ("Red Car Detection (HSV)", "red_car"),
                    ("Face Recognition (DeepFace)", "face_recognition"),
                    ("ANPR / Licence Plate (EasyOCR)", "anpr"),
                    ("Motion Detection (Live only)", "motion"),
                ],
                id="detector-type", value="red_car",
            )
            yield Label("Label (friendly description)")
            yield Input(placeholder="e.g. Car Park Camera A", id="feed-label")
            yield Label("Target Plate (ANPR only — leave blank for other detectors)")
            yield Input(placeholder="e.g. BP63 LYH", id="feed-param")
            with Horizontal(id="modal-buttons"):
                yield Button("Add Feed", id="confirm-add", variant="primary")
                yield Button("Cancel", id="cancel-add", variant="default")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-add":
            self.dismiss(None)
            return
        feed_id = self.query_one("#feed-id", Input).value.strip()
        source = self.query_one("#feed-source", Input).value.strip()
        if not feed_id or not source:
            return
        self.dismiss({
            "feed_id": feed_id,
            "source": source,
            "feed_type": self.query_one("#feed-type", Select).value,
            "detector_type": self.query_one("#detector-type", Select).value,
            "label": self.query_one("#feed-label", Input).value.strip() or feed_id,
            "param": self.query_one("#feed-param", Input).value.strip(),
        })


class QueryModal(ModalScreen):
    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Label("Query Detections", id="modal-title")
            yield Rule()
            yield Label("Filter by Feed ID  (blank = all feeds)")
            yield Input(placeholder="feed_cam1", id="q-feed-id")
            yield Label("Filter by Detector Type  (blank = all)")
            yield Input(placeholder="red_car / face_recognition / anpr / motion",
                        id="q-detector")
            yield Label("Filter by Label keyword  (blank = all)")
            yield Input(placeholder="Red Vehicle / Match / Plate", id="q-label")
            with Horizontal(id="modal-buttons"):
                yield Button("Search", id="confirm-query", variant="primary")
                yield Button("Cancel", id="cancel-query", variant="default")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-query":
            self.dismiss(None)
            return
        self.dismiss({
            "feed_id": self.query_one("#q-feed-id", Input).value.strip(),
            "detector_type": self.query_one("#q-detector", Input).value.strip(),
            "label": self.query_one("#q-label", Input).value.strip(),
        })


class WatchkeeperApp(App):
    CSS_PATH = str(Path(__file__).resolve().parent / "style.tcss")
    TITLE = "WATCHKEEPER - Surveillance Augmentation System"
    SUB_TITLE = "https://github.com/Richycam"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "add_feed", "Add Feed"),
        Binding("s", "start_selected", "Start"),
        Binding("x", "stop_selected", "Stop"),
        Binding("u", "upload_face", "Upload Face"),
        Binding("f", "query_detections", "Query"),
        Binding("r", "refresh_ui", "Refresh"),
    ]

    def __init__(self, store: DetectionStore, registry: FeedRegistry,
                 orchestrator: FeedOrchestrator, face_store: FaceStore):
        super().__init__()
        self.store = store
        self.registry = registry
        self.orchestrator = orchestrator
        self.face_store = face_store
        self._selected_feed_id: str = ""
        self.store.subscribe(self._on_new_detection)

    def compose(self) -> ComposeResult:
        yield Header()
        yield NotificationBanner(id="notification-banner")
        with TabbedContent():
            with TabPane("Feeds", id="tab-feeds"):
                with Vertical():
                    with Horizontal(id="feeds-actions"):
                        yield Button("Add Feed [A]", id="btn-add-feed", variant="primary")
                        yield Button("Start Selected [S]", id="btn-start-sel", variant="success")
                        yield Button("Stop Selected [X]", id="btn-stop-sel", variant="error")
                        yield Button("Start All", id="btn-start-all", variant="warning")
                        yield Button("Refresh [R]", id="btn-refresh-feeds", variant="default")
                    yield DataTable(id="feeds-table", cursor_type="row")
            with TabPane("Detections", id="tab-detections"):
                with Vertical():
                    with Horizontal(id="detections-actions"):
                        yield Button("Query / Filter [F]", id="btn-query", variant="primary")
                        yield Button("Refresh", id="btn-refresh-detections", variant="default")
                    yield DataTable(id="detections-table")
            with TabPane("Faces", id="tab-faces"):
                with Vertical():
                    with Horizontal(id="faces-actions"):
                        yield Button("Upload Reference Face [U]",
                                     id="btn-upload-face", variant="primary")
                        yield Button("Refresh", id="btn-refresh-faces", variant="default")
                    yield DataTable(id="faces-table")
            with TabPane("Motion Events", id="tab-motion"):
                with Vertical():
                    yield Button("Refresh", id="btn-refresh-motion", variant="default")
                    yield DataTable(id="motion-table")
            with TabPane("Metrics", id="tab-metrics"):
                with Vertical():
                    yield Button("Refresh Metrics", id="btn-refresh-metrics", variant="default")
                    yield Static(id="metrics-display")
        yield Footer()

    def on_mount(self):
        self._setup_tables()
        self._populate_feeds_table()
        self._populate_faces_table()
        self.set_interval(3, self._auto_refresh)

    def _setup_tables(self):
        t = self.query_one("#feeds-table", DataTable)
        t.add_columns("Feed ID", "Type", "Detector", "Label", "Status", "Progress")
        t = self.query_one("#detections-table", DataTable)
        t.add_columns("Feed ID", "Detector", "Label", "Confidence", "Timestamp", "Thumbnail")
        t = self.query_one("#faces-table", DataTable)
        t.add_columns("Name", "Status")
        t = self.query_one("#motion-table", DataTable)
        t.add_columns("Feed ID", "Start Time", "Duration (s)", "Clip Path")

    def _populate_feeds_table(self):
        table = self.query_one("#feeds-table", DataTable)
        table.clear()
        colour_map = {
            FeedStatus.IDLE: "dim",
            FeedStatus.PROCESSING: "yellow",
            FeedStatus.COMPLETE: "green",
            FeedStatus.ERROR: "red",
            FeedStatus.STOPPED: "cyan",
        }
        for feed in self.registry.all_feeds():
            running = self.orchestrator.is_feed_running(feed.feed_id)
            status = feed.status
            if running and status != FeedStatus.PROCESSING:
                status = FeedStatus.PROCESSING
            col = colour_map.get(status, "white")
            table.add_row(
                feed.feed_id,
                feed.feed_type.value,
                feed.detector_type,
                feed.label,
                "[" + col + "]" + status.value + "[/" + col + "]",
                str(round(feed.progress)) + "%",
                key=feed.feed_id,
            )

    def _populate_detections_table(self, detections=None):
        table = self.query_one("#detections-table", DataTable)
        table.clear()
        items = detections if detections is not None else self.store.all_detections()
        for d in reversed(items):
            thumb = Path(str(d.thumbnail_path)).name if d.thumbnail_path else "-"
            table.add_row(
                d.feed_id, d.detector_type, d.label,
                str(round(d.confidence, 2)),
                str(round(d.timestamp_seconds, 1)) + "s",
                thumb,
            )

    def _populate_faces_table(self):
        table = self.query_one("#faces-table", DataTable)
        table.clear()
        names = self.face_store.names()
        if not names:
            table.add_row("No reference faces loaded", "-")
        else:
            for name in names:
                table.add_row(name, "[green]Loaded[/green]")

    def _populate_motion_table(self):
        table = self.query_one("#motion-table", DataTable)
        table.clear()
        for e in self.store.all_motion_events():
            table.add_row(
                e.feed_id,
                e.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                str(round(e.duration_seconds, 1)),
                str(e.clip_path),
            )

    def _refresh_metrics(self):
        stats = self.store.stats()
        lines = [
            "[bold]Total Detections:[/bold] " + str(stats["total"]),
            "[bold]Reference Faces Loaded:[/bold] " + str(len(self.face_store.names())),
            "",
            "[bold]By Feed:[/bold]",
        ]
        for fid, cnt in stats.get("by_feed", {}).items():
            lines.append("  " + fid + ": " + str(cnt))
        lines.append("")
        lines.append("[bold]By Detector Type:[/bold]")
        for dt, cnt in stats.get("by_type", {}).items():
            lines.append("  " + dt + ": " + str(cnt))
        lines.append("")
        lines.append("[bold]Motion Events:[/bold] " + str(len(self.store.all_motion_events())))
        self.query_one("#metrics-display", Static).update("\n".join(lines))

    def _auto_refresh(self):
        self._populate_feeds_table()

    def _on_new_detection(self, detection: Detection):
        banner = self.query_one("#notification-banner", NotificationBanner)
        msg = (
            "[" + detection.feed_id + "] " + detection.label +
            " at " + str(round(detection.timestamp_seconds, 1)) + "s" +
            " (conf: " + str(round(detection.confidence, 2)) + ")"
        )
        banner.show_alert(msg, severity="warning")

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.data_table.id == "feeds-table":
            self._selected_feed_id = str(event.row_key.value)

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-add-feed":
            self.action_add_feed()
        elif bid == "btn-start-sel":
            self.action_start_selected()
        elif bid == "btn-stop-sel":
            self.action_stop_selected()
        elif bid == "btn-start-all":
            self.action_start_all()
        elif bid == "btn-upload-face":
            self.action_upload_face()
        elif bid == "btn-query":
            self.action_query_detections()
        elif bid == "btn-refresh-feeds":
            self._populate_feeds_table()
        elif bid == "btn-refresh-detections":
            self._populate_detections_table()
        elif bid == "btn-refresh-faces":
            self._populate_faces_table()
        elif bid == "btn-refresh-motion":
            self._populate_motion_table()
        elif bid == "btn-refresh-metrics":
            self._refresh_metrics()

    def action_add_feed(self):
        self.push_screen(AddFeedModal(), self._handle_add_feed)

    def action_start_selected(self):
        banner = self.query_one("#notification-banner", NotificationBanner)
        if not self._selected_feed_id:
            banner.show_alert("Select a feed row first.", "error")
            return
        self.orchestrator.start_feed(self._selected_feed_id)
        banner.show_alert("Feed [" + self._selected_feed_id + "] starting...", "success")
        self._populate_feeds_table()

    def action_stop_selected(self):
        banner = self.query_one("#notification-banner", NotificationBanner)
        if not self._selected_feed_id:
            banner.show_alert("Select a feed row first.", "error")
            return
        self.orchestrator.stop_feed(self._selected_feed_id)
        banner.show_alert("Feed [" + self._selected_feed_id + "] stopping...", "warning")
        self._populate_feeds_table()

    def action_start_all(self):
        self.orchestrator.start_all()
        self._populate_feeds_table()

    def action_upload_face(self):
        self.push_screen(UploadFaceModal(), self._handle_upload_face)

    def action_query_detections(self):
        self.push_screen(QueryModal(), self._handle_query)

    def action_refresh_ui(self):
        self._populate_feeds_table()
        self._populate_detections_table()
        self._populate_faces_table()
        self._populate_motion_table()
        self._refresh_metrics()

    def _handle_add_feed(self, result):
        if not result:
            return
        banner = self.query_one("#notification-banner", NotificationBanner)
        feed_type = FeedType.LIVE if result["feed_type"] == "live" else FeedType.RECORDED
        detector_params = {}
        if result["detector_type"] == "anpr" and result["param"]:
            detector_params["target_plate"] = result["param"]
        self.orchestrator.register_feed(
            feed_id=result["feed_id"],
            source=result["source"],
            feed_type=feed_type,
            detector_type=result["detector_type"],
            label=result["label"],
            detector_params=detector_params,
        )
        self._populate_feeds_table()
        banner.show_alert(
            "Feed '" + result["feed_id"] + "' added. Select it and press S to start.",
            "success",
        )

    def _handle_upload_face(self, result):
        if not result:
            return
        banner = self.query_one("#notification-banner", NotificationBanner)
        det = self._get_any_face_detector()
        if det is None:
            from src.detector_face import FaceRecognitionDetector
            det = FaceRecognitionDetector(
                "_tmp", self.store, self.registry, Path("/tmp"), self.face_store
            )
        embedding = det.get_embedding(result["path"])
        if embedding is None:
            banner.show_alert(
                "Could not generate embedding for '" + result["name"] + "' — check path.",
                "error",
            )
            return
        self.face_store.add_reference(result["name"], embedding)
        self._populate_faces_table()
        banner.show_alert("Face '" + result["name"] + "' loaded.", "success")

    def _get_any_face_detector(self):
        for feed in self.registry.all_feeds():
            if feed.detector_type == "face_recognition":
                det = self.orchestrator.get_face_detector_for_feed(feed.feed_id)
                if det:
                    return det
        return None

    def _handle_query(self, result):
        if not result:
            return
        detections = self.store.query(
            feed_id=result["feed_id"] or None,
            detector_type=result["detector_type"] or None,
            label=result["label"] or None,
        )
        self._populate_detections_table(detections)
