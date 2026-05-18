import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_ROOT    = Path("output")
DETECTIONS_DIR = OUTPUT_ROOT / "detections"
THUMBNAILS_DIR = OUTPUT_ROOT / "thumbnails"


def parse_args():
    args = sys.argv[1:]
    params = {}
    for flag in ["--face", "--name"]:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                params[flag] = args[idx + 1]
    return params


def main():
    cli = parse_args()

    from src.detection_store import DetectionStore
    from src.face_store import FaceStore
    from src.feed_registry import FeedRegistry
    from src.orchestrator import FeedOrchestrator
    from src.app import WatchkeeperApp

    store        = DetectionStore(persist_path=DETECTIONS_DIR)
    registry     = FeedRegistry()
    face_store   = FaceStore()
    orchestrator = FeedOrchestrator(store, registry, face_store, THUMBNAILS_DIR)

    # Pre-load a reference face if supplied on the command line
    if cli.get("--face") and cli.get("--name"):
        from src.detector_face import FaceRecognitionDetector
        tmp = FaceRecognitionDetector(
            "_init", store, registry, THUMBNAILS_DIR / "_init", face_store
        )
        emb = tmp.get_embedding(cli["--face"])
        if emb is not None:
            face_store.add_reference(cli["--name"], emb)

    # No feeds registered at startup — add them all from the TUI
    WatchkeeperApp(
        store=store, registry=registry,
        orchestrator=orchestrator, face_store=face_store,
    ).run()


if __name__ == "__main__":
    main()
