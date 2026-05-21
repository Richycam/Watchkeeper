# WATCHKEEPER 


<img width="1854" height="1001" alt="Screenshot from 2026-05-18 23-23-31" src="https://github.com/user-attachments/assets/a702b327-a649-4ff5-a676-21b81bb73493" />





---

## What It Does

WATCHKEEPER is a terminal-based surveillance tool that ingests recorded video files and live RTSP camera streams, runs automated detectors against them, and presents all results inside a keyboard-driven interface directly in your terminal. No GUI, no browser

You add feeds on demand, start and stop them individually, and review detections in real time — all without leaving the terminal.

---



### 1. Create the virtual environment and install dependencies

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a `venv/` folder and installs all Python packages automatically.

### 2. Activate the environment

```bash
source venv/bin/activate
```

### 3. Launch

```bash
python run.py
```

The TUI will open immediately. No feeds are loaded at startup — you add them yourself.

---

## Optional Launch Flags

These are only needed if you want to pre-load a reference face before the UI opens:

| Flag | Value | Purpose |
|---|---|---|
| `--face` | `/path/to/image.jpg` | Path to a reference face image |
| `--name` | `John_Doe` | Label to assign to that face |

**Example:**
```bash
python run.py --face /home/richard/suspects/obj_b.jpg --name Obj_B
```

---

## Interface Overview

The interface has five tabs. Navigate between them with the mouse or arrow keys.

```
┌─ WATCHKEEPER v5 ──────────────────────────────────────────────┐
│ ALERT: [feed_cam1] Red Vehicle at 12.5s (conf: 0.87)          │
├──────────────────────────────────────────────────────────────-─┤
│ Feeds │ Detections │ Faces │ Motion Events │ Metrics           │
├───────────────────────────────────────────────────────────────-┤
│                                                                │
│  [Add Feed A]  [Start Selected S]  [Stop Selected X]          │
│  [Start All]   [Refresh R]                                     │
│                                                                │
│  Feed ID       Type      Detector    Label      Status  Prog   │
│  feed_cam1     recorded  red_car     Car Park   idle    0%     │
│                                                                │
└───────────────────────────────────────────────────────────────-┘
│ ^Q Quit  ^A Add Feed  ^S Start  ^X Stop  ^U Upload  ^R Refresh │
└───────────────────────────────────────────────────────────────-┘
```

### Tab Summary

| Tab | Purpose |
|---|---|
| **Feeds** | Add, start, and stop feeds. Shows live status and progress |
| **Detections** | Table of everything detected across all feeds. Filterable |
| **Faces** | Lists loaded reference face embeddings |
| **Motion Events** | Log of motion clips recorded from live feeds |
| **Metrics** | Detection counts broken down by feed and detector type |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` | Open "Add Feed" dialog |
| `S` | Start the currently selected feed |
| `X` | Stop the currently selected feed |
| `U` | Open "Upload Reference Face" dialog |
| `F` | Open "Query / Filter Detections" dialog |
| `R` | Refresh all tables |
| `Q` | Quit |

All shortcuts also have matching buttons on-screen for mouse users.

---

## Adding a Feed

Press **`A`** (or click **Add Feed**). A dialog will appear with the following fields:

| Field | What to Enter |
|---|---|
| **Feed ID** | A unique name with no spaces, e.g. `feed_cam1` or `feed_entrance` |
| **Source** | Full file path (`/home/richard/footage/clip1.mp4`) or RTSP URL (`rtsp://192.168.1.50/cam`) |
| **Feed Type** | `Recorded Video File` for MP4/AVI files — `Live RTSP Stream` for cameras |
| **Detector Type** | See the Detectors section below |
| **Label** | A friendly description shown in the table, e.g. `Car Park Camera A` |
| **Target Plate** | ANPR feeds only — the licence plate to search for, e.g. `BP63 LYH`. Leave blank for all other detector types |

Click **Add Feed** to register it. The feed appears in the Feeds table with status `idle`. It will not start until you explicitly start it.

---

## Starting and Stopping Feeds

1. **Click a row** in the Feeds table to select it (the row highlights).
2. Press **`S`** or click **Start Selected** to start it.
3. Press **`X`** or click **Stop Selected** to stop it at any time.

You can also click **Start All** to start every registered feed simultaneously.

**Status colours in the table:**

| Colour | Status | Meaning |
|---|---|---|
| Dim | `idle` | Registered but not yet started |
| Yellow | `processing` | Actively running |
| Green | `complete` | Finished processing (recorded feeds) |
| Cyan | `stopped` | Manually stopped mid-run |
| Red | `error` | Failed to open source — check the path or URL |

Recorded video feeds automatically reach `complete` when the file ends. Live RTSP feeds stay `processing` until you stop them with `X`.

---

## Detectors

### Red Car Detection
- Analyses each sampled frame in HSV colour space.
- Identifies regions matching red hue (both ends of the hue wheel, 0–15° and 155–180°) with sufficient saturation and brightness.
- Filters by contour shape (aspect ratio and minimum area) to exclude noise.
- Works on recorded video files. CPU only — no GPU required.

### Face Recognition
- project orginally used Deep face deprciated, now using face_recognition

### ANPR (Licence Plate Recognition)
- Uses EasyOCR to read text from candidate regions in each frame.
- Candidate regions are found using edge detection and contour filtering (aspect ratio matching a number plate).
- Compares read text against your target plate after normalising both (uppercase, spaces and hyphens removed).
- Accepts exact matches and high-confidence partial matches to handle occlusion or OCR misreads.
- **Note:** EasyOCR downloads its English language model (~40 MB) on first run.

### Motion Detection (Live feeds only)
- Uses OpenCV MOG2 background subtraction — learns the static background over time and flags pixels that deviate from it.
- When significant motion is detected, automatically starts recording an AVI clip.
- Recording continues for 2 seconds after motion stops, then the clip is saved to `output/thumbnails/<feed_id>/`.
- Each recorded clip appears in the **Motion Events** tab.

---

## Uploading a Reference Face

Press **`U`** (or click **Upload Reference Face**) to load a photo of a person for face recognition:

1. Enter a name or label (e.g. `Obj_B` or `John_Doe`).
2. Enter the full path to a `.jpg` or `.png` image of their face.
3. Click **Upload**.

WATCHKEEPER generates a face embedding from the image and stores it in memory. All running face recognition feeds immediately start comparing against this reference — no restart needed. Loaded faces appear in the **Faces** tab.

---

## Querying Detections

Press **`F`** (or click **Query / Filter**) to search detections:

- **Feed ID** — filter to one specific feed, e.g. `feed_cam1`
- **Detector Type** — filter by `red_car`, `face_recognition`, `anpr`, or `motion`
- **Label keyword** — filter by partial label text, e.g. `Red Vehicle`, `Match`, or `Plate`

Leave any field blank to match all. Results replace the Detections table view.

---

## Output Files

All output is written to an `output/` directory created in the folder where you run `run.py`:

```
output/
├── detections/
│   └── detections.json          # All detections, persisted to disk
└── thumbnails/
    ├── feed_cam1/
    │   ├── feed_cam1_12.5_red_car.jpg    # Annotated frame at moment of detection
    │   └── feed_cam1_45.0_red_car.jpg
    └── feed_live/
        └── motion_20260518_221500.avi    # Motion clip recordings
```

Detections are saved to `detections.json` after every new detection, so results are preserved if the program is closed and restarted.

---

## Notification Banner

A yellow banner appears at the top of the screen whenever a new detection occurs:

```
ALERT: [feed_cam1] Red Vehicle at 12.5s (conf: 0.87)
```

The banner disappears automatically after 6 seconds.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Frame reading, colour detection, MOG2, Haar cascades |
| `deepface` | Face detection and Facenet512 embedding |
| `easyocr` | Number plate OCR |
| `textual` | Terminal UI framework |
| `numpy` | Embedding maths (cosine similarity) |

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Feed shows `error` | Path or RTSP URL is wrong | Check the source path — re-add the feed with the correct value |
| Face recognition uses "Unknown Face" for everyone | No reference faces loaded | Press `U` and upload a reference image |
| ANPR finds no detections | Poor video quality or wrong plate format | Ensure the plate string in the feed matches the format on-screen (e.g. `BP63 LYH`) |
| DeepFace slow on first detection | Downloading model weights | Wait — subsequent detections will be faster once the model is cached |
| Motion detector shows `error` | RTSP stream unreachable | Verify the camera IP and stream URL are accessible from this machine |
| `ModuleNotFoundError` on launch | Virtual environment not activated | Run `source venv/bin/activate` before `python run.py` |
