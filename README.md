# ⚽ Football Player Detection and Tracking with YOLOv11

This project performs real-time detection and tracking of football players in a 15-second video using a pre-trained **YOLOv11** model. The goal is to maintain player identity across frames, even when players leave and re-enter the frame.

---

## 📁 Project Structure
```bash
project-root/
│
├── main.py # Main tracking script
├── yolov11_model.pt # Trained YOLOv11 model (not shared here)
├── 15sec_input_720p.mp4 # Input video
├── tracked_match.mp4 # Output video (generated)
└── requirements.txt # Required Python packages
```
---

## ⚙️ Setup Instructions

### 1. Create and Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate.bat       # Windows
```

pip install -r requirements.txt
## How to Run
Use the following command to run the tracking pipeline:
```bash
python main.py --model yolov11_model.pt --video 15sec_input_720p.mp4 --output tracked_match.mp4 --show
```

## Tracking Output Summary
When this script runs, you’ll get statistics like:

Total unique players detected: 39

Frames processed: 375

Average track length: ~14.9 frames

Each player is tracked across frames and assigned a consistent ID, even with some occlusions.

# Player Re-Identification Report for Football Match Video

## Objective

The goal of this project was to **detect and track football players** in a 15-second video using a pre-trained YOLOv11 model. The system aims to assign a **unique and consistent ID** to each player across frames, forming the basis of a simple **re-identification pipeline**.

---

## Approach & Methodology

### 1. **Model**
- Used a custom-trained **YOLOv11 model** (`yolov11_model.pt`) for object detection (players).
- Detection confidence threshold was kept default (e.g., 0.25).

### 2. **Tracking**
- Basic IOU-based tracker was used.
- Every detection in a frame was matched to existing tracks based on bounding box overlap.
- A simple track management system maintained consistent IDs.

### 3. **Pipeline Overview**
1. Load YOLOv11 model.
2. Read video frame-by-frame.
3. Run player detection on each frame.
4. Update tracker with new detections.
5. Save output with overlaid bounding boxes and IDs.
6. Collect tracking stats (unique players, frames, etc.).

---

## Experiments & Outcomes

| Metric                   | Result                     |
|--------------------------|----------------------------|
| Video length             | 15 seconds (375 frames)    |
| Resolution               | 1280x720                   |
| Total unique players     | 39                         |
| Average track length     | ~14.9 frames               |

The model was able to **consistently assign IDs** to players even through short occlusions or motion blur.

---

## Techniques Tried

| Technique                    | Outcome/Remarks                                                  |
|-----------------------------|------------------------------------------------------------------|
| YOLOv11 + IOU Tracking       | ✅ Worked reliably for short videos                              |
| DeepOCSort integration       | ❌ Faced import issues / compatibility errors in Colab & local   |
| BoxMOT + YOLOv11 combo       | ❌ Could not be installed due to repo cloning/authentication issues |
| PyTorch Unpickling override  | ✅ Manually resolved using trusted deserialization options       |

---

## Challenges Faced

- **Model format issues**: The provided YOLOv11 model failed under strict `torch.load()` defaults (fixed with `weights_only=False`).
- **Dependency Conflicts**: Ultralytics and BoxMOT had compatibility/version issues.
- **Re-ID Limitations**: Without appearance-based re-identification, players leaving and re-entering could be assigned new IDs.
- **Environment Instability**: Frequent Colab restarts and broken package installs slowed progress.

---

## Remaining Work

This project functions as a baseline. With more time/resources, the following improvements are planned:

- 🔄 **Integrate appearance-based Re-ID** (e.g., DeepOCSort or a ReIDNet).
- 🧬 **Improve ID persistence** using deep feature embeddings.
- 🎞️ **Visual Analytics**: Add heatmaps, trajectory plots, or per-player stats.
- 💾 **Save to JSON/CSV**: Output track data for downstream analytics.

---

## Conclusion

Despite constraints, the system achieves **basic tracking and re-identification** with YOLOv11 on a short football match clip. This forms a good foundation for more advanced tracking systems.


