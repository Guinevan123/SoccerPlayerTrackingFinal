# Soccer Player Tracking

This project implements a complete pipeline for detecting and tracking soccer players in video footage using YOLO for detection and SORT (Simple Online Real-time Tracking) for tracking. The pipeline extracts frames from a video, detects players, tracks them across frames, and produces both annotated images and a final video.

## Project Workflow Summary

1. **Data Preparation & Model Training**
   - Downloaded the SoccerNet dataset, which contains annotated soccer game footage.
   - Fine-tuned a YOLO model specifically for soccer player detection, focusing on relevant classes.
   - Experimented with different YOLO configurations to optimise detection performance.

2. **Detection Evaluation**
   - Used the test split of the SoccerNet dataset to generate detection results with the trained YOLO model.
   - Compared YOLO's detection outputs to the provided ground truth annotations.
   - Calculated a variety of detection metrics, including mean Average Precision (mAP) and others, to assess and improve model performance.

3. **Tracking & Metric Analysis**
   - Applied the SORT (Simple Online Real-time Tracking) algorithm to the YOLO detections to track players across frames.
   - Evaluated tracking performance using metrics such as HOTA (Higher Order Tracking Accuracy), DetA (Detection Accuracy), and AssA (Association Accuracy).
   - Iteratively improved both detection and tracking by analyzing these metrics and adjusting.

4. **Full Game Processing & Final Output**
   - Selected a full 45-minute soccer game video for end-to-end processing.
   - Split the video into individual frames using a frame extraction script.
   - Ran the fine-tuned YOLO model on each frame to detect players.
   - Used SORT to track detected players across all frames, assigning consistent IDs.
   - Combined the tracking-annotated frames into a final output video, visually showing tracked player movements throughout the match.

**In summary:**
This project covers the entire pipeline from data acquisition and model training, through evaluation and experimentation, to real-world application on full-length soccer games, culminating in a video that visualises player tracking results.

## Features

- **Frame Extraction:** Extracts frames from an input `.mkv` video.
- **Player Detection:** Uses a YOLO model to detect players in each frame.
- **Tracking:** Applies the SORT algorithm to assign consistent IDs to players across frames.
- **Visualization:** Saves annotated frames with bounding boxes and IDs.
- **Video Creation:** Compiles the annotated frames into a video for easy viewing.

---

## Directory Structure

```
Soccer Player Tracking Final/
│
├── final/
│   ├── extract_frames.py         
│   ├── final_detection.py        
│   ├── final_track.py            
│   ├── final_video.py            
│   ├── 1_720p.mkv                
│   ├── extracted_frames/        
│   ├── detection_results/        
│   ├── detections/               
│   ├── tracking_results/         
│   └── tracking_videos/          
└── runs/
    └── detect/
        └── soccer_tracking/
            └── weights/
                └── best.pt
```

---

## Step-by-Step Usage

### 1. Extract Frames from Video

Place input video (e.g., `1_720p.mkv`) in the `final/` directory.

```bash
python final/extract_frames.py
```
- **Input:** `final/1_720p.mkv`
- **Output:** Frames saved in `final/extracted_frames/`

---

### 2. Run Player Detection

Make sure YOLO model weights are at `runs/detect/soccer_tracking/weights/best.pt`.

```bash
python final/final_detection.py
```
- **Input:** Frames from `final/extracted_frames/`
- **Output:** 
  - Detection visualisations in `final/detection_results/`
  - Detection data in MOT format in `final/detections/det.txt`

---

### 3. Run Tracking

```bash
python final/final_track.py
```
- **Input:** 
  - Detections from `final/detections/det.txt`
  - Frames from `final/extracted_frames/`
- **Output:** Tracked frames with bounding boxes and IDs in `final/tracking_results/`

---

### 4. Create Output Video

```bash
python final/final_video.py
```
- **Input:** Tracked frames from `final/tracking_results/`
- **Output:** Video at `final/tracking_videos/tracking.mp4`

---
