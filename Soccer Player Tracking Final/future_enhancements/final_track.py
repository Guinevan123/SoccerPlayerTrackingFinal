import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from pathlib import Path

class Track:
    def __init__(self, detection, track_id):
        x, y, w, h, conf = detection
        
        self.kf = KalmanFilter(dim_x=8, dim_z=4) 
        

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  
            [0, 1, 0, 0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0, 0, 1, 0], 
            [0, 0, 0, 1, 0, 0, 0, 1], 
            [0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 1],  
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        self.kf.R *= 5  
        
        self.kf.Q[4:, 4:] *= 0.05 
        
        self.kf.x[:4] = np.array([x, y, w, h]).reshape(-1, 1)
        
        self.track_id = track_id
        self.time_since_update = 0
        self.hit_streak = 0
        
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()
        
    def update(self, detection):
        self.kf.update(detection.reshape(-1, 1))
        self.time_since_update = 0
        self.hit_streak += 1
        
    def get_state(self):
        state = self.kf.x[:4].flatten()
        return np.array([state[0], state[1], state[2], state[3], self.track_id])

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=10, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """Update tracks using detection data."""
        predicted_tracks = []
        remaining_tracks = []
        for track in self.tracks:
            pred = track.predict()
            if track.time_since_update < self.max_age:
                predicted_tracks.append(pred)
                remaining_tracks.append(track)
        
        self.tracks = remaining_tracks
        predicted_tracks = np.array(predicted_tracks) if predicted_tracks else np.empty((0, 5))

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return np.array([track.get_state() for track in self.tracks])

        iou_matrix = np.zeros((len(detections), len(predicted_tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(predicted_tracks):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.column_stack((row_ind, col_ind))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                self.tracks[m[1]].update(detections[m[0], :4])
            else:
                unmatched_detections.append(m[0])

        for d in unmatched_detections:
            new_track = Track(detections[d], self.next_id)
            self.next_id += 1
            self.tracks.append(new_track)

        results = []
        for track in self.tracks:
            if track.time_since_update < 1 and track.hit_streak >= self.min_hits:
                results.append(track.get_state())
        
        return np.array(results) if results else np.empty((0, 5))

    def _iou(self, box1, box2):
        """Calculate IOU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

def main():
    current_dir = os.getcwd()
    
    detections_dir = os.path.join(current_dir, 'future_enhancements', 'detections')
    frames_dir = os.path.join(current_dir, 'future_enhancements', 'extracted_frames')
    output_dir = os.path.join(current_dir, 'future_enhancements', 'tracking_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading detections from: {detections_dir}")
    print(f"Reading frames from: {frames_dir}")
    print(f"Saving tracking results to: {output_dir}")
    
    tracker = SimpleTracker(iou_threshold=0.3, max_age=10, min_hits=3)
    
    detections = []
    det_path = os.path.join(detections_dir, 'det.txt')
    with open(det_path, 'r') as f:
        for line in f:
            frame_id, _, x, y, w, h, conf = map(float, line.strip().split(','))
            if conf > 0.5:
                detections.append([frame_id, x, y, w, h, conf])
    detections = np.array(detections)
    
    frame_ids = np.unique(detections[:, 0])
    
    for frame_id in frame_ids:
        frame_dets = detections[detections[:, 0] == frame_id]
        
        dets = frame_dets[:, 1:6]  # Skip frame_id
        
        tracked_objects = tracker.update(dets)
        
        img_path = os.path.join(frames_dir, f"{int(frame_id):06d}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: Frame {frame_id} not found at {img_path}")
            continue
            
        img = cv2.imread(img_path)
        
        if tracked_objects is not None and len(tracked_objects) > 0:
            for track in tracked_objects:
                x, y, w, h, track_id = track
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(img, f'ID: {int(track_id)}', (int(x), int(y - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        output_path = os.path.join(output_dir, f"{int(frame_id):06d}.jpg")
        cv2.imwrite(output_path, img)
        
        print(f'Processed frame {int(frame_id)}')

if __name__ == '__main__':
    main() 