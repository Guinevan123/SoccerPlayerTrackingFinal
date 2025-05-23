import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

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
    def __init__(self, iou_threshold=0.5, max_age=10, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []  
        self.next_id = 1
        self.gt_tracks = {} 
        self.pred_tracks = {}
        self.frame_count = 0

    def update(self, detections, gt_detections=None):

        self.frame_count += 1
        
        if gt_detections is not None and len(gt_detections) > 0:
            self.gt_tracks[self.frame_count] = gt_detections
        
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
            results = np.array([track.get_state() for track in self.tracks])
            self.pred_tracks[self.frame_count] = results
            return results

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
        
        results = np.array(results) if results else np.empty((0, 5))
        self.pred_tracks[self.frame_count] = results
        return results

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

    def calculate_metrics(self):
        if not self.gt_tracks:
            return None, None, None

        id_switches = 0
        fragmentations = 0
        matches_per_frame = {}
        prev_matches = {}
        total_gt = 0
        total_pred = 0
        total_matches = 0
        
        
        for frame_id in sorted(self.gt_tracks.keys()):
            gt_boxes = self.gt_tracks[frame_id]
            pred_boxes = self.pred_tracks.get(frame_id, [])
            
            gt_ids = [int(gt[0]) for gt in gt_boxes]
            pred_ids = [int(pred[4]) for pred in pred_boxes]
            
            total_gt += len(gt_ids)
            total_pred += len(pred_boxes)
            
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for g, gt in enumerate(gt_boxes):
                for p, pred in enumerate(pred_boxes):
                    iou_matrix[g, p] = self._iou(gt[1:5], pred[:4])
            
            frame_matches = {}
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                for g_idx, p_idx in zip(row_ind, col_ind):
                    if iou_matrix[g_idx, p_idx] > 0.5:  # IOU threshold
                        gt_id = gt_ids[g_idx]
                        pred_id = pred_ids[p_idx]
                        frame_matches[gt_id] = pred_id
                        total_matches += 1
                        
                        if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                            id_switches += 1
                        
                        if gt_id not in prev_matches and frame_id > min(self.gt_tracks.keys()):
                            fragmentations += 1
            
            matches_per_frame[frame_id] = frame_matches
            prev_matches = frame_matches

        if total_gt == 0:
            return 0.0, 0.0, 0.0

        deta = total_matches / total_gt

        total_tracked_frames = sum(len(matches) for matches in matches_per_frame.values())
        if total_tracked_frames == 0:
            assa = 0
        else:
            id_switch_penalty = id_switches * 10  
            frag_penalty = fragmentations * 5 
            error_frames = id_switch_penalty + frag_penalty
            assa = max(0, (total_tracked_frames - error_frames)) / total_tracked_frames

        hota = np.sqrt(deta * assa)


        self.additional_metrics = {
            'id_switches': id_switches,
            'fragmentations': fragmentations,
            'total_matches': total_matches,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'total_tracked_frames': total_tracked_frames
        }

        return hota, deta, assa

def main():
    all_metrics = {}
    
    test_base_dir = 'test/test'
    detections_base_dir = 'detections'
    
    for folder in sorted(os.listdir(test_base_dir)):
        if not folder.startswith('SNMOT-'):
            continue
            
        print(f"\nProcessing {folder}...")
        
        det_path = os.path.join(detections_base_dir, folder, 'det.txt')
        gt_path = os.path.join(test_base_dir, folder, 'gt', 'gt.txt')
        img_dir = os.path.join(test_base_dir, folder, 'img1')
        
        if not all(os.path.exists(p) for p in [det_path, gt_path, img_dir]):
            print(f"Skipping {folder}: missing required files")
            continue
            
        tracker = SimpleTracker(iou_threshold=0.5, max_age=10, min_hits=3)
        
        detections = []
        with open(det_path, 'r') as f:
            for line in f:
                frame_id, _, x, y, w, h, conf = map(float, line.strip().split(','))
                if conf > 0.5: 
                    detections.append([frame_id, x, y, w, h, conf])
        detections = np.array(detections)
        
        ground_truth = {}
        with open(gt_path, 'r') as f:
            for line in f:
                frame_id, track_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
                if frame_id not in ground_truth:
                    ground_truth[frame_id] = []
                ground_truth[frame_id].append([track_id, x, y, w, h, conf])
        
        output_dir = os.path.join('output', folder)
        os.makedirs(output_dir, exist_ok=True)
        
        frame_ids = np.unique(detections[:, 0])
        
        for frame_id in frame_ids:
            frame_dets = detections[detections[:, 0] == frame_id]
            dets = frame_dets[:, 1:6]
            gt_dets = np.array(ground_truth.get(frame_id, []))
            tracked_objects = tracker.update(dets, gt_dets if len(gt_dets) > 0 else None)
            
            img_path = os.path.join(img_dir, f"{int(frame_id):06d}.jpg")
            if not os.path.exists(img_path):
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
            
            print(f'Processed {folder} frame {int(frame_id)}')

        hota, deta, assa = tracker.calculate_metrics()
        if hota is not None:
            all_metrics[folder] = {
                'HOTA': hota,
                'DetA': deta,
                'AssA': assa,
                'ID_Switches': tracker.additional_metrics['id_switches'],
                'Fragmentations': tracker.additional_metrics['fragmentations'],
                'Total_Matches': tracker.additional_metrics['total_matches'],
                'Total_GT': tracker.additional_metrics['total_gt'],
                'Total_Pred': tracker.additional_metrics['total_pred'],
                'Total_Tracked_Frames': tracker.additional_metrics['total_tracked_frames']
            }
            print(f"\nMetrics for {folder}:")
            print(f"HOTA: {hota:.3f}")
            print(f"DetA: {deta:.3f}")
            print(f"AssA: {assa:.3f}")
            print(f"ID Switches: {tracker.additional_metrics['id_switches']}")
            print(f"Fragmentations: {tracker.additional_metrics['fragmentations']}")
            print(f"Total Matches: {tracker.additional_metrics['total_matches']}")
            print(f"Total GT: {tracker.additional_metrics['total_gt']}")
            print(f"Total Predictions: {tracker.additional_metrics['total_pred']}")
            print(f"Total Tracked Frames: {tracker.additional_metrics['total_tracked_frames']}")

    if all_metrics:
        total_gt_all = sum(m['Total_GT'] for m in all_metrics.values())
        total_matches_all = sum(m['Total_Matches'] for m in all_metrics.values())
        total_tracked_frames_all = sum(m['Total_Tracked_Frames'] for m in all_metrics.values())
        total_id_switches = sum(m['ID_Switches'] for m in all_metrics.values())
        total_fragmentations = sum(m['Fragmentations'] for m in all_metrics.values())

        avg_deta = total_matches_all / total_gt_all if total_gt_all > 0 else 0

        if total_tracked_frames_all == 0:
            avg_assa = 0
        else:
            id_switch_penalty = total_id_switches * 10 
            frag_penalty = total_fragmentations * 5
            error_frames = id_switch_penalty + frag_penalty
            avg_assa = max(0, (total_tracked_frames_all - error_frames)) / total_tracked_frames_all

        avg_hota = np.sqrt(avg_deta * avg_assa)
        
        print("\nAverage metrics across all sequences:")
        print(f"HOTA: {avg_hota:.3f}")
        print(f"DetA: {avg_deta:.3f}")
        print(f"AssA: {avg_assa:.3f}")
        print(f"Total ID Switches: {total_id_switches}")
        print(f"Total Fragmentations: {total_fragmentations}")
        print(f"Average ID Switches per sequence: {total_id_switches/len(all_metrics):.1f}")
        print(f"Average Fragmentations per sequence: {total_fragmentations/len(all_metrics):.1f}")
        
        print("\nPer-sequence metrics:")
        for folder, metrics in all_metrics.items():
            print(f"\n{folder}:")
            print(f"HOTA: {metrics['HOTA']:.3f}")
            print(f"DetA: {metrics['DetA']:.3f}")
            print(f"AssA: {metrics['AssA']:.3f}")
            print(f"ID Switches: {metrics['ID_Switches']}")
            print(f"Fragmentations: {metrics['Fragmentations']}")

if __name__ == '__main__':
    main() 