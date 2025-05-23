import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

def load_ground_truth(gt_path):
    gt_data = []
    with open(gt_path, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            frame = int(fields[0])
            obj_id = int(fields[1])
            x = float(fields[2])
            y = float(fields[3])
            w = float(fields[4])
            h = float(fields[5])
            gt_data.append({
                'frame': frame,
                'id': obj_id,
                'bbox': [x, y, w, h]
            })
    return gt_data

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, 
            box1[0] + box1[2]/2, box1[1] + box1[3]/2]
    box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2,
            box2[0] + box2[2]/2, box2[1] + box2[3]/2]
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):

    gt_by_image = defaultdict(list)
    for img_id, box in gt_boxes:
        gt_by_image[img_id].append(box)

    pred_boxes = sorted(pred_boxes, key=lambda x: x[2], reverse=True)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    detected = defaultdict(set)

    for i, (img_id, pred_box, score) in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(gt_by_image[img_id]):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx not in detected[img_id]:
            tp[i] = 1
            detected[img_id].add(best_gt_idx)
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (len(gt_boxes) if len(gt_boxes) > 0 else 1)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += np.max(p) if p.size > 0 else 0
    ap /= 11
    return ap

def compute_map_range(pred_boxes, pred_scores, gt_boxes):
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        ap = compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=iou)
        aps.append(ap)
    return np.mean(aps)

def evaluate_model(model_path, test_dir, gt_path, results_dir=None, detections_dir=None, conf_threshold=0.5, iou_threshold=0.5, calc_map=False, snmot_name=None):
    model = YOLO(model_path)
    gt_data = load_ground_truth(gt_path)
    total_predictions = 0
    total_gt = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    pred_boxes_map = []
    gt_boxes_map = []

    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if detections_dir and not os.path.exists(detections_dir):
        os.makedirs(detections_dir, exist_ok=True)

    det_file = None
    if detections_dir:
        det_file = open(os.path.join(detections_dir, 'det.txt'), 'w')

    for image_name in tqdm(os.listdir(test_dir)):
        if not image_name.endswith(('.jpg', '.png')):
            continue
        image_path = os.path.join(test_dir, image_name)
        frame_id = int(image_name.split('.')[0])
        results = model(image_path, conf=conf_threshold)[0]
        predictions = results.boxes.data.cpu().numpy()
        if results_dir:
            out_name = f"{snmot_name}_{image_name}" if snmot_name else image_name
            out_path = os.path.join(results_dir, out_name)
            results.save(filename=out_path)
        if det_file:
            for pred in predictions:
                x1, y1, x2, y2 = pred[:4]
                w = x2 - x1
                h = y2 - y1
                score = float(pred[4]) if len(pred) > 4 else 1.0
                det_file.write(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.2f}\n")
        gt_boxes = []
        for ann in gt_data:
            if ann['frame'] == frame_id:
                x, y, w, h = ann['bbox']
                x_center = x + w / 2
                y_center = y + h / 2
                gt_boxes.append([x_center, y_center, w, h])
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                gt_boxes_map.append((frame_id, [x1, y1, x2, y2]))
        total_predictions += len(predictions)
        total_gt += len(gt_boxes)
        matched_gt = set()
        for pred in predictions:
            x1, y1, x2, y2 = pred[:4]
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            pred_box = [x_center, y_center, w, h]
            if calc_map:
                pred_boxes_map.append((frame_id, [x1, y1, x2, y2], float(pred[4]) if len(pred) > 4 else 1.0))
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        false_negatives += len(gt_boxes) - len(matched_gt)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    map50 = None
    map5095 = None
    if calc_map:
        map50 = compute_map(pred_boxes_map, None, gt_boxes_map, iou_threshold=0.5)
        map5095 = compute_map_range(pred_boxes_map, None, gt_boxes_map)
    if det_file:
        det_file.close()
    return {
        'total_predictions': total_predictions,
        'total_ground_truth': total_gt,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'map50': map50,
        'map5095': map5095
    }

def main():
    model_path = 'runs/detect/soccer_tracking/weights/best.pt'
    base_dir = 'test/test'
    results_base = 'test_results'
    detections_base = 'detections'
    if not os.path.exists(results_base):
        os.makedirs(results_base, exist_ok=True)
    if not os.path.exists(detections_base):
        os.makedirs(detections_base, exist_ok=True)
    overall_metrics = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path) or not folder.startswith('SNMOT-'):
            continue
        test_dir = os.path.join(folder_path, 'img1')
        gt_path = os.path.join(folder_path, 'gt', 'gt.txt')
        if not os.path.exists(test_dir) or not os.path.exists(gt_path):
            print(f"Skipping {folder}: missing img1 or gt.txt")
            continue
        print(f"\nEvaluating {folder}...")
        snmot_results_dir = os.path.join(results_base, folder)
        if not os.path.exists(snmot_results_dir):
            os.makedirs(snmot_results_dir, exist_ok=True)
        snmot_detections_dir = os.path.join(detections_base, folder)
        if not os.path.exists(snmot_detections_dir):
            os.makedirs(snmot_detections_dir, exist_ok=True)
        metrics = evaluate_model(model_path, test_dir, gt_path, results_dir=snmot_results_dir, detections_dir=snmot_detections_dir, calc_map=True, snmot_name=folder)
        overall_metrics.append(metrics)
        print("-" * 50)
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Ground Truth: {metrics['total_ground_truth']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f}" if metrics['map50'] is not None else "mAP@0.5: N/A")
        print(f"mAP@0.5:0.95: {metrics['map5095']:.4f}" if metrics['map5095'] is not None else "mAP@0.5:0.95: N/A")
    if overall_metrics:
        avg_precision = sum(m['precision'] for m in overall_metrics) / len(overall_metrics)
        avg_recall = sum(m['recall'] for m in overall_metrics) / len(overall_metrics)
        avg_f1 = sum(m['f1_score'] for m in overall_metrics) / len(overall_metrics)
        avg_map50 = sum(m['map50'] for m in overall_metrics if m['map50'] is not None) / max(1, sum(1 for m in overall_metrics if m['map50'] is not None))
        avg_map5095 = sum(m['map5095'] for m in overall_metrics if m['map5095'] is not None) / max(1, sum(1 for m in overall_metrics if m['map5095'] is not None))
        print("\nAverage across all games:")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1 Score: {avg_f1:.4f}")
        print(f"mAP@0.5: {avg_map50:.4f}")
        print(f"mAP@0.5:0.95: {avg_map5095:.4f}")

if __name__ == "__main__":
    main() 