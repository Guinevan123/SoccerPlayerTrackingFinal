import os
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def process_detections(model_path, test_dir, results_dir=None, detections_dir=None, conf_threshold=0.5):
    model = YOLO(model_path)

    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if detections_dir and not os.path.exists(detections_dir):
        os.makedirs(detections_dir, exist_ok=True)

    det_file = None
    if detections_dir:
        det_file = open(os.path.join(detections_dir, 'det.txt'), 'w')

    for image_name in tqdm(sorted(os.listdir(test_dir))):
        if not image_name.endswith(('.jpg', '.png')):
            continue
            
        image_path = os.path.join(test_dir, image_name)
        frame_id = int(image_name.split('.')[0])
        
        results = model(image_path, conf=conf_threshold)[0]
        predictions = results.boxes.data.cpu().numpy()

        if results_dir:
            out_path = os.path.join(results_dir, image_name)
            results.save(filename=out_path)

        if det_file:
            for pred in predictions:
                x1, y1, x2, y2 = pred[:4]
                w = x2 - x1
                h = y2 - y1
                score = float(pred[4]) if len(pred) > 4 else 1.0
                det_file.write(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.2f}\n")

    if det_file:
        det_file.close()

def main():
    current_dir = os.getcwd()
    
    model_path = os.path.join(current_dir, 'runs', 'detect', 'soccer_tracking', 'weights', 'best.pt')
    test_dir = os.path.join(current_dir, 'future_enhancements', 'extracted_frames')
    results_dir = os.path.join(current_dir, 'future_enhancements', 'detection_results')
    detections_dir = os.path.join(current_dir, 'future_enhancements', 'detections')
    
    print(f"Model path: {model_path}")
    print(f"Processing frames from: {test_dir}")
    print(f"Saving results to: {results_dir}")
    print(f"Saving detections to: {detections_dir}")
    
    process_detections(
        model_path=model_path,
        test_dir=test_dir,
        results_dir=results_dir,
        detections_dir=detections_dir
    )
    print("Detection complete!")
    print("\nDetection data has been saved in MOT format for SORT:")
    print(f"- File: {os.path.join(detections_dir, 'det.txt')}")
    print("- Format: frame_id, -1, x, y, w, h, score")

if __name__ == "__main__":
    main() 