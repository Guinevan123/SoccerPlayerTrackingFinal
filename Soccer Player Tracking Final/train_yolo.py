import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import glob

def download_yolo_model():
    """Download YOLOv8 model if it doesn't exist"""
    if not os.path.exists('yolov8s.pt'):
        print("Downloading YOLOv8 model...")
        model = YOLO('yolov8s.pt')
        print("Download complete!")
    else:
        print("YOLOv8 model already exists!")

def get_class_id(track_id, gameinfo_file):
    """Map track ID to class ID based on gameinfo.ini"""
    with open(gameinfo_file, 'r') as f:
        content = f.read()
        
    tracklet_line = None
    for line in content.split('\n'):
        if f'trackletID_{track_id}=' in line:
            tracklet_line = line
            break
    
    if not tracklet_line:
        return None
        
    role = tracklet_line.split('=')[1].strip()
    
    if 'player team left' in role:
        return 0  # player_team_left
    elif 'player team right' in role:
        return 1  # player_team_right
    elif 'referee' in role:
        return 2  # referee
    elif 'goalkeepers team left' in role or 'goalkeeper team left' in role:
        return 3  # goalkeeper_team_left
    elif 'goalkeeper team right' in role:
        return 4  # goalkeeper_team_right
    elif 'ball' in role:
        return 5  # ball
    return None

def convert_to_yolo_format():
    seq_dir = Path('train/SNMOT-060')
    labels_dir = seq_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    print(f"Processing sequence: {seq_dir.name}")
    print(f"Labels will be saved to: {labels_dir}")
        
    gt_file = seq_dir / 'gt' / 'gt.txt'
    gameinfo_file = seq_dir / 'gameinfo.ini'
    
    if not gt_file.exists() or not gameinfo_file.exists():
        print(f"Error: Missing required files in {seq_dir.name}")
        return
        
    first_frame_processed = False
    with open(gt_file, 'r') as f:
        for line in f:
            try:
                frame_id, track_id, x, y, w, h, _, _, _, _ = map(float, line.strip().split(','))
                
                if frame_id > 1 and first_frame_processed:
                    break
                
                class_id = get_class_id(int(track_id), gameinfo_file)
                if class_id is None:
                    continue
                
                img_width = 1920 
                img_height = 1080
                
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                frame_name = f'{int(frame_id):06d}.jpg'
                label_file = labels_dir / f'{frame_name[:-4]}.txt'
                
                with open(label_file, 'a') as lf:
                    lf.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
                
                first_frame_processed = True
                
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {str(e)}")
                continue

def prepare_all_snmot_dataset():
    """Prepare a dataset with all frames from all SNMOT-* sequences in train/"""
    temp_dir = Path('temp_dataset')
    train_dir = temp_dir / 'train'
    train_images = train_dir / 'images'
    train_labels = train_dir / 'labels'
    if train_images.exists() and train_labels.exists():
        print(f"Images and labels already exist in {train_dir}, skipping copy.")
        return True

    temp_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    train_images.mkdir(exist_ok=True)
    train_labels.mkdir(exist_ok=True)

    snmot_dirs = sorted(Path('train').glob('SNMOT-*'))
    img_count = 0
    for seq_dir in snmot_dirs:
        seq_name = seq_dir.name
        img1_dir = seq_dir / 'img1'
        labels_dir = seq_dir / 'labels'
        if not img1_dir.exists() or not labels_dir.exists():
            print(f"Warning: {seq_dir} missing img1 or labels directory.")
            continue
        for img_path in sorted(img1_dir.glob('*.jpg')):
            label_path = labels_dir / (img_path.stem + '.txt')
            if img_path.exists() and label_path.exists():
                new_img_name = f"{seq_name}_{img_path.name}"
                new_label_name = f"{seq_name}_{label_path.name}"
                shutil.copy2(img_path, train_images / new_img_name)
                shutil.copy2(label_path, train_labels / new_label_name)
                img_count += 1
            else:
                print(f"Warning: Could not find {img_path.name} or {label_path.name} in {seq_dir}")
    print(f"Copied {img_count} images and their labels from all SNMOT sequences to {train_dir}")
    return img_count > 0

def train_yolo():
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='yolov8.yaml',
        epochs=15,
        imgsz=640,
        batch=4,
        name='soccer_tracking',
        save=True,
        save_period=1,
        plots=True,
        verbose=True
    )
    
    print("\nTraining Metrics:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.3f}")

def get_latest_weights():
    dirs = glob.glob('runs/detect/soccer_tracking*/weights/best.pt')
    if not dirs:
        raise FileNotFoundError("No weights found in runs/detect/")
    return max(dirs, key=os.path.getctime)

if __name__ == '__main__':
    print("Downloading YOLOv8 model")
    download_yolo_model()
    
    print("Preparing all SNMOT sequences dataset")
    if prepare_all_snmot_dataset():
        print("Starting YOLO training")
        train_yolo()
        
        print("Running prediction on the first image")
        model = YOLO(get_latest_weights())
        img_path = 'temp_dataset/train/images/000001.jpg'
        results = model.predict(img_path, save=True, imgsz=640, conf=0.5, show_labels=True, show_conf=True)
        print("Prediction complete, runs/detect/predict directory for the output image.")
    else:
        print("Failed to prepare dataset. Exiting.") 