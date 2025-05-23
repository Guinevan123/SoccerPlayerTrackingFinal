import os
from pathlib import Path

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
        return 0
    elif 'player team right' in role:
        return 1
    elif 'referee' in role:
        return 2
    elif 'goalkeepers team left' in role or 'goalkeeper team left' in role:
        return 3
    elif 'goalkeeper team right' in role:
        return 4
    elif 'ball' in role:
        return 5
    return None

def convert_all_snmots_to_yolo(train_root='train'):
    snmot_dirs = sorted(Path(train_root).glob('SNMOT-*'))
    for seq_dir in snmot_dirs:
        gt_file = seq_dir / 'gt' / 'gt.txt'
        gameinfo_file = seq_dir / 'gameinfo.ini'
        img_dir = seq_dir / 'img1'
        labels_dir = seq_dir / 'labels'
        seqinfo_file = seq_dir / 'seqinfo.ini'
        if not gt_file.exists() or not gameinfo_file.exists() or not img_dir.exists() or not seqinfo_file.exists():
            print(f"Skipping {seq_dir}: missing gt.txt, gameinfo.ini, img1, or seqinfo.ini")
            continue
        with open(seqinfo_file, 'r') as f:
            lines = f.readlines()
        img_width = None
        img_height = None
        for line in lines:
            if line.startswith('imWidth='):
                img_width = int(line.split('=')[1].strip())
            if line.startswith('imHeight='):
                img_height = int(line.split('=')[1].strip())
        if img_width is None or img_height is None:
            print(f"Skipping {seq_dir}: could not find image size in seqinfo.ini")
            continue
        labels_dir.mkdir(exist_ok=True)
        print(f"Converting {seq_dir}...")
        with open(gt_file, 'r') as f:
            for line in f:
                try:
                    frame_id, track_id, x, y, w, h, _, _, _, _ = map(float, line.strip().split(','))
                    class_id = get_class_id(int(track_id), gameinfo_file)
                    if class_id is None:
                        continue
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    frame_name = f'{int(frame_id):06d}.txt'
                    label_file = labels_dir / frame_name
                    with open(label_file, 'a') as lf:
                        lf.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
                except Exception as e:
                    print(f"Error processing line in {seq_dir}: {line.strip()} | Error: {str(e)}")
                    continue
        print(f"Done: {seq_dir}")

if __name__ == '__main__':
    convert_all_snmots_to_yolo('train')
    print("All SNMOT ground truth files converted to YOLO format.") 