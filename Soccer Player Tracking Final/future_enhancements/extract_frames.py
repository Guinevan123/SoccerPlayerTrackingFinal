import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_rate == 0:
                frame_path = os.path.join(output_dir, f"{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"\nExtraction complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {output_dir}")

def main():
    current_dir = os.getcwd()
    
    video_path = os.path.join(current_dir, "future_enhancements", "Future_Enhancement.mp4")
    output_dir = os.path.join(current_dir, "future_enhancements", "extracted_frames")
    
    print(f"Looking for video at: {video_path}")
    print(f"Will save frames to: {output_dir}")
    
    extract_frames(video_path, output_dir)
    

if __name__ == "__main__":
    main() 