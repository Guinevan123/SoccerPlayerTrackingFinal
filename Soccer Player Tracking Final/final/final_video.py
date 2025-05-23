import os
import cv2

OUTPUT_DIR = 'final/tracking_results'
VIDEO_DIR = 'final/tracking_videos'
FPS = 25 

os.makedirs(VIDEO_DIR, exist_ok=True)

images = [img for img in os.listdir(OUTPUT_DIR) if img.endswith('.jpg')]
if not images:
    print(f'No images found in {OUTPUT_DIR}')
    exit(1)

images.sort()
first_img_path = os.path.join(OUTPUT_DIR, images[0])
frame = cv2.imread(first_img_path)
height, width, layers = frame.shape
video_path = os.path.join(VIDEO_DIR, 'tracking.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
for img_name in images:
    img_path = os.path.join(OUTPUT_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
video.release()
print(f'Created video: {video_path}') 