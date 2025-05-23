import os
import cv2

OUTPUT_DIR = 'output'
VIDEO_DIR = 'videos'
FPS = 25 

os.makedirs(VIDEO_DIR, exist_ok=True)

for folder in sorted(os.listdir(OUTPUT_DIR)):
    folder_path = os.path.join(OUTPUT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
    if not images:
        continue
    images.sort()
    first_img_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_img_path)
    height, width, layers = frame.shape
    video_path = os.path.join(VIDEO_DIR, f'{folder}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
    video.release()
    print(f'Created video: {video_path}') 