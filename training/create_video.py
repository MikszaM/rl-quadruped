import cv2
import os
img_array = []

image_path = "rl-quadruped/training/videos/Random"
filenames = os.listdir(image_path)
filenames.sort(key=lambda x: os.stat(os.path.join(image_path, x)).st_mtime)

for filename in filenames:
    img = cv2.imread(os.path.join(image_path,filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter('rl-quadruped/training/videos/random.mp4',cv2.VideoWriter_fourcc(*'H264'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
    


