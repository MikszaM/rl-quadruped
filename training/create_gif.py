import imageio
import os
images = []

image_path = "rl-quadruped/training/videos/Training gif/500000"
filenames = os.listdir(image_path)
filenames.sort(key=lambda x: os.stat(os.path.join(image_path, x)).st_mtime)

for filename in filenames:
    images.append(imageio.imread(os.path.join(image_path,filename)))
imageio.mimsave('rl-quadruped/training/videos/Training gif/trained500000.gif', images, duration=1./30.)