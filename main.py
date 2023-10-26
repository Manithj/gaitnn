import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
from human_pose_nn import HumanPoseIRNetwork
import image_array as imgarr
import video_slicer as vs

mpl.use('Agg')

net_pose = HumanPoseIRNetwork()
net_pose.restore('./models/MPII+LSP.ckpt')

joint_names = [
    'right ankle ',
    'right knee ',
    'right hip',
    'left hip',
    'left knee',
    'left ankle',
    'pelvis',
    'thorax',
    'upper neck',
    'head top',
    'right wrist',
    'right elbow',
    'right shoulder',
    'left shoulder',
    'left elbow',
    'left wrist'
]


video_path1 = "1.0.mp4" # Replace with your relative video path
output_folder1 = "output_folder1"
video_path2 = "1.2.mp4" # Replace with your relative video path
output_folder2 = "output_folder2"

vs.video_frame_slicer(video_path1, output_folder1)
vs.video_frame_slicer(video_path2, output_folder2)

folder_path1 = './output_folder1'  
image_paths1 = imgarr.get_image_filepaths_from_folder(folder_path1)
print(image_paths1)

folder_path2 = './output_folder2'
image_paths2 = imgarr.get_image_filepaths_from_folder(folder_path2)
print(image_paths2)


def process_images(image_paths):
    all_x = []
    all_y = []
    for image_path in image_paths:
        img = imread(image_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = imresize(img, [299, 299])
        img_batch = np.expand_dims(img, 0)
        y, x, a = net_pose.estimate_joints(img_batch)
        y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)
        all_x.append(x)
        all_y.append(y)
    return np.mean(all_x, axis=0), np.mean(all_y, axis=0)

avg_x1, avg_y1 = process_images(image_paths1)
avg_x2, avg_y2 = process_images(image_paths2)

for i in range(16):
    print('%s - Image Set 1: x=%.02f, y=%.02f' % (joint_names[i], avg_x1[i], avg_y1[i]))
    print('%s - Image Set 2: x=%.02f, y=%.02f' % (joint_names[i], avg_x2[i], avg_y2[i]))

# Compute the Euclidean distance between average joint positions for the two sets of images
distances = np.sqrt((avg_x1 - avg_x2)**2 + (avg_y1 - avg_y2)**2)
print("\nAverage Euclidean distances between joint positions:")
for i in range(16):
    print('%s: %.02f' % (joint_names[i], distances[i]))

print("\nOverall average distance: %.02f" % np.mean(distances))


if np.mean(distances)<=22:
    print("This is the same person")
elif np.mean(distances)>22:
    print("This is a different person")
else:
    print("Calculation Error")