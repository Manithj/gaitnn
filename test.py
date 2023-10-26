import numpy as np
import matplotlib as mpl
from scipy.misc import imresize, imread
from human_pose_nn import HumanPoseIRNetwork
import image_array as imgarr
import video_slicer as vs
import tkinter as tk
from tkinter import filedialog, messagebox

# Configuration
mpl.use('Agg')
MODEL_PATH = './models/MPII+LSP.ckpt'
JOINT_NAMES = [
    'right ankle', 'right knee', 'right hip', 'left hip', 'left knee',
    'left ankle', 'pelvis', 'thorax', 'upper neck', 'head top',
    'right wrist', 'right elbow', 'right shoulder', 'left shoulder',
    'left elbow', 'left wrist'
]

# Initialize the pose estimation network
net_pose = HumanPoseIRNetwork()
net_pose.restore(MODEL_PATH)

def slice_video_to_frames(video_path, output_folder):
    vs.video_frame_slicer(video_path, output_folder)
    return imgarr.get_image_filepaths_from_folder('./{}'.format(output_folder))

def process_images(image_paths):
    all_x, all_y = [], []
    for image_path in image_paths:
        img = imread(image_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = imresize(img, [299, 299])
        img_batch = np.expand_dims(img, 0)
        y, x, _ = net_pose.estimate_joints(img_batch)
        all_x.append(np.squeeze(x))
        all_y.append(np.squeeze(y))
    return np.mean(all_x, axis=0), np.mean(all_y, axis=0)

def print_joint_positions(joint_names, avg_x, avg_y, label):
    for i, name in enumerate(joint_names):
        print('{} - {}: x={:.02f}, y={:.02f}'.format(name, label, avg_x[i], avg_y[i]))

def main(video1_path, video2_path, output_folder1, output_folder2):
    image_paths1 = slice_video_to_frames(video1_path, output_folder1)
    image_paths2 = slice_video_to_frames(video2_path, output_folder2)

    avg_x1, avg_y1 = process_images(image_paths1)
    avg_x2, avg_y2 = process_images(image_paths2)

    print_joint_positions(JOINT_NAMES, avg_x1, avg_y1, "Image Set 1")
    print_joint_positions(JOINT_NAMES, avg_x2, avg_y2, "Image Set 2")

    distances = np.sqrt((avg_x1 - avg_x2)**2 + (avg_y1 - avg_y2)**2)
    print("\nAverage Euclidean distances between joint positions:")
    for i, name in enumerate(JOINT_NAMES):
        print('{}: {:.02f}'.format(name, distances[i]))

    overall_avg_distance = np.mean(distances)
    print("\nOverall average distance: {:.02f}".format(overall_avg_distance))

    result = ""
    if overall_avg_distance <= 22:   #change this 
        result = "This is the same person"
    elif overall_avg_distance > 22:
        result = "This is a different person"
    else:
        result = "Calculation Error"
    

    message = "Overall average distance: {:.2f}\n{}".format(overall_avg_distance, result)
    messagebox.showinfo("Result", message)

def browse_file(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filepath)

def execute():
    video1_path = video1_entry.get()
    video2_path = video2_entry.get()
    main(video1_path, video2_path, "output_folder1", "output_folder2")

app = tk.Tk()
app.title("Gait Estimation by Induwara")

video1_label = tk.Label(app, text="Select Video 1:")
video1_label.pack(pady=10)
video1_entry = tk.Entry(app, width=50)
video1_entry.pack(pady=10)
video1_browse = tk.Button(app, text="Browse", command=lambda: browse_file(video1_entry))
video1_browse.pack(pady=10)

video2_label = tk.Label(app, text="Select Video 2:")
video2_label.pack(pady=10)
video2_entry = tk.Entry(app, width=50)
video2_entry.pack(pady=10)
video2_browse = tk.Button(app, text="Browse", command=lambda: browse_file(video2_entry))
video2_browse.pack(pady=10)

execute_button = tk.Button(app, text="Execute", command=execute)
execute_button.pack(pady=20)

app.mainloop()
