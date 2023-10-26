import cv2
import os

def video_frame_slicer(video_path, output_folder, interval=0.5):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #print("Frames per second: {}".format(fps))

    # Calculate the frame step based on the desired interval
    frame_step = int(fps * interval)

    frame_count = 0
    while True:
        # Set the video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Save frame as an image
        frame_filename = os.path.join(output_folder, "frame_{:04d}.jpg".format(frame_count))
        cv2.imwrite(frame_filename, frame)
        #print("Saved {}".format(frame_filename))

        frame_count += frame_step

    # Release the video capture object
    cap.release()

    print("Video slicing completed.")

if __name__ == "__main__":
    video_path = "1.0.mp4"
    output_folder = "frames_output_folder"
    video_frame_slicer(video_path, output_folder)
