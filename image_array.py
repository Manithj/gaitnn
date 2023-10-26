import os

def get_image_filepaths_from_folder(folder):
    image_filepaths = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # You can add more extensions if needed
            relative_path = os.path.join(folder, filename).replace('\\', '/')
            image_filepaths.append(relative_path)
    return image_filepaths

# folder_path = './path_to_your_folder'  # Replace with your relative folder path
# image_filepaths_array = get_image_filepaths_from_folder(folder_path)

# print("Image filepaths:", image_filepaths_array)
