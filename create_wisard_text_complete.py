import os
from glob import glob  # You forgot this import
path_to_datasets = "/home/mattias/Documents/projects/Wisard_usecase/datasets"

for dataset in os.listdir(path_to_datasets):
    if dataset.startswith("dataset"):
        # TRAIN
        image_dir = os.path.join(path_to_datasets, dataset, "train/images/")
        output_txt = os.path.join(path_to_datasets, dataset, "train.txt")

        image_paths = sorted(
            glob(os.path.join(image_dir, "*.jpg")) +
            glob(os.path.join(image_dir, "*.jpeg"))
        )

        with open(output_txt, "w") as f:
            for path in image_paths:
                f.write(os.path.abspath(path) + "\n")

        print(f"Saved {len(image_paths)} image paths to {output_txt}")

        # VALID
        image_dir = os.path.join(path_to_datasets, dataset, "valid/images/")
        output_txt = os.path.join(path_to_datasets, dataset, "valid.txt")

        image_paths = sorted(
            glob(os.path.join(image_dir, "*.jpg")) +
            glob(os.path.join(image_dir, "*.jpeg"))
        )

        with open(output_txt, "w") as f:
            for path in image_paths:
                f.write(os.path.abspath(path) + "\n")

        print(f"Saved {len(image_paths)} image paths to {output_txt}")