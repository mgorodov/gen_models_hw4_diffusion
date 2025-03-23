import os
import random
import zipfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re
from PIL import Image


class CelebADataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data/celeba",
        transform: transforms.Compose | None = None,
        subset_size: int | None = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.subset_size = subset_size

        self.header = None

        self.dataset_folder = os.path.join(root_dir, "images")

        self.filenames = os.listdir(self.dataset_folder)
        if self.subset_size is not None:
            self.filenames = random.sample(self.filenames, self.subset_size)
        self.filenames = sorted(self.filenames)


        attr_file_path = os.path.join(root_dir, "list_attr_celeba.txt")
        if not os.path.isfile(attr_file_path):
            raise FileNotFoundError(
                f"Could not find 'list_attr_celeba.txt' in the annotations folder."
            )

        # Load attributes
        self.annotations = []
        with open(attr_file_path, "r") as f:
            lines = f.read().splitlines()

        # First line has the number of images, second line has the attribute names
        # The rest lines each correspond to one image
        for i, line in enumerate(lines):
            # line might have variable spaces, so split robustly
            line = re.sub(r"\s+", " ", line.strip())
            if i == 0:
                continue  # number of images
            elif i == 1:
                # header line with attribute names
                self.header = line.split(" ")
            else:
                parts = line.split(" ")
                filename = parts[0]
                attr_vals = [
                    int(attr == '1')
                    for attr in parts[1:]
                ]
                self.annotations.append((filename, attr_vals))

        # Convert to a dict: filename -> attribute array
        # so we can quickly look up attributes by filename
        self.attr_map = {
            fn: torch.tensor(attr_vals, dtype=torch.long)
            for fn, attr_vals in self.annotations
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)

        # Load the image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        # Fetch attributes (if they exist in the attr_map; some extra files might appear)
        attributes = self.attr_map.get(img_name, torch.zeros(len(self.header), dtype=torch.long))

        return img, attributes
