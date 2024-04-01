import csv
import os
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
from csv_utils import SOCOFING_FOLDERS, get_csv_name, print_done_info


def crop(image):
    border_left = 2
    border_top = 2
    border_right = 4
    border_bottom = 4

    width, height = image.size
    crop_left = border_left
    crop_top = border_top
    crop_right = width - border_right
    crop_bottom = height - border_bottom

    cropped_image = F.crop(
        image, crop_top, crop_left, crop_bottom - crop_top, crop_right - crop_left
    )
    return cropped_image


dataset_dir = "dataset"
csv_dir = "csv_dir"
for folder in SOCOFING_FOLDERS:
    real_path = os.path.join(dataset_dir, f"SOCOFing/{folder}")

    save_dir = f"preprocessed/SOCOFing/{folder}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_file = get_csv_name(folder)
    reader = csv.reader(open(os.path.join(csv_dir, f"{csv_file}.csv"), "r"))
    print(f"Preprocessing: {real_path}")
    for row in reader:
        image_path = os.path.join(real_path, row[0])
        image = Image.open(image_path)
        image = image.convert("L")
        image = crop(image)
        image.save(os.path.join(save_dir, row[0]))
    print_done_info(save_dir)
