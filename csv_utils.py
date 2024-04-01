import os
import csv
import numpy as np
import pandas as pd


def print_creation_info(info):
    print(f"Creating {info}...")


def print_done_info(filepath, n_rows=None):
    info = f"Done: {filepath}"
    if n_rows is not None:
        info = "".join([info, f", n_rows = {n_rows}\n"])
    else:
        info = "".join([info, "\n"])
    print(info)


def get_csv_name(folder_path):
    return folder_path.split("/")[1].lower().replace("-", "_")


SOCOFING_FOLDERS = [
    "/Real",
    "Altered/Altered-Easy",
    "Altered/Altered-Medium",
    "Altered/Altered-Hard",
]


def create_csv(
    socofing_folders=SOCOFING_FOLDERS,
    csv_dir="csv_dir",
):
    for folder in socofing_folders:
        csv_name = get_csv_name(folder)
        # CSV file path to save the results
        csv_file = f"{csv_dir}/{csv_name}.csv"

        # List to store the image file names
        image_files = []

        # Folder path containing the images
        folder_path = f"dataset/SOCOFing/{folder}"
        print(f"Reading: {folder_path}...")
        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            image_files.append(filename)

        # Write the file names to the CSV file
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow(["Image File"])
            for image_file in image_files:
                writer.writerow([image_file])
        print_done_info(csv_file, n_rows=len(image_files))


def create_negative_pairs(csv_dir="csv_dir", verbose=False):
    # read in real data
    csv_file_path = os.path.join(csv_dir, "real.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        real = list(reader)

    csv_file_path = os.path.join(csv_dir, "altered_easy.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        altered = list(reader)

    real = list(csv.reader(open(os.path.join(csv_dir, "real.csv"))))

    goal = 17931
    found = 0
    negatives = []
    print_creation_info("negative pairs")
    while found < goal:
        # get random real image
        full_filename = real[np.random.randint(0, len(real))][0]
        # filename_without_extension = full_filename.split(".")[0]
        # get random altered image
        if np.random.randint(0, 2) == 0:
            altered_full_filename = altered[np.random.randint(0, len(altered))][0]
        else:
            altered_full_filename = real[np.random.randint(0, len(real))][0]
        if full_filename in altered_full_filename:
            continue
        else:
            negatives.append([full_filename, altered_full_filename])
        found += 1
        if found % 1000 == 0:
            if verbose:
                print(found)

    df = pd.DataFrame(negatives)
    csv_file = os.path.join(csv_dir, "negatives.csv")
    df.to_csv(csv_file, index=False, header=False)
    print_done_info(csv_file, n_rows=len(negatives))


def search(real, altered, verbose=False):
    goal = 17931
    found = 0
    negatives = []
    print_creation_info("negative pairs of the same finger")
    while found < goal:
        search_start = np.random.randint(0, len(real) - 100)
        # get random real image
        full_filename = real[np.random.randint(0, len(real))][0]
        filename_without_extension = full_filename.split(".")[0]
        hand = filename_without_extension.split("_")[3]
        finger = filename_without_extension.split("_")[4]
        if np.random.randint(0, 2) == 0:
            search_in_real = True
        else:
            search_in_real = False
        for i in range(search_start, len(real)):
            # get random real or altered image that is the same finger
            if search_in_real:
                altered_full_filename = real[i][0]
            else:
                altered_full_filename = altered[i][0]

            if full_filename in altered_full_filename:
                continue

            if hand in altered_full_filename and finger in altered_full_filename:
                negatives.append([full_filename, altered_full_filename])
                break

        found += 1
        if found % 1000 == 0:
            if verbose:
                print(found)
    return negatives


def create_negative_pairs_same_finger(csv_dir="csv_dir"):
    output_name = "negatives_same_fingers.csv"

    # read in real data
    csv_file_path = os.path.join(csv_dir, "real.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        real = list(reader)

    csv_file_path = os.path.join(csv_dir, "altered_easy.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        altered = list(reader)

    real = list(csv.reader(open(os.path.join(csv_dir, "real.csv"))))

    negatives = search(real, altered)
    df = pd.DataFrame(negatives)
    csv_file = os.path.join(csv_dir, output_name)
    df.to_csv(csv_file, index=False, header=False)
    print_done_info(csv_file, n_rows=len(negatives))


def create_positive_pairs(csv_dir="csv_dir", verbose=False):
    # read in real data
    csv_file_path = os.path.join(csv_dir, "real.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        real = list(reader)

    csv_file_path = os.path.join(csv_dir, "altered_easy.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        altered = list(reader)

    positives = []
    print_creation_info("positive pairs")
    for i in range(len(real)):
        # full_filename = real[i][0]
        filename_without_extension = real[i][0].split(".")[0]
        original_class = int(filename_without_extension.split("__")[0])
        j = 0
        found = 0
        while found < 3:
            altered_full_filename = altered[j][0]
            altered_class = int(altered[j][0].split("__")[0])
            if original_class < altered_class:
                if verbose:
                    print("breaking")
                    print(original_class, altered_class)
                break
            if filename_without_extension in altered_full_filename:
                if verbose:
                    print("found", found)
                found += 1
                positives.append([real[i][0], altered[j][0]])
            j += 1

    # save to csv positives
    df = pd.DataFrame(positives)
    csv_file = os.path.join(csv_dir, "positives.csv")
    df.to_csv(csv_file, index=False, header=False)
    print_done_info(csv_file, n_rows=len(positives))


def create_test_train_split(csv_dir="csv_dir"):
    # read in real data
    csv_file_path = os.path.join(csv_dir, "positives.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        positives = list(reader)

    # csv_file_path = os.path.join(dir, 'negatives.csv')

    csv_file_path = os.path.join(csv_dir, "negatives_same_fingers.csv")
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        negatives = list(reader)

    # combine positives and negatives and add labels
    positives = [[x[0], x[1], 1] for x in positives]
    negatives = [[x[0], x[1], 0] for x in negatives]

    # shuffle
    data = positives + negatives
    np.random.shuffle(data)

    # split into train and test
    split = int(len(data) * 0.8)
    train = data[:split]
    test = data[split:]

    # write to csv
    df = pd.DataFrame(train)
    train_csv = os.path.join(csv_dir, "train.csv")
    df.to_csv(train_csv, index=False, header=False)
    print_done_info(train_csv, n_rows=len(train))

    df = pd.DataFrame(test)
    test_csv = os.path.join(csv_dir, "test.csv")
    df.to_csv(test_csv, index=False, header=False)
    print_done_info(test_csv, n_rows=len(test))


def sort_csv(csv_dir="csv_dir", socofing_folders=SOCOFING_FOLDERS):
    for folder in socofing_folders:
        csv_name = get_csv_name(folder)
        # CSV file path to save the results
        csv_file = f"{csv_dir}/{csv_name}.csv"
        print(f"Sorting: {csv_file}")
        with open(csv_file, "r") as sample:
            csv1 = csv.reader(sample, delimiter="\n")
            data = list(csv1)
            # sort data by the number in the filename
            sort = sorted(data, key=lambda x: int(x[0].split("__")[0]))
        with open(csv_file, "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerows(sort)  # write the sorted rows
        print_done_info(csv_file)
