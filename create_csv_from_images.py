from csv_utils import (
    create_csv,
    sort_csv,
    create_negative_pairs,
    create_negative_pairs_same_finger,
    create_positive_pairs,
    create_test_train_split,
)
import os

if __name__ == "__main__":
    csv_dir = "csv_dir"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    print(f"Created directory {csv_dir}")
    create_csv(csv_dir=csv_dir)
    sort_csv(csv_dir=csv_dir)
    # Functions below operate on created and sorted csv files
    create_negative_pairs(csv_dir=csv_dir)
    create_negative_pairs_same_finger(csv_dir=csv_dir)
    create_positive_pairs(csv_dir=csv_dir)
    # Test train split must be done last
    create_test_train_split(csv_dir=csv_dir)
    print("Successfully created csv files")
