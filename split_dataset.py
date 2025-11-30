import os, shutil
from sklearn.model_selection import train_test_split

#old directory
data_dir = 'C:\\Users\\Viviktha\\Desktop\\Projects\\New-ASL\\asl_dataset'

#new directories
base_dir = 'C:\\Users\\Viviktha\\Desktop\\Projects\\New-ASL\\asl_split'
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    train_files, test_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.33, random_state=42)

    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_class_dir = os.path.join(base_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(split_class_dir, f))