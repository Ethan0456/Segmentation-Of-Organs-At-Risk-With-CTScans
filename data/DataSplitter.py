import os
import random
from shutil import copyfile

class DataSplit:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def split(self):
        # Create list of all files in data_dir which has volume- and labels- as prefix
        volume_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('volume-')])
        label_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('labels-')])

        # Shuffle the indices of the files for random splitting
        indices = list(range(len(volume_files)))
        random.shuffle(indices)

        # Calculate the split point
        split_point = int(0.75 * len(indices))
        # Split the data into training and testing
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        # Create subdirectories for training and testing data
        train_dir = os.path.join(self.output_dir, 'train')
        test_dir = os.path.join(self.output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy the selected data to the appropriate directories
        for idx in train_indices:
            volume_file = volume_files[idx]
            label_file = label_files[idx]
            copyfile(os.path.join(self.data_dir, volume_file), os.path.join(train_dir, volume_file))
            copyfile(os.path.join(self.data_dir, label_file), os.path.join(train_dir, label_file))

        for idx in test_indices:
            volume_file = volume_files[idx]
            label_file = label_files[idx]
            copyfile(os.path.join(self.data_dir, volume_file), os.path.join(test_dir, volume_file))
            copyfile(os.path.join(self.data_dir, label_file), os.path.join(test_dir, label_file))