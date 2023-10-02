import os
import numpy as np
import nibabel as nib
import patchify
from tensorflow.keras.utils import Sequence
from preprocess.Padding import Padding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class CustomImageDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=1, n_classes=6):
        self.n_classes = n_classes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith('volume-')])
        self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('labels-')])
        self.indices = np.arange(len(self.volume_files))
        self.patch_size = 64
        self.current_epoch = 0

    def __len__(self):
        return int(np.ceil(len(self.volume_files) * self.patch_count() / self.batch_size))

    def __getitem__(self, index):
        batch_volumes = []
        batch_labels = []

        for _ in range(self.batch_size):
            volume_file, label_file = self.get_next_patch_files()

            volume_path = os.path.join(self.data_dir, volume_file)
            label_path = os.path.join(self.data_dir, label_file)

            volume = nib.load(volume_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            padd = Padding()
            volume = padd.paddData(volume)
            label = padd.paddData(label)

            img_patches = patchify(volume, (self.patch_size, self.patch_size, self.patch_size), step=64)
            mask_patches = patchify(label, (self.patch_size, self.patch_size, self.patch_size), step=64)

            input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
            input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))

            train_img = np.stack((input_img,)*3, axis=-1)
            train_mask = np.expand_dims(input_mask, axis=4)

            unique_labels = np.unique(train_mask)
            train_mask_cat = to_categorical(train_mask, num_classes=self.n_classes)

            batch_volumes.append(train_img)
            batch_labels.append(train_mask_cat)

        return np.array(batch_volumes), np.array(batch_labels)

    def on_epoch_end(self):
        self.current_epoch += 1
        np.random.shuffle(self.indices)

    def patch_count(self):
        # Calculate the total number of patches in the dataset
        count = 0
        for idx in self.indices:
            volume_file = self.volume_files[idx]
            label_file = self.label_files[idx]

            volume_path = os.path.join(self.data_dir, volume_file)
            label_path = os.path.join(self.data_dir, label_file)

            volume = nib.load(volume_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            padd = Padding()
            volume = padd.paddData(volume)
            label = padd.paddData(label)

            img_patches = patchify(volume, (self.patch_size, self.patch_size, self.patch_size), step=64)
            count += img_patches.shape[0]

        return count

    def get_next_patch_files(self):
        idx = self.indices[self.current_epoch % len(self.indices)]
        self.current_epoch += 1
        return self.volume_files[idx], self.label_files[idx]
