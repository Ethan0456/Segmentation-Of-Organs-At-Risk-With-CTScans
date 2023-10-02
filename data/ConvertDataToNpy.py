import os
import numpy as np
import nibabel as nib
from patchify import patchify
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from patchify import patchify
from preprocess.Padding import Padding

class ConvertDataToNpy:
    def __init__(self):
        # Define your input and output directories
        data_dir = 'Data-Split'
        preprocessed_dir = 'preprocessed'

        # Ensure the output directories exist
        os.makedirs(os.path.join(preprocessed_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_dir, 'test'), exist_ok=True)

        # Process 'train' and 'test' directories
        for data_split in ['train', 'test']:
            data_split_dir = os.path.join(data_dir, data_split)
            preprocessed_split_dir = os.path.join(preprocessed_dir, data_split)

            # Create output directory for the current split
            os.makedirs(preprocessed_split_dir, exist_ok=True)

            # Iterate through volume-label pairs in the current split
            for volume_filename in os.listdir(data_split_dir):
                if volume_filename.startswith('volume-'):
                    volume_path = os.path.join(data_split_dir, volume_filename)
                    label_filename = f"labels-{volume_filename.split('-')[1]}"  # Corresponding label filename
                    label_path = os.path.join(data_split_dir, label_filename)

                    # Preprocess and save patches for the current volume-label pair
                    self.preprocess_and_save(volume_path, label_path, preprocessed_split_dir, volume_filename.split('-')[1])

        print("Preprocessing and saving completed.")

    
    def preprocess_and_save(self, volume_path, label_path, output_dir, volume_id):
            # Load the volume and label
        volume = nib.load(volume_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # First Padd Data
        padd = Padding()
        volume = padd.paddData(volume)
        label = padd.paddData(label)

        volume_patches = patchify(volume, (64, 64, 64), step=64)
        label_patches = patchify(label, (64, 64, 64), step=64)

        input_img = np.reshape(volume_patches, (-1, volume_patches.shape[3], volume_patches.shape[4], volume_patches.shape[5]))
        input_mask = np.reshape(label_patches, (-1, label_patches.shape[3], label_patches.shape[4], label_patches.shape[5]))

        # train_img = np.stack((input_img,)*3, axis=-1)
        # train_mask = np.expand_dims(input_mask, axis=4)

        # unique_labels = np.unique(train_mask)
        # train_mask_cat = to_categorical(train_mask, num_classes=6)

        # Create output directory for the current volume
        volume_output_dir = os.path.join(output_dir, f'volume-{volume_id}')
        os.makedirs(volume_output_dir, exist_ok=True)

        # Use the patch_generator to process and save patches one by one
        for idx, (volume_patch, label_patch) in enumerate(self.atch_generator(input_img, input_mask)):
            # Define the filenames for saving
            volume_filename = os.path.join(volume_output_dir, f'volume-{volume_id[:2]}-{idx}.npy')
            label_filename = os.path.join(volume_output_dir, f'label-{volume_id[:2]}-{idx}.npy')

            # Save the patches as numpy files
            np.save(volume_filename, volume_patch)
            np.save(label_filename, label_patch)

            # Explicitly delete variables to release memory
            del volume_patch
            del label_patch

        # Explicitly delete the volume and label variables
        del volume
        del label


    def patch_generator(self, volume, label):
        for i in range(0, volume.shape[0], 64):
            for j in range(0, volume.shape[1], 64):
                for k in range(0, volume.shape[2], 64):
                    volume_patch = volume[i:i+64, j:j+64, k:k+64]
                    label_patch = label[i:i+64, j:j+64, k:k+64]
                    yield volume_patch, label_patch
