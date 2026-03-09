import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np


def dataio(folder_name, batch_size, shuffle_data=True, val_num_per_type=100, type_str='b-t', target_size=None):
    dataset_train_val = WBCDataSet(train=True, folder_name=folder_name, type_str=type_str)
    dataset_test = WBCDataSet(train=False, folder_name=folder_name, type_str=type_str)

    train_indices, train_sampling_weight, val_indices = split_train_and_val(dataset_train_val.index_for_each_type, val_num_per_type, shuffle_data=shuffle_data)

    # ============================================================================
    # Data Normalization (Critical for fpga deployment!)
    # ============================================================================
    # 1. Convert to float32 (IMPORTANT: int8 / 255 = 0)
    train_val_data = dataset_train_val.dataset.astype('float32')
    test_data = dataset_test.dataset.astype('float32')
    
    # 2. Normalize to [0, 1] if data is in [0, 255] range
    if train_val_data.max() > 1.0:
        print(f"[Normalization] Original data range: [{train_val_data.min():.2f}, {train_val_data.max():.2f}]")
        train_val_data = train_val_data / 255.0
        test_data = test_data / 255.0
        print(f"[Normalization] After /255: [{train_val_data.min():.4f}, {train_val_data.max():.4f}]")
    
    # 3. Calculate mean and std from TRAINING set only
    train_data = train_val_data[train_indices, ...]
    data_mean = np.mean(train_data, axis=(0, 2, 3), keepdims=True)  # [1, C, 1, 1]
    data_std = np.std(train_data, axis=(0, 2, 3), keepdims=True)    # [1, C, 1, 1]
    
    print(f"[Normalization] Training set statistics:")
    print(f"  Mean: {data_mean.flatten()}")
    print(f"  Std:  {data_std.flatten()}")
    
    # 4. Apply standardization: (x - mean) / std
    train_val_data = (train_val_data - data_mean) / (data_std + 1e-7)
    test_data = (test_data - data_mean) / (data_std + 1e-7)
    
    print(f"[Normalization] After standardization: [{train_val_data.min():.4f}, {train_val_data.max():.4f}]")
    
    # Save normalization parameters for hls4ml deployment
    np.save('normalization_params.npy', {'mean': data_mean, 'std': data_std})
    print(f"[Normalization] Parameters saved to 'normalization_params.npy' for deployment")
    print("="*80)
    # ============================================================================

    # Build transforms list (no normalization needed here, already done above)
    transform_list = []
    if target_size is not None:
        # Use BICUBIC (antialias=True) for high-quality downsampling (BICUBIC supports Tensor, LANCZOS only PIL)
        transform_list.append(transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    transform_list.append(transforms.RandomAffine(degrees=180, translate=(0.01, 0.01)))
    
    data_transforms = transforms.Compose(transform_list)
    
    # Validation/test transforms (no augmentation, only resize)
    val_transform_list = []
    if target_size is not None:
        # Use BICUBIC (antialias=True) for high-quality downsampling (BICUBIC supports Tensor, LANCZOS only PIL)
        val_transform_list.append(transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    val_transforms = transforms.Compose(val_transform_list) if val_transform_list else None
    
    # Create datasets with normalized data
    dataset_train = CustomDataset(train_val_data[train_indices, ...], dataset_train_val.targetset[train_indices], data_transforms)
    dataset_val = CustomDataset(train_val_data[val_indices, ...], dataset_train_val.targetset[val_indices], val_transforms)
    dataset_test = CustomDataset(test_data, dataset_test.targetset, val_transforms)

    train_sampler = WeightedRandomSampler(train_sampling_weight, len(train_sampling_weight))
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset_train_val.type_count
    
def split_train_and_val(index_for_each_type, val_num_per_type, shuffle_data=True):
    train_indices = []
    train_sampling_weight = []
    val_indices = []
    for i in range(len(index_for_each_type)):
        index_type_i = index_for_each_type[str(i)]
        if shuffle_data:
            np.random.shuffle(index_type_i)
        index_type_i_for_train = index_type_i[val_num_per_type:, 0]
        train_indices = np.concatenate((train_indices, index_type_i_for_train))
        train_sampling_weight = np.concatenate((train_sampling_weight, [1 / len(index_type_i_for_train)]*len(index_type_i_for_train)))
        val_indices = np.concatenate((val_indices, index_type_i[:val_num_per_type, 0]))
    train_indices = train_indices.astype(int)
    val_indices = val_indices.astype(int)
    return train_indices, train_sampling_weight, val_indices

    
    
    
class WBCDataSet(Dataset):
    def __init__(self, train, folder_name, type_str='b-t-m-g'):
        self.dataset, self.targetset = self.load_data_from_pickle(train, folder_name)
        if self.dataset.shape[1] == 2:
            self.dataset = self.dataset[:, 0, ...]
            self.dataset = self.dataset[:, np.newaxis, :, :]
            self.targetset = self.targetset + 1 # for direct_cut dataset
        self.dataset, self.targetset = self.select_type(type_str, self.dataset, self.targetset)
        self.index_for_each_type = self.get_index_for_each_type(self.targetset)
        self.type_count = np.max(self.targetset) + 1
        self.img_width = self.dataset.shape[1]
        self.img_height=self.dataset.shape[2]
       
    def load_data_from_pickle(self, train, file):

        if train: 
            with open(file + 'train_data_set.pickle', 'rb') as input1:
                dataset = pickle.load(input1)
            with open(file + 'train_target_index.pickle', 'rb') as input2:
                targetset = pickle.load(input2)

        else:
            with open(file + 'test_data_set.pickle', 'rb') as input3:
                dataset = pickle.load(input3)
            with open(file + 'test_target_set.pickle', 'rb') as input4:
                targetset = pickle.load(input4)

        targetset = np.array(targetset)
        targetset = targetset - 1

        dataset = np.transpose(dataset, (0, 3, 1, 2))

        return dataset, targetset

    def decode_select_type_str(self, type_str: str, label_set):
        type_directory= {'b': 0, 't': 1, 'm': 2, 'g': 3, 'cd4': 4, 'cd8': 5}
        type_str_list = re.findall(r'[a-zA-Z]+\d*', type_str)
        type_idx_list = [type_directory[type_name] for type_name in type_str_list]
        target_idx_list = self.string_to_position_dict(type_str)
        chooser = np.array([0] * len(label_set))
        label = np.array([0] * len(label_set))
        for i, type_number in enumerate(type_idx_list):
            chooser[label_set == type_number] = 1
            label[label_set == type_number] = target_idx_list[i]
        return chooser > 0, label

    
    def select_type(self, type_str, image_set, label_set):
        chooser, label_set = self.decode_select_type_str(type_str, label_set)
        image_set = image_set[chooser]
        label_set = label_set[chooser]
        return image_set, label_set

    def get_index_for_each_type(self, label_set):
        type_count = np.max(label_set) + 1
        index_for_each_type = {}
        for i in range(type_count):
            index_for_each_type[str(i)] = np.argwhere(label_set == i)
        return index_for_each_type
    
    def string_to_position_dict(slef, type_str):
        elements = type_str.split('-')  # Split the string by dashes
        output = []
        count = 0

        for elem in elements:
            # Check for grouped elements (enclosed in parentheses)
            if '(' in elem:
                output.append(count)
            else:
                output.append(count)
                count += 1

        return output
        
        

    def __len__(self):
        return len(self.targetset)

    def __getitem__(self, index):
        image = self.dataset[index, ...]
        image = torch.tensor(image)
        target = self.targetset[index]
        return image, target
        

        
class CustomDataset(Dataset):
    def __init__(self, data, target, transforms=None):
        """
        Args:
            data: numpy array, already normalized (float32, standardized)
            target: numpy array of labels
            transforms: torchvision transforms (resize, augmentation)
        """
        self.data = data
        self.target = target
        self.transforms = transforms

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # Data is already normalized in dataio: float32, /255, (x-mean)/std
        image = self.data[index, ...]
        target = self.target[index]
        
        # Convert to tensor (data is already float32)
        image = torch.tensor(image, dtype=torch.float32)
        
        # Apply transforms (resize, augmentation)
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target
        
if __name__ == '__main__':
    # dataio returns 4 values: train_loader, val_loader, test_loader, type_count
    train_loader, val_loader, test_loader, type_count = dataio('/home/ye/classification-master/BTMG/', 32, shuffle_data=True)
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {type_count}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    feature, target = next(iter(train_loader))
    print(f"\nBatch shape: {feature.shape}")
    print(f"Data range: [{feature.min():.4f}, {feature.max():.4f}]")
    print(f"Target shape: {target.shape}")
    
    # Show the first image
    img = feature[0].numpy()
    if img.shape[0] == 1:  # Single channel
        plt.imshow(img[0], cmap='gray')
    else:
        plt.imshow(img.transpose(1, 2, 0))
    plt.title(f"Normalized image (class: {target[0].item()})")
    plt.colorbar()
    plt.show()
     # Save image instead of displaying
    plt.savefig('normalized_sample.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Sample image saved to 'normalized_sample.png'")
