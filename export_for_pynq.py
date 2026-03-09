# export_for_pynq.py — Export 16×16 interpolated test data and labels to pickle
from wbcdataset import dataio
import numpy as np
import pickle

TARGET_SIZE = (16, 16)  # Interpolation target size
DATA_PICKLE = 'test_data_verified_16x16.pickle'
LABELS_PICKLE = 'test_labels_verified_16x16.pickle'

print("="*80)
print("Exporting 16×16 interpolated verified test data and labels (consistent with training dataio)")
print("="*80)

train_loader, val_loader, test_loader, type_count = dataio(
    folder_name='BTMG/',
    batch_size=32,
    shuffle_data=False,
    val_num_per_type=100,
    type_str='m-g-(b-t)',
    target_size=TARGET_SIZE,
)

print(f"\nExtracting 16×16 interpolated data from test_loader...")

all_data = []
all_labels = []

for batch_data, batch_labels in test_loader:
    all_data.append(batch_data.numpy())
    all_labels.append(batch_labels.numpy())

test_data = np.concatenate(all_data, axis=0)
test_labels = np.concatenate(all_labels, axis=0)

print(f"Test data:")
print(f"  Shape: {test_data.shape}  (N, C, {TARGET_SIZE[0]}, {TARGET_SIZE[1]})")
print(f"  Range: [{test_data.min():.4f}, {test_data.max():.4f}]")
print(f"  Mean: {test_data.mean():.4f}, Std: {test_data.std():.4f}")

print(f"\nLabel distribution (type_str='m-g-(b-t)', total {type_count} classes):")
label_names = ['monocyte', 'granulocyte', '(basophil+lymphocyte)']
for i in range(type_count):
    name = label_names[i] if i < len(label_names) else f"class_{i}"
    count = int(np.sum(test_labels == i))
    print(f"  {name:25s}: {count:3d} images")

with open(DATA_PICKLE, 'wb') as f:
    pickle.dump(test_data, f)
with open(LABELS_PICKLE, 'wb') as f:
    pickle.dump(test_labels, f)

print(f"\n✓ Export successful:")
print(f"  - {DATA_PICKLE} ({test_data.nbytes/1024**2:.1f} MB)")
print(f"  - {LABELS_PICKLE}")

print(f"\nUpload to Pynq:")
print(f"  scp {DATA_PICKLE} {LABELS_PICKLE} xilinx@<PYNQ_IP>:/home/xilinx/")
print("="*80)