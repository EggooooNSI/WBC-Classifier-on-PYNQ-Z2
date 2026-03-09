# WBC Classifier on FPGA

FPGA-accelerated White Blood Cell (WBC) classifier using Quantitative Phase Imaging (QPI) data, deployed on Pynq Z2 board with Tensil AI framework.

## Overview

This project implements a ResNet10-based classifier for three types of white blood cells:
- Monocyte
- Granulocyte  
- Lymphocyte


## Repository Structure

```
WBC_classifier_on_FPGA/
├── main.py                    # Training and evaluation entry point
├── resnet_torch.py            # ResNet10 model architecture
├── wbcdataset.py              # Data loading and preprocessing
├── export_for_pynq.py         # Export preprocessed data for FPGA
├── export_32ch_onnx.py        # Export trained model to ONNX format
├── tester.py                  # Model testing utilities
├── utils.py                   # Helper functions
├── config.py                  # Configuration settings
└── PYNQ-Z2/                   # Pre-compiled files for FPGA deployment
    ├── *.tmodel                # Tensil model files (16ch & 32ch)
    ├── *.tprog                 # Tensil program files
    ├── *.tdata                 # Tensil weight data files
    ├── test_data_verified_32x32.pickle   # Test data (preprocessed)
    ├── test_labels_verified_32x32.pickle # Test labels
    └── Jupyter/
        └── tensil.ipynb        # Inference demo notebook
```

## Two Ways to Use This Project

### Option 1: Direct Deployment (Recommended for Quick Start)

Use pre-compiled Tensil AI files without retraining or recompiling.

#### Steps:

1. **Prepare your Pynq Z2 board**
   - Boot the Pynq Z2 board with SD card
   - Ensure Jupyter Notebook server is running
   - Connect to the board via network

2. **Upload files to the board**
   
   Copy all files from `PYNQ-Z2/` directory to the board:
   
   ```bash
   # Copy model files to /home/xilinx/
   scp PYNQ-Z2/*.tmodel xilinx@pynq-board:/home/xilinx/
   scp PYNQ-Z2/*.tprog xilinx@pynq-board:/home/xilinx/
   scp PYNQ-Z2/*.tdata xilinx@pynq-board:/home/xilinx/
   scp PYNQ-Z2/*.pickle xilinx@pynq-board:/home/xilinx/
   
   # Copy Jupyter notebook to Jupyter directory (IMPORTANT!)
   scp PYNQ-Z2/Jupyter/tensil.ipynb xilinx@pynq-board:/home/xilinx/jupyter_notebooks/
   ```
   
   **⚠️ Important**: Place the `tensil.ipynb` file in the Jupyter notebooks directory (`/home/xilinx/jupyter_notebooks/`), NOT in the home directory root.

3. **Run inference**
   
   - Open Jupyter Notebook in your browser (typically `http://pynq-board:9090`)
   - Navigate to `tensil.ipynb`
   - Run all cells to perform inference
   
   The notebook will load the model and test data, then run inference on 300 test samples.

#### Available Models:

- **16-channel model** (faster):
  - Files: `resnet10_tensil_fp32_16ch_32x32_onnx_pynqz1.*`
  
- **32-channel model** (more accurate):
  - Files: `resnet10_new_32ch_fp32_16x16_onnx_pynqz1.*`

---

### Option 2: Custom Training and Export

Train your own model from scratch and compile it for FPGA.

#### Prerequisites:

- Python 3.8+
- PyTorch 1.10+
- ONNX
- Tensil AI toolchain ([installation guide](https://github.com/tensil-ai/tensil))

#### Steps:

##### Step 1: Prepare Data

Ensure your training data is in the correct format:
- Training data: `BTMG/train_data_set.pickle` (phase images)
- Training labels: `BTMG/train_label_set.pickle`
- Test data: Preprocessed test set

##### Step 2: Modify Model Architecture (Optional)

Edit `resnet_torch.py` to customize the ResNet10 architecture:

```python
# Adjust network width (initial channels)
INITIAL_CHANNELS = 16  # or 32 for higher accuracy
```

Edit `main.py` to configure training:

```python
# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 8e-5
NUM_EPOCHS = 50
INPUT_SIZE = 32  # or 16x16
```

##### Step 3: Train the Model

```bash
python main.py --mode train
```

This will:
- Load and preprocess data with bicubic resizing and standardization
- Train ResNet10 with weighted sampling for class balance
- Save checkpoints to `checkpoint/` directory
- Output: `*.pth` checkpoint files

##### Step 4: Export Data for FPGA

```bash
python export_for_pynq.py
```

This generates:
- `test_data_verified_32x32.pickle`: Preprocessed test images (standardized)
- `test_labels_verified_32x32.pickle`: Test labels

**⚠️ Important**: These files must use the SAME preprocessing (mean/std) as training data.

##### Step 5: Export Model to ONNX

```bash
python export_32ch_onnx.py
```

This will:
- Load the latest checkpoint from `checkpoint/` directory
- Export the model to ONNX format (opset 10)
- Output: `resnet10_32ch_fp32_16x16.onnx` (or similar)

**Note**: Edit the script to match your model configuration:
```python
INITIAL_CHANNELS = 16  # Must match your trained model
INPUT_SIZE = (1, 1, 16, 16)  # (batch, channel, height, width)
CHECKPOINT_DIR = 'checkpoint/your_experiment_name'
```

##### Step 6: Compile ONNX to Tensil Format

Use Tensil AI compiler to convert ONNX to FPGA-ready format:

```bash
# Install Tensil AI (if not already installed)
# See: https://www.tensil.ai/docs/installation/

# Compile for Pynq Z1/Z2 architecture
tensil compile \
  -a /path/to/pynqz1.tarch \
  -m resnet10_32ch_fp32_16x16.onnx \
  -o resnet10_32ch_fp32_16x16_onnx_pynqz1
```

This generates three files:
- `.tmodel`: Model architecture metadata
- `.tprog`: Compiled instruction stream
- `.tdata`: Model weights

##### Step 7: Deploy to Pynq Board

Follow the same upload steps as Option 1:

1. Copy `*.tmodel`, `*.tprog`, `*.tdata` to `/home/xilinx/`
2. Copy test data pickles to `/home/xilinx/`
3. Copy `Jupyter/tensil.ipynb` to `/home/xilinx/jupyter_notebooks/`
4. Run the notebook

---

## Model Architecture

**ResNet10** with configurable network width:

- **Input**: 32×32 grayscale QPI phase images (single channel)
- **Preprocessing**: Bicubic resize → Z-score standardization
- **Architecture**: 4 residual blocks with decreasing spatial dimensions
- **Output**: 3-class logits (softmax classification)

**Network Width Options:**
- 16 channels: [16, 32, 64, 128] → Faster inference
- 32 channels: [32, 64, 128, 256] → Higher accuracy

## Data Preprocessing

All images undergo standardization using training set statistics:

```python
# Mean and std calculated from training data
mean = 0.0584
std = 0.4915

# Standardization formula
normalized_image = (image - mean) / std
```

**Input format for FPGA:**
- Shape: `[batch, 1, 32, 32]` (NCHW format)
- Data type: `float32`
- Preprocessed: Already standardized

## FPGA Deployment Details

**Hardware:** Pynq Z2 (Xilinx Zynq-7000 series, Artix-7 FPGA)

**Tensil TCU Architecture (Pynq Z1/Z2):**
- Array size: 8
- Data type: FP16BP8 (16-bit floating point, 8-bit bias)
- Fixed resource utilization: 30% BRAM, 33% DSP

**Performance Trade-offs:**
- 16-channel model: Lower latency, higher throughput
- 32-channel model: Better quantization tolerance, higher accuracy
- Both models use the SAME fixed FPGA resources (Tensil's TPU-style architecture)

## Requirements

### For Training (Option 2):
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
scikit-learn>=0.24.0
onnx>=1.10.0
opencv-python>=4.5.0
```

### For FPGA Deployment (Both Options):
- Pynq Z2 board with Pynq 2.6+ image
- Tensil AI overlay bitstream
- Python packages on board: `numpy`, `pynq`, `tcu_pynq`

## Quick Start

**Option 1 (Recommended):**
```bash
# 1. Copy files to Pynq board
scp -r PYNQ-Z2/* xilinx@pynq-board:/home/xilinx/

# 2. SSH to board and move notebook
ssh xilinx@pynq-board
mv /home/xilinx/Jupyter/tensil.ipynb /home/xilinx/jupyter_notebooks/

# 3. Open browser and run notebook
# http://pynq-board:9090
```

**Option 2 (Custom Training):**
```bash
# 1. Train model
python main.py --mode train

# 2. Export data and ONNX
python export_for_pynq.py
python export_32ch_onnx.py

# 3. Compile with Tensil AI
tensil compile -a pynqz1.tarch -m your_model.onnx -o output_name

# 4. Deploy (same as Option 1)
```

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'tcu_pynq'"**  
A: Ensure Tensil AI driver is installed on Pynq board. Follow [Tensil installation guide](https://www.tensil.ai/docs/tutorials/pynq/).

**Q: "std::bad_alloc" or memory errors on Pynq**  
A: Increase `dma_buffer_size` in notebook:
```python
tcu = Driver(pynqz1, overlay.axi_dma_0, dma_buffer_size=512*1024)
```

**Q: Accuracy is very low after deployment**  
A: Check that:
1. Test data uses the SAME mean/std as training data
2. Input size matches model compilation (16×16 or 32×32)
3. Data is correctly standardized before inference

**Q: How to switch between 16ch and 32ch models?**  
A: In `tensil.ipynb`, change the `MODEL_NAME` variable:
```python
# For 16-channel model:
MODEL_NAME = 'resnet10_new_tensil_fp32_16ch_32x32_onnx_pynqz1'

# For 32-channel model:
MODEL_NAME = 'resnet10_32ch_fp32_16x16_onnx_pynqz1'
```

## Acknowledgments

This project uses:
- [Tensil AI](https://www.tensil.ai/) - Neural network compiler for FPGAs
- [Pynq](http://www.pynq.io/) - Python framework for Zynq devices
- [PyTorch](https://pytorch.org/) - Deep learning framework
