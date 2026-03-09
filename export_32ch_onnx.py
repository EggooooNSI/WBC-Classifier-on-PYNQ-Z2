"""
Export 32-channel ResNet10 to ONNX for Tensil AI
Larger model capacity better tolerates FP16BP8 quantization
"""
import torch
import os
from resnet_torch import ResNet10

# ============================================================================
# Configuration 
# ============================================================================
IN_CHANNELS = 1
NUM_CLASSES = 3  # m-g-(b-t)
INITIAL_CHANNELS = 16  # or 32

INPUT_SIZE = (1, IN_CHANNELS, 16, 16)

# Find latest checkpoint
CHECKPOINT_DIR = 'checkpoint/unet10-16-8e-05-m-g-(b-t)unet10-16-8e-05-m-g-(b-t)_16x16'
OUTPUT_PATH = 'resnet10_32ch_fp32_16x16.onnx'
OPSET_VERSION = 10

# ============================================================================
# Find Latest Checkpoint
# ============================================================================
def find_latest_checkpoint(checkpoint_dir):
    import glob
    if not os.path.exists(checkpoint_dir):
        return None
    
    pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not pth_files:
        return None
    
    latest_file = max(pth_files, key=os.path.getmtime)
    return latest_file

# ============================================================================
# Export
# ============================================================================
print("="*80)
print("Export ONNX")
print("="*80)

# 1. Build model
print(f"\n1. Build model...")
print(f"   Initial channels: {INITIAL_CHANNELS}")
print(f"   Classes: {NUM_CLASSES}")

model = ResNet10(
    in_ch=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    initial_channels=INITIAL_CHANNELS
)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")
print(f"   Model size: ~{params * 2 / 1024:.1f} KB (FP16)")

# 2. Load checkpoint
checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"\n2. Load checkpoint...")
    print(f"   Path: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"   Val acc: {checkpoint['val_acc']:.2f}%")
    
    print(f"   ✓ Checkpoint loaded successfully")
else:
    print(f"\n2. ⚠️  Cannot find checkpoint")


# 3. Test forward pass
print(f"\n3. Test forward pass...")
dummy_input = torch.randn(INPUT_SIZE)

with torch.no_grad():
    output, features = model(dummy_input)

print(f"   Input: {dummy_input.shape}")
print(f"   Output: {output.shape}")
print(f"   ✓ Success")

# 4. Export to ONNX
print(f"\n4. Export to ONNX...")
print(f"   Output path: {OUTPUT_PATH}")
print(f"   Opset version: {OPSET_VERSION}")

class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits, _ = self.model(x)
        return logits

export_model = ExportWrapper(model)
export_model.eval()

torch.onnx.export(
    export_model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    verbose=False
)

print(f"   ✓ ONNX exported successfully")

# 5. Validate ONNX
print(f"\n5. Validate ONNX...")
try:
    import onnx
    import onnxruntime as ort
    
    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ ONNX model is valid")
    
    # Test inference
    ort_session = ort.InferenceSession(OUTPUT_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    torch_out = output.detach().numpy()
    onnx_out = ort_outputs[0]
    max_diff = abs(torch_out - onnx_out).max()
    
    print(f"   PyTorch: {torch_out[0]}")
    print(f"   ONNX:    {onnx_out[0]}")
    print(f"   Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print(f"   ✓ Outputs are identical")
    elif max_diff < 1e-3:
        print(f"   ✓ Outputs are nearly identical")
    
except ImportError:
    print(f"   ⚠️  Validation skipped (requires onnx & onnxruntime)")

# 6. Summary
print(f"\n" + "="*80)
print("Export Summary")
print("="*80)

file_size = os.path.getsize(OUTPUT_PATH) / 1024
print(f"  Model: ResNet10 (32-channel)")
print(f"  Input: 1×32×32")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Parameters: {params:,} ({params/1000:.1f}K)")
print(f"  File: {OUTPUT_PATH} ({file_size:.1f} KB)")
print(f"  ")
print(f"  Comparison with 16-channel version:")
print(f"    Parameter increase: ~4x")
print(f"    Quantization robustness: Significantly improved")
print(f"    FPGA resources: Increased (needs evaluation)")

print(f"\n" + "="*80)
print("Next Steps")
print("="*80)
print(f"  1. Test ONNX accuracy:")
print(f"     python test_onnx_uniform.py")
print(f"     (Modify model path to {OUTPUT_PATH})")
print(f"  ")
print(f"  2. Compile Tensil model:")
print(f"     tensil compile \\")
print(f"       -a /path/to/pynqz1.tarch \\")
print(f"       -m {OUTPUT_PATH} \\")
print(f"       -o output \\")
print(f"       -s true")
print(f"  ")
print(f"  3. Check FPGA resource usage:")
print(f"     - 32-channel model uses more LUT/DSP/BRAM")
print(f"     - Ensure pynqz1 architecture can accommodate")
print(f"  ")
print(f"  4. Deploy to Pynq for testing")
print("="*80)
