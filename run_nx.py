import cv2
import math
import sys
import torch
import numpy as np
import argparse
import time
import os
import zipfile
import shutil

'''==========import from our code=========='''
sys.path.append('.')
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VFIMamba', type=str)
parser.add_argument('--scale', default=0, type=float)
parser.add_argument('--n', default=32, type=int)
parser.add_argument('--img1', default='example/frame_00083.png', type=str)
parser.add_argument('--img2', default='example/frame_00084.png', type=str)
parser.add_argument('--output', default='example/output', type=str)  # Folder/zip name (without extension)
parser.add_argument('--resize', default=0.5, type=float, help='Resize factor (0.5 = 50%)')
parser.add_argument('--keep_folder', action='store_true', help='Keep folder after creating zip')

args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'
assert args.n > 0 and (args.n & (args.n - 1)) == 0, 'n must be power of 2 (2, 4, 8, 16, 32, ...)'


'''==========Model setting=========='''
TTA = False

print("=" * 60)
print("VFIMamba Frame Interpolation")
print("=" * 60)

print(f"\n[1/6] Loading model: {args.model}")
start_time = time.time()
model = Model.from_pretrained(args.model)
model.eval()
model.device()
print(f"      Model loaded in {time.time() - start_time:.2f}s")


# Global counter for progress
inference_count = 0
total_inferences = 0


def count_inferences(n):
    """Calculate total number of inferences needed"""
    if n <= 1:
        return 0
    return (n - 1)


def generate_frames(frame1, frame2, num_interp, depth=0):
    """
    Generate num_interp intermediate frames between frame1 and frame2.
    """
    global inference_count
    
    if num_interp == 0:
        return []
    
    # Progress indicator
    inference_count += 1
    progress = (inference_count / total_inferences) * 100
    bar_length = 30
    filled = int(bar_length * inference_count / total_inferences)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r      [{bar}] {progress:5.1f}% ({inference_count}/{total_inferences} interpolations)", end="", flush=True)
    
    # Generate middle frame
    mid_frame = model.inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA, scale=args.scale)
    
    if num_interp == 1:
        return [mid_frame]
    
    # Recursively generate frames for each half
    left_frames = generate_frames(frame1, mid_frame, num_interp // 2, depth + 1)
    right_frames = generate_frames(mid_frame, frame2, num_interp // 2, depth + 1)
    
    return left_frames + [mid_frame] + right_frames


def resize_to_divisible_by_32(img):
    """Resize image so dimensions are divisible by 32"""
    h, w = img.shape[:2]
    new_h = math.ceil(h / 32) * 32
    new_w = math.ceil(w / 32) * 32
    if new_h != h or new_w != w:
        resized = cv2.resize(img, (new_w, new_h))
        print(f"      Resized: {w}x{h} -> {new_w}x{new_h} (divisible by 32)")
        return resized
    print(f"      Size OK: {w}x{h} (already divisible by 32)")
    return img


def tensor_to_numpy(tensor, padder, original_size):
    """Convert tensor to numpy image and restore original size"""
    img = (padder.unpad(tensor[0]).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    h, w = original_size
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h))
    return img


def resize_high_quality(img, scale):
    """Resize image with high quality interpolation to preserve colors"""
    if scale == 1.0:
        return img
    
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for downscaling (best quality for shrinking)
    # Use INTER_LANCZOS4 for upscaling (best quality for enlarging)
    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def create_zip(folder_path, zip_path):
    """Create a zip file from a folder"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


# ========== MAIN ==========

print(f"\n[2/6] Loading images")
print(f"      Image 1: {args.img1}")
I0 = cv2.imread(args.img1)
if I0 is None:
    raise FileNotFoundError(f"Could not load image: {args.img1}")
print(f"               Loaded: {I0.shape[1]}x{I0.shape[0]} ({I0.shape[2]} channels)")

print(f"      Image 2: {args.img2}")
I2 = cv2.imread(args.img2)
if I2 is None:
    raise FileNotFoundError(f"Could not load image: {args.img2}")
print(f"               Loaded: {I2.shape[1]}x{I2.shape[0]} ({I2.shape[2]} channels)")

# Store original size
original_size = (I0.shape[0], I0.shape[1])

print(f"\n[3/6] Preprocessing images")
I0_resized = resize_to_divisible_by_32(I0)
I2_resized = resize_to_divisible_by_32(I2)

# Convert to tensors
print(f"      Converting to tensors...")
I0_ = (torch.tensor(I0_resized.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2_resized.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
print(f"      Tensor shape: {list(I0_.shape)}")

# Pad
padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
print(f"      Padded shape: {list(I0_.shape)}")

# Calculate interpolations
num_interp = args.n - 1
total_inferences = count_inferences(args.n)

print(f"\n[4/6] Generating frames")
print(f"      Target frames: {args.n}")
print(f"      Interpolations needed: {total_inferences}")
print(f"      Recursion depth: {int(math.log2(args.n))}")
print()

start_time = time.time()

# Generate intermediate frames
intermediate = generate_frames(I0_, I2_, num_interp - 1)

elapsed = time.time() - start_time
print()  # New line after progress bar
print(f"      Completed in {elapsed:.2f}s ({elapsed/max(total_inferences,1):.2f}s per frame)")

# Build final list
all_frames = [I0_] + intermediate + [I2_]
print(f"      Total frames: {len(all_frames)}")

print(f"\n[5/6] Converting and resizing frames")
print(f"      Resize factor: {args.resize * 100:.0f}%")

# Calculate output size
output_w = int(original_size[1] * args.resize)
output_h = int(original_size[0] * args.resize)
print(f"      Output size: {output_w}x{output_h}")

# Create output folder
output_folder = args.output
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)
print(f"      Output folder: {output_folder}/")

frames_saved = []
for i, frame in enumerate(all_frames):
    # Convert tensor to numpy
    img = tensor_to_numpy(frame, padder, original_size)
    
    # Resize with high quality
    img_resized = resize_high_quality(img, args.resize)
    
    # Save as PNG (lossless, preserves color quality)
    filename = f"frame_{i:05d}.png"
    filepath = os.path.join(output_folder, filename)
    
    # Use PNG compression with best quality settings
    # PNG compression level 3 is a good balance (0=none, 9=max compression)
    cv2.imwrite(filepath, img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    frames_saved.append(filepath)
    
    if (i + 1) % 10 == 0 or i == len(all_frames) - 1:
        print(f"\r      Saved: {i+1}/{len(all_frames)} frames", end="", flush=True)

print()

print(f"\n[6/6] Creating ZIP archive")
zip_path = f"{args.output}.zip"
print(f"      Creating: {zip_path}")

create_zip(output_folder, zip_path)

# Get zip file size
zip_size = os.path.getsize(zip_path)
if zip_size > 1024 * 1024:
    size_str = f"{zip_size / (1024*1024):.2f} MB"
else:
    size_str = f"{zip_size / 1024:.2f} KB"

print(f"      ZIP size: {size_str}")

# Clean up folder if not keeping
if not args.keep_folder:
    shutil.rmtree(output_folder)
    print(f"      Folder cleaned up")
else:
    print(f"      Folder kept: {output_folder}/")

print("\n" + "=" * 60)
print("Done!")
print(f"  Output: {zip_path}")
print(f"  Frames: {len(all_frames)}")
print(f"  Size: {output_w}x{output_h} ({args.resize * 100:.0f}% of original)")
print("=" * 60)
