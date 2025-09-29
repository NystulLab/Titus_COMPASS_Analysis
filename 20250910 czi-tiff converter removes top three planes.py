import os
import numpy as np
from aicspylibczi import CziFile
import tifffile

input_dir = "/media/el_tito/T7 Shield/20250605 histone compass/2025-09-19/"
output_dir = "/media/el_tito/T7 Shield/20250605 histone compass/tiffed30z/h3k4me2/"
os.makedirs(output_dir, exist_ok=True)

def czi_to_tiff_final(input_path, output_path):
    czi = CziFile(input_path)
    arr, _ = czi.read_image()

    # Handle dims
    dims_shape = czi.get_dims_shape()[0]
    print(f"\nProcessing: {os.path.basename(input_path)}")
    print(f" - Raw shape: {arr.shape}")
    for dim, val in dims_shape.items():
        print(f"   {dim}: {val}")

    arr = np.squeeze(arr)  # drop singleton S/H
    print(f" - Squeezed shape: {arr.shape}")

    # Transpose from (C, Z, Y, X) → (Z, C, Y, X)
    if arr.shape[0] <= 10 and arr.shape[1] > arr.shape[0]:
        arr = np.transpose(arr, (1, 0, 2, 3))
        print(f" - Transposed to: {arr.shape} (Z, C, Y, X)")
    else:
        print(" - Warning: Unexpected axis layout, skipping transpose.")

    # Remove top 3 Z planes
    if arr.shape[0] > 3:
        arr = arr[3:, ...]
        print(f" - Cropped shape (removed 3 Z planes): {arr.shape}")
    else:
        print(" - Warning: Less than 3 Z planes, skipping cropping.")

    tifffile.imwrite(output_path, arr, imagej=True)
    print(f" - Saved to: {output_path}")

def batch_convert(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.lower().endswith(".czi"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".tiff")
            czi_to_tiff_final(input_path, output_path)

# Run it
batch_convert(input_dir, output_dir)