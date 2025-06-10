# Face Avatar Gaussian Splatting Example

This repository contains a simple demonstration of face avatar rendering using 3D Gaussian splatting in pure Python. The implementation does not rely on external packages.

The main script `face_avatar_gaussian_splatting.py` generates a dummy face based on randomly sampled sphere points (as a stand‑in for a 3D morphable model), regresses basic Gaussian parameters from UV coordinates, and renders an image from a specified camera viewpoint. The resulting image is saved to `render.ppm` when the script is executed.

For reference, an alternative version that depends on `numpy` and `matplotlib` is provided in `face_avatar_gaussian_splatting_numpy.py`. This version cannot run in restricted environments without installing additional packages.

## Usage

Run the following command:

```bash
python3 face_avatar_gaussian_splatting.py
```

The script outputs `render.ppm` in the current directory and prints a confirmation message.
