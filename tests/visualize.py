from pathlib import Path

import pycolmap
from hloc import visualization
from hloc.utils import viz_3d

# Load the reconstruction
reconstruction = pycolmap.Reconstruction('/home/roborock/datasets/roborock/stereo/rr_stereo_grass_02/sparse/0')
image_dir = Path('/home/roborock/datasets/roborock/stereo/rr_stereo_grass_02/camera/camera0')

# 检查 reconstruction 的类型
if not isinstance(reconstruction, pycolmap.Reconstruction):
    print("Converting reconstruction to pycolmap.Reconstruction")
    reconstruction = pycolmap.Reconstruction(reconstruction)

visualization.visualize_sfm_2d(reconstruction, image_dir, color_by="track_length", n=5)


