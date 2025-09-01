import os
from glob import glob
import numpy as np
import imageio.v3 as iio  
from skimage.morphology import disk, binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes


def postprocess_tme(cws_folder, ss1_dir, ss1_post_dir, nfile=0, file_pattern='*.ndpi', disk_radius=15, area_min=900):
    if not os.path.exists(ss1_post_dir):
        os.makedirs(ss1_post_dir)

    cws_files = sorted(glob(os.path.join(cws_folder, file_pattern)))
    file_name = os.path.split(cws_files[nfile])[-1]
    print(file_name)
   
    img = iio.imread(os.path.join(ss1_dir, file_name+'_Ss1.png'))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]  # drop alpha if present

    tme_colors = [
        (128, 0, 0),
        (255, 204, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 0),
    ]

    mask = np.zeros(img.shape[:2], dtype=bool)
    for c in tme_colors:
        mask |= np.all(img == np.array(c, dtype=img.dtype), axis=2)

    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, footprint=disk(disk_radius))
    mask = binary_fill_holes(mask)

    mask = remove_small_objects(mask, min_size=area_min)
    masked = (mask.astype(np.uint8)[..., None] * img).astype(np.uint8)

    iio.imwrite(os.path.join(ss1_post_dir, file_name+'_Ss1.png'), masked)
