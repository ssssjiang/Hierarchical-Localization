# shu.song@ninebot.com
import argparse
import sqlite3
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from pathlib import Path
import logging

from utils.hypermap_database import HyperMapDatabase, image_ids_to_pair_id
from utils.hfnet_database import HFNetDatabase


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_pairs', type=str, default='/third_party/SuperGluePretrainedNetwork/assets/mower_pairs_with_gt.txt',
    help='Path to the list of image pairs')
parser.add_argument(
    '--input_dir', type=str, default='/third_party/SuperGluePretrainedNetwork/assets/map/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--database', type=str, default='/persist/data/',
    help='Path to the hfnet.db')
parser.add_argument('--outputs', type=Path, default='outputs/mower',
                    help='Path to the output directory, default: %(default)s')
args = parser.parse_args()

# Setup the paths
hypermap_database = args.database + "hypermap.db"
hfnet_database = args.database + "hfnet.db"

outputs = args.outputs  # where everything will be saved

hypermap_cursor = HyperMapDatabase.connect(hypermap_database)
hfnet_cursor = HFNetDatabase.connect(hfnet_database)

image1_name = "cam0/1624530162714062848-f.jpg"
image2_name = "cam0/1624530162814053888-f.jpg"

image1_id = hypermap_cursor.read_image_id_from_name(image1_name)
image2_id = hypermap_cursor.read_image_id_from_name(image2_name)

pair_id = image_ids_to_pair_id(image1_id, image2_id)
raw_matches = hypermap_cursor.read_matches_from_pair_id(pair_id)

# keypoint1 = hfnet_cursor.read_keypoints_from_image_id(image1_id)
# keypoint2 = hfnet_cursor.read_keypoints_from_image_id(image2_id)
keypoint1 = hypermap_cursor.read_keypoints_from_image_id(image1_id)[:, 0:2]
keypoint2 = hypermap_cursor.read_keypoints_from_image_id(image2_id)[:, 0:2]

matches = np.full((min(np.shape(keypoint1)[0], np.shape(keypoint2)[0]), ), -1)
if raw_matches is not None:
    for match in raw_matches:
        matches[match[0]] = match[1]





print("Done!")


