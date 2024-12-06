from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

#%%
images = Path("/home/roborock/datasets/roborock/stereo/414-living-room/selected_images/")

outputs = Path("/home/roborock/datasets/roborock/stereo/414-living-room/sfm/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superpoint+lightglue"]

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=30)
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

opts = dict(camera_model='OPENCV_FISHEYE', camera_params=','.join(map(str, (259.7568821036086,260.10819099573615,394.9905360190833,294.44467823631834,0.0008605939481375175,0.015921588486384006,-0.012233412348966891,0.0012503893360738545))))
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options=opts)