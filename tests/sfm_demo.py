import argparse
import os
from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    pairs_from_retrieval,
)

def check_cuda_env():
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"CUDA environment is set. Path: {cuda_path}")
    else:
        print("CUDA environment is not set.")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Run SfM pipeline with custom paths.")
    parser.add_argument("--images", type=str, required=True, help="Path to the images directory.")
    parser.add_argument("--outputs", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_matched", type=int, default=30, help="Number of image pairs to match.")
    return parser.parse_args()


def check_paths(paths):
    """
    检查路径是否存在，抛出异常以确保流程的完整性。
    :param paths: 待检查的路径列表
    """
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")


def main(args):
    """
    主函数：运行 SfM 流程，包括特征提取、匹配、重建等步骤。
    """
    images = Path(args.images)
    images_list = images / "images_list.txt"
    outputs = Path(args.outputs)

    # 输出路径
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    # 配置
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    check_cuda_env()

    try:
        # 确保路径存在
        check_paths([images, images_list, outputs])

        # 特征检索
        retrieval_path = extract_features.main(
            retrieval_conf, images, outputs, image_list=images_list
        )
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=args.num_matched)

        # 特征提取
        feature_path = extract_features.main(
            feature_conf, images, outputs, image_list=images_list
        )

        # 特征匹配
        match_path = match_features.main(
            matcher_conf, sfm_pairs, feature_conf["output"], outputs
        )

        # 三维重建
        model = reconstruction.main(
            sfm_dir, images, sfm_pairs, feature_path, match_path
        )

        # 打印结果
        print(f"Structure-from-Motion completed. Model saved at: {sfm_dir}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
