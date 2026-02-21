# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm
import pymomentum.geometry as pym_geometry


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )

        if args.save_glb and len(outputs) > 0:
            base_name = os.path.basename(image_path)[:-4]
            glb_path = os.path.join(output_folder, f"{base_name}.glb")
            glb_output = outputs[0]
            full_params = np.concatenate(
                [
                    glb_output["mhr_model_params"],
                    glb_output["shape_params"],
                    glb_output["expr_params"],
                ]
            )
            character = estimator.model.head_pose.mhr.character
            model_params = torch.from_numpy(full_params.reshape(1, -1)).float()
            coord_fix = np.array([1.0, -1.0, -1.0], dtype=np.float32)
            joint_positions = (
                glb_output["pred_joint_coords"].astype(np.float32) * coord_fix
            )
            joints = []
            for joint_index, joint_name in enumerate(character.skeleton.joint_names):
                joint_parent = character.skeleton.joint_parents[joint_index]
                if joint_parent == -1:
                    offset = joint_positions[joint_index]
                else:
                    offset = joint_positions[joint_index] - joint_positions[joint_parent]
                joints.append(
                    pym_geometry.Joint(
                        joint_name,
                        joint_parent,
                        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        offset,
                    )
                )
            skeleton = pym_geometry.Skeleton(joints)
            mesh_faces = glb_output.get("faces", estimator.faces)
            posed_mesh = pym_geometry.Mesh(
                glb_output["pred_vertices"].astype(np.float32) * coord_fix,
                mesh_faces.astype(np.int32),
            )
            baked_character = pym_geometry.Character(
                character.name,
                skeleton,
                character.parameter_transform,
                locators=character.locators,
            ).with_mesh_and_skin_weights(posed_mesh, character.skin_weights).rebind_skin()
            baked_character = baked_character.with_collision_geometry([])
            options = pym_geometry.FileSaveOptions(
                mesh=True,
                locators=False,
                collisions=False,
                blend_shapes=True,
                gltf_file_format=pym_geometry.GltfFileFormat.Binary,
            )
            pym_geometry.Character.save_gltf(
                glb_path,
                baked_character,
                fps=1.0,
                options=options,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save a GLB for the first detected person in each image",
    )
    args = parser.parse_args()

    main(args)
