import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymomentum.geometry as pym_geometry
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body


class InferenceRequest(BaseModel):
    image_base64: str
    bbox_thr: float = 0.8
    use_mask: bool = False


def _load_image_from_base64(image_base64: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(image_base64)
    except base64.binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build_glb(
    estimator: SAM3DBodyEstimator,
    output: dict[str, Any],
    output_path: Path,
) -> None:
    full_params = np.concatenate(
        [
            output["mhr_model_params"],
            output["shape_params"],
            output["expr_params"],
        ]
    )
    character = estimator.model.head_pose.mhr.character
    model_params = torch.from_numpy(full_params.reshape(1, -1)).float()
    coord_fix = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    # coord_scale = 100.0
    coord_scale = 1.0
    joint_positions = (
        output["pred_joint_coords"].astype(np.float32) * coord_fix * coord_scale
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
    mesh_faces = output.get("faces", estimator.faces)
    posed_mesh = pym_geometry.Mesh(
        output["pred_vertices"].astype(np.float32) * coord_fix * coord_scale,
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
        str(output_path),
        baked_character,
        fps=1.0,
        options=options,
    )


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(item) for item in value]
    return value


app = FastAPI()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SAM-3D-Body server"}


@app.on_event("startup")
def _startup() -> None:
    global ESTIMATOR

    checkpoint_dir = os.environ.get("SAM3D_CHECKPOINT_DIR", "")
    checkpoint_file = os.environ.get("SAM3D_CHECKPOINT_FILE", "model.ckpt")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if not checkpoint_path and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if not checkpoint_path:
        raise RuntimeError("SAM3D_CHECKPOINT_PATH is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhr_path = os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = os.environ.get("SAM3D_FOV_PATH", "")
    detector_name = os.environ.get("SAM3D_DETECTOR_NAME", "vitdet")
    segmentor_name = os.environ.get("SAM3D_SEGMENTOR_NAME", "sam2")
    fov_name = os.environ.get("SAM3D_FOV_NAME", "moge2")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector = None
    human_segmentor = None
    fov_estimator = None
    if detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=detector_name, device=device, path=detector_path
        )

    if (segmentor_name == "sam2" and len(segmentor_path)) or segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device, path=fov_path)

    ESTIMATOR = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


@app.post("/infer")
def infer(request: InferenceRequest) -> dict[str, Any]:
    if "ESTIMATOR" not in globals():
        raise HTTPException(status_code=500, detail="Estimator not initialized")

    image = _load_image_from_base64(request.image_base64)
    outputs = ESTIMATOR.process_one_image(
        image,
        bbox_thr=request.bbox_thr,
        use_mask=request.use_mask,
    )
    if not outputs:
        return {"results": []}

    output_dir = Path("/tmp/sam3d_glb")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, output in enumerate(outputs):
        glb_path = output_dir / f"result_{idx:03d}.glb"
        _build_glb(ESTIMATOR, output, glb_path)
        glb_bytes = glb_path.read_bytes()

        faces = output.get("faces")
        output = {key: value for key, value in output.items() if key != "faces"}
        results.append(
            {
                "outputs": _make_jsonable(output),
                "faces": _make_jsonable(faces),
                "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
            }
        )

    return {"results": results}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
