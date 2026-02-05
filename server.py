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
    return image


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
    motion = (
        estimator.model.head_pose.mhr.character.parameter_transform.names,
        full_params.reshape(1, -1),
    )
    options = pym_geometry.FileSaveOptions(
        mesh=True,
        locators=True,
        collisions=True,
        blend_shapes=True,
        gltf_file_format=pym_geometry.GltfFileFormat.Binary,
    )
    pym_geometry.Character.save_gltf(
        str(output_path),
        estimator.model.head_pose.mhr.character,
        fps=1.0,
        motion=motion,
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
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device)
    ESTIMATOR = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)


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
        return {"outputs": [], "faces": None, "glb_base64": None}

    output_dir = Path("/tmp/sam3d_glb")
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / "result.glb"
    _build_glb(ESTIMATOR, outputs[0], glb_path)
    glb_bytes = glb_path.read_bytes()

    faces = None
    if outputs:
        faces = outputs[0].get("faces")
    outputs = [
        {key: value for key, value in output.items() if key != "faces"}
        for output in outputs
    ]

    return {
        "outputs": _make_jsonable(outputs),
        "faces": _make_jsonable(faces),
        "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
