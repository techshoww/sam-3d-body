import base64
from pathlib import Path

import cv2
import numpy as np
import requests

from tools.vis_utils import visualize_sample_together


def encode_image(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_image(image_path: str) -> np.ndarray:
    image_bytes = Path(image_path).read_bytes()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    return image


def main() -> None:
    image_path = "./images/demo.jpg"
    image_base64 = encode_image(image_path)
    response = requests.post(
        "http://localhost:8000/infer",
        json={"image_base64": image_base64, "bbox_thr": 0.8, "use_mask": False},
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if payload.get("glb_base64"):
        glb_bytes = base64.b64decode(payload["glb_base64"])
        (output_dir / "result.glb").write_bytes(glb_bytes)
        print("Saved GLB to output/result.glb")
    else:
        print("No GLB returned")

    outputs = payload.get("outputs", [])
    if outputs:
        img = decode_image(image_path)
        faces = np.array(outputs[0]["faces"], dtype=np.int64)
        rend_img = visualize_sample_together(img, outputs, faces)
        cv2.imwrite(str(output_dir / "result.jpg"), rend_img.astype(np.uint8))
        print("Saved visualization to output/result.jpg")
    else:
        print("No outputs returned")


if __name__ == "__main__":
    main()
