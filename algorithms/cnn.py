from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from algorithms.utils import prepare_gray


# =========================
# CONFIG
# =========================
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_finetuned.pth"
MODEL_STRIDE = 8
MAX_CNN_DIM = 1024


# =========================
# MODEL
# =========================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(1, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            block(256, 512),
            nn.Dropout(0.3),
        )

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = block(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        return self.final(d3)


# =========================
# LOAD MODEL (CACHED)
# =========================
@lru_cache(maxsize=1)
def _get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


# =========================
# MAIN INFERENCE FUNCTION
# =========================
def apply_trained_model(gray_img: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    gray = prepare_gray(gray_img, bgr_to_gray=True)

    if output_size is None:
        output_size = (gray.shape[1], gray.shape[0])

    target_width, target_height = output_size

    # Resize if needed
    if (target_width, target_height) != (gray.shape[1], gray.shape[0]):
        interpolation = (
            cv2.INTER_AREA
            if target_width < gray.shape[1] or target_height < gray.shape[0]
            else cv2.INTER_CUBIC
        )
        gray = cv2.resize(gray, (target_width, target_height), interpolation=interpolation)

    # Cap resolution to prevent OOM on large images
    final_size = (gray.shape[1], gray.shape[0])  # (w, h) to restore later
    h, w = gray.shape[:2]
    if max(h, w) > MAX_CNN_DIM:
        scale = MAX_CNN_DIM / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Pad for UNet compatibility
    padded_gray, pad_right, pad_bottom = _pad_to_multiple(gray, MODEL_STRIDE)

    # =========================
    # CORRECT NORMALIZATION 
    # =========================
    L = padded_gray.astype(np.float32) / 255.0 * 100.0   # [0,100]
    model_input = L / 50.0 - 1.0                         # [-1,1]

    tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0)

    model, device = _get_model()
    tensor = tensor.to(device)

    with torch.inference_mode():
        pred_ab = model(tensor)[0].cpu().numpy()

    # =========================
    # COLOR BOOST (IMPROVES REDS)
    # =========================
    pred_ab = pred_ab * 1.2
    pred_ab = np.clip(pred_ab, -1, 1)

    # Convert back to LAB
    ab = pred_ab.transpose(1, 2, 0) * 128.0

    lab = np.zeros((padded_gray.shape[0], padded_gray.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = padded_gray.astype(np.float32) * 100.0 / 255.0
    lab[:, :, 1:] = ab

    # Convert LAB → BGR
    lab = lab.astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Remove padding
    if pad_bottom or pad_right:
        bgr = bgr[: gray.shape[0], : gray.shape[1]]

    # Upscale back to original requested size if we capped resolution
    if (bgr.shape[1], bgr.shape[0]) != final_size:
        bgr = cv2.resize(bgr, final_size, interpolation=cv2.INTER_CUBIC)

    return bgr





def _pad_to_multiple(gray: np.ndarray, multiple: int):
    h, w = gray.shape[:2]
    pad_right = (-w) % multiple
    pad_bottom = (-h) % multiple

    if pad_right == 0 and pad_bottom == 0:
        return gray, 0, 0

    padded = cv2.copyMakeBorder(
        gray,
        0, pad_bottom,
        0, pad_right,
        cv2.BORDER_REFLECT_101
    )

    return padded, pad_right, pad_bottom