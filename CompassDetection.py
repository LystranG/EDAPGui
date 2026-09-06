"""在模型定位的罗盘内，用青色导航点的形状校验前后方向。"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from Screen_Regions import Quad


@dataclass
class NavpointShape:
    behind: bool
    bounding_quad: Quad


def detect_navpoint_shape(image: np.ndarray, compass: Quad) -> NavpointShape | None:
    """只接受唯一、轮廓明确的圆点；颜色不支持或形状不明确时返回 None。"""
    diameter = min(compass.width, compass.height)
    if diameter < 20 or image.size == 0:
        return None
    padding = math.ceil(diameter * .12)
    left = max(0, math.floor(compass.left) - padding)
    top = max(0, math.floor(compass.top) - padding)
    right = min(image.shape[1], math.ceil(compass.right) + padding)
    bottom = min(image.shape[0], math.ceil(compass.bottom) + padding)
    if left >= right or top >= bottom:
        return None

    pixels = image[top:bottom, left:right, :3].astype(np.int16)
    # 青色同时含蓝、绿分量；取两者较小值，排除橙色 HUD 和纯绿色 Overlay。
    signal = np.maximum(0, np.minimum(pixels[:, :, 0], pixels[:, :, 1]) - pixels[:, :, 2])
    mask = (signal >= 12).astype(np.uint8)
    kernel_size = max(3, round(diameter * .03) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 闭运算仅用于定位，空心/实心判断仍使用原始像素，避免把小孔填实。
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(connected)
    candidates = []
    for component in stats[1:]:
        x, y, width, height, _ = map(int, component)
        if not (max(5, diameter * .045) <= min(width, height)
                and max(width, height) <= diameter * .25
                and .65 <= width / height <= 1.55):
            continue
        cx, cy = left + x + width / 2, top + y + height / 2
        dx = (cx - (compass.left + compass.right) / 2) / (compass.width / 2)
        dy = (cy - (compass.top + compass.bottom) / 2) / (compass.height / 2)
        if dx * dx + dy * dy > 1.15 ** 2:
            continue
        local = signal[y:y + height, x:x + width]
        foreground = local >= max(12, float(local.max()) * .25)
        yy, xx = np.mgrid[:height, :width]
        nx, ny = (xx - (width - 1) / 2) / (width / 2), (yy - (height - 1) / 2) / (height / 2)
        radius = np.hypot(nx, ny)
        core = radius <= .35
        rim = (radius >= .55) & (radius <= 1.15)
        sectors = ((np.arctan2(ny, nx) + np.pi) * 4 / np.pi).astype(int) % 8
        coverage = sum(np.any(foreground[rim & (sectors == index)]) for index in range(8))
        center_fill = float(foreground[core].mean())
        total_fill = float(foreground.mean())
        brightness = np.minimum(pixels[y:y + height, x:x + width, 0],
                                pixels[y:y + height, x:x + width, 1])
        ring_pixels = rim & foreground
        # 白色高亮会失去青色色差，不能因此被当作空心；空心还必须有暗色中心。
        dark_center = (np.any(ring_pixels)
                       and float(brightness[core].mean()) < float(brightness[ring_pixels].mean()) * .7)
        if center_fill <= .2 and dark_center and coverage >= 7 and .18 <= total_fill <= .7:
            behind = True
        elif center_fill >= .85 and coverage == 8 and .55 <= total_fill <= .9:
            behind = False
        else:
            continue
        bounds = Quad.from_rect([left + x, top + y, left + x + width, top + y + height])
        candidates.append(NavpointShape(behind=behind, bounding_quad=bounds))

    return candidates[0] if len(candidates) == 1 else None
