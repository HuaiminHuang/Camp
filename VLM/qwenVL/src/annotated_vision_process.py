# annotated_vision_process.py
# 这是一个对 `qwen_vl_utils/vision_process.py` 的注释学习版
# 主要解释了动态分辨率和动态帧率采样等关键预处理逻辑

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


# --- 全局常量定义 ---
MAX_RATIO = 200  # 允许的最大图像长宽比
SPATIAL_MERGE_SIZE = 2 # 空间合并尺寸，即 2x2 的 patch 会被合并
IMAGE_MIN_TOKEN_NUM = 4 # 图像转换后允许的最小 Token 数
IMAGE_MAX_TOKEN_NUM = 16384 # 图像转换后允许的最大 Token 数
VIDEO_MIN_TOKEN_NUM = 128 # 视频转换后允许的最小 Token 数
VIDEO_MAX_TOKEN_NUM = 768 # 视频转换后允许的最大 Token 数

FPS = 2.0 # 默认的视频采样帧率
FRAME_FACTOR = 2 # 帧数必须是该因子的倍数
# ... 其他常量 ...

logger = logging.getLogger(__name__)


# --- 尺寸处理辅助函数 ---

def round_by_factor(number: int, factor: int) -> int:
    """将数字四舍五入到最接近的 `factor` 的倍数。"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """将数字向上取整到 `factor` 的倍数。"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """将数字向下取整到 `factor` 的倍数。"""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    【核心函数 - 动态分辨率的“大脑”】
    智能地重新缩放图像尺寸，以满足以下三个条件：
    1. 新的高度和宽度都必须能被 `factor` (通常是28) 整除。
    2. 缩放后的总像素数必须在 `min_pixels` 和 `max_pixels` 范围内。
    3. 尽可能地保持原始图像的长宽比。

    Args:
        height (int): 原始高度。
        width (int): 原始宽度。
        factor (int): 尺寸必须为其倍数的因子 (来自 patch_size * merge_size)。
        min_pixels (int, optional): 最小允许的总像素数。
        max_pixels (int, optional): 最大允许的总像素数。

    Returns:
        Tuple[int, int]: (新的高度, 新的宽度)
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    
    # ... 检查长宽比是否极端 ...

    # 初始尝试：直接将高和宽调整到最接近 factor 的倍数
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # 如果调整后像素过多，则按比例缩小
    if h_bar * w_bar > max_pixels:
        # 计算缩放比例 beta，使得总像素数接近 max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 按 beta 缩放并向下取整到 factor 的倍数
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    # 如果调整后像素过少，则按比例放大
    elif h_bar * w_bar < min_pixels:
        # 计算缩放比例 beta，使得总像素数接近 min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 按 beta 缩放并向上取整到 factor 的倍数
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """确保图像是 RGB 格式。"""
    if pil_image.mode == 'RGBA':
        # 创建一个白色背景并粘贴原图，以处理透明通道
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    """
    获取并处理单张图像。
    - 从本地路径、URL或base64等多种来源读取图像。
    - 调用 `smart_resize` 实现动态分辨率调整。
    """
    # ... (从不同来源读取图像到 image_obj 的逻辑) ...
    image = to_rgb(image_obj)

    # 关键步骤：计算用于缩放的 `factor`
    # 它等于 patch_size * spatial_merge_size (14 * 2 = 28)
    # 这确保了缩放后的尺寸能被模型架构完美处理
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    
    width, height = image.size
    min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
    max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
    
    # 调用核心函数计算新的目标尺寸
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    
    # 使用PIL的resize函数将图像缩放到计算出的新尺寸
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(ele: Dict[str, Any], total_frames: int, video_fps: Union[int, float]) -> int:
    """
    【核心函数 - 动态帧率采样的“大脑”】
    根据用户指定的 `fps` 或 `nframes` (采样帧数)，智能地计算出最终要从视频中采样多少帧。
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    
    if "nframes" in ele:
        # 如果直接指定了帧数，则调整到 FRAME_FACTOR 的倍数
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        # 如果指定了目标fps，则根据视频总时长和目标fps计算应采样的帧数
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        
        # 确保计算出的帧数在合理的最小和最大范围内
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)

    # ... (检查 nframes 是否在有效范围内) ...
    return nframes


def _read_video_torchvision(ele: Dict[str, Any]) -> Tuple[torch.Tensor, dict, float]:
    """
    使用 torchvision 后端读取和采样视频。
    """
    # ... (读取视频的逻辑) ...
    video, audio, info = io.read_video(...)
    total_frames, video_fps = video.size(0), info["video_fps"]
    
    # 1. 决定需要采样多少帧
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    # 2. 生成均匀间隔的帧索引
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    
    # 3. 实际采样帧
    video = video[idx]
    
    # 计算实际的采样率
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    
    # ... (返回 video, metadata, sample_fps) ...
    return video, video_metadata, sample_fps

# ... (_read_video_decord, _read_video_torchcodec 等其他后端实现类似逻辑) ...


def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, **kwargs) -> Union[torch.Tensor, List[Image.Image]]:
    """
    获取并处理视频。
    - 选择一个可用的后端 (torchcodec, decord, torchvision) 读取视频。
    - 调用 `smart_nframes` 实现动态帧率采样。
    - 调用 `smart_resize` 对采样出的每一帧进行动态分辨率调整。
    """
    # ... (选择后端并调用 _read_video_* 函数读取视频) ...
    # 视频被读取为 `video` 张量, shape: [T, C, H, W]
    
    nframes, _, height, width = video.shape

    # 对视频的每一帧，同样应用动态分辨率调整
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    # ... (计算 min_pixels, max_pixels) ...
    
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    
    # 对所有采样出的帧进行 resize
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    # ... (返回处理好的 video) ...
    return final_video

# ... (其他辅助函数) ...
