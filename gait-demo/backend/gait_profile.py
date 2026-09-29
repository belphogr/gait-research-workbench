"""Descriptive video metrics for the UI; never classifier inputs or diagnoses."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import displays


def _score(value):
    return round(float(np.clip(value, 0, 100)), 1)


def build_profile(folder: Path, valid_ratio: float):
    provenance = json.loads((folder / 'research-cycle-provenance.json').read_text(encoding='utf-8'))
    cycles = provenance['cycles']
    with np.load(folder / 'landmarks-2d.npz') as pose:
        points = pose['keypoints']
        valid = pose['valid'].astype(bool)
    if not cycles or not valid.any():
        return None

    durations = np.array([c['duration'] for c in cycles], dtype=float)
    median_duration = float(np.median(durations))
    rhythm_deviation = float(np.median(np.abs(durations - median_duration)) / median_duration) if median_duration > 0 and len(durations) >= 3 else None

    shoulder = points[:, [11, 12], :2].mean(axis=1)
    hip = points[:, [23, 24], :2].mean(axis=1)
    trunk = shoulder - hip
    angles = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))
    usable = valid & np.isfinite(angles)
    trunk_mad = float(np.median(np.abs(angles[usable] - np.median(angles[usable])))) if usable.sum() >= 3 else None

    left_rom, right_rom = [], []
    for cycle in cycles:
        rom = next((row for row in displays(pd.read_csv(folder / cycle['file']))['rom'] if row['joint'] == '膝'), None)
        if rom and rom['left'] is not None and rom['right'] is not None:
            left_rom.append(rom['left'])
            right_rom.append(rom['right'])
    left = float(np.median(left_rom)) if left_rom else None
    right = float(np.median(right_rom)) if right_rom else None
    symmetry = abs(left - right) / max(left, right) if left is not None and right is not None and max(left, right) > 0 else None
    knee_rom = (left + right) / 2 if left is not None and right is not None else None

    items = [
        dict(key='quality', label='数据质量', score=_score(valid_ratio * 100), value=f'{valid_ratio * 100:.1f}%', formula='有效人体帧 / 视频总帧 × 100', source='MediaPipe 二维姿态'),
        dict(key='rhythm', label='节律一致', score=_score(100 * (1 - rhythm_deviation)) if rhythm_deviation is not None else None, value=f'周期时长中位数 {median_duration:.2f} 秒' if rhythm_deviation is not None else '周期不足', formula='100 × (1 − 周期时长绝对偏差中位数 / 周期时长中位数)', source='候选运动周期；至少 3 个'),
        dict(key='symmetry', label='左右协调', score=_score(100 * (1 - symmetry)) if symmetry is not None else None, value=f'左 {left:.1f}° · 右 {right:.1f}°' if symmetry is not None else '不可计算', formula='100 × (1 − 左右膝活动范围差值 / 较大侧活动范围)', source='VP3D 三维估计；周期中位数'),
        dict(key='stability', label='躯干平稳', score=_score(100 * (1 - trunk_mad / 15)) if trunk_mad is not None else None, value=f'角度波动 {trunk_mad:.1f}°' if trunk_mad is not None else '不可计算', formula='100 × (1 − 躯干倾角绝对偏差中位数 / 15°)', source='MediaPipe 图像平面；15°仅为绘图刻度'),
        dict(key='mobility', label='膝部活动', score=_score(100 * knee_rom / 70) if knee_rom is not None else None, value=f'活动范围 {knee_rom:.1f}°' if knee_rom is not None else '不可计算', formula='双侧膝活动范围均值 / 70° × 100', source='VP3D 三维估计；70°仅为绘图刻度'),
    ]
    return {'version': 'observational-profile-v1', 'items': items, 'note': '本次视频的描述性观察画像；高分不等于健康或康复改善。15°与70°只是固定绘图刻度，未经临床标定。'}
