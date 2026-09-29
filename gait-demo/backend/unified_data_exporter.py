"""Reconstructed exporter v1. This is newly implemented, not recovered source.

Axis swap is validated against supplied VP3D CSVs; centering/scaling follows
the supplied TRC preparation. The Z heel-event hypothesis has NOT passed paired
video validation. Neck coordinate dominance alone does not establish the physical
height axis. Keep v1 frozen for audit; do not enable live grading on format checks.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from .models import COLUMNS

VERSION='reconstructed-exporter-v1'
DEFAULT_UNIFIED_DIR=Path(__file__).resolve().parents[1]/'data'

def vp3d_keypoints_to_unified(pose):
    pose=np.asarray(pose,dtype=np.float64)
    if pose.ndim!=3 or pose.shape[1:]!=(19,3) or not np.isfinite(pose).all():
        raise ValueError('三维骨架必须是有限的 T×19×3 数组')
    return pose[:,:,[2,1,0]].copy()

def normalize_3d(pose):
    pose=np.asarray(pose,dtype=np.float64)
    relative=pose-pose[:,8:9,:]
    lengths=np.linalg.norm(relative[:,1,:],axis=1)
    scale=float(np.median(lengths))
    if not np.isfinite(scale) or scale<1e-6:
        raise ValueError('躯干尺度无效，无法归一化三维姿态')
    return relative/scale

def detect_heel_strikes(normalized,fps=30):
    if not np.isfinite(fps) or fps<=0:raise ValueError('帧率无效')
    events=[]
    for foot,joint in [('R',15),('L',17)]:
        height=normalized[:,joint,2]
        window=max(3,int(round(fps*.15))|1)
        if len(height)>=window:height=savgol_filter(height,window,2)
        amplitude=float(np.ptp(height))
        if amplitude<.01:continue
        indices,properties=find_peaks(-height,distance=max(1,int(round(fps*.4))),
                                     prominence=max(.01,amplitude*.04))
        events.extend({'frame':int(i),'foot':foot,'prominence':float(p)}
                      for i,p in zip(indices,properties['prominences']))
    return sorted(events,key=lambda e:(e['frame'],e['foot']))

def resample_stride(pose,target_frames=40):
    if len(pose)<2:raise ValueError('周期不足两帧')
    old=np.linspace(0,1,len(pose));new=np.linspace(0,1,target_frames)
    return np.stack([np.interp(new,old,pose[:,j,c]) for j in range(19) for c in range(3)],axis=1)

def process_and_export(pose,subject_id,scenario,level=None,source='VP3D-19J',
                       output_dir=DEFAULT_UNIFIED_DIR,target_frames=40,fps=30,valid=None):
    if target_frames!=40:raise ValueError('当前分类模型只接受40帧周期')
    norm=normalize_3d(pose)
    if valid is None:valid=np.ones(len(norm),dtype=bool)
    valid=np.asarray(valid,dtype=bool)
    if valid.shape!=(len(norm),):raise ValueError('有效帧掩码长度不匹配')
    events=detect_heel_strikes(norm,fps);records=[];strides=[]
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for foot in ['R','L']:
        same=[e['frame'] for e in events if e['foot']==foot]
        for start,end in zip(same,same[1:]):
            duration=(end-start)/fps
            if not .5<=duration<=2.5:continue
            if valid[start:end+1].mean()<.7:continue
            # Reject long interpolated gaps within a proposed cycle.
            gap=0;max_gap=0
            for value in valid[start:end+1]:
                gap=0 if value else gap+1;max_gap=max(max_gap,gap)
            if max_gap/fps>.25:continue
            records.append({'start':start,'end':end,'foot':foot,'frames':end-start+1,
                            'duration':duration,'valid_ratio':float(valid[start:end+1].mean())})
    records.sort(key=lambda r:(r['start'],r['foot']))
    for index,record in enumerate(records,1):
        frame=resample_stride(norm[record['start']:record['end']+1],target_frames)
        name=f'stride_{index:03d}.csv'
        pd.DataFrame(frame,columns=COLUMNS).to_csv(output_dir/name,index=False)
        record.update({'file':name,'source':source,'exporter_version':VERSION})
        strides.append(frame)
    (output_dir/'cycle-provenance.json').write_text(json.dumps({
        'version':VERSION,'fps':fps,'coordinate_permutation':[2,1,0],
        'normalization':'framewise MidHip centering / sequence median Neck-MidHip length',
        'resampling':'linear, endpoint inclusive, 40 frames',
        'heel_detection':'Z minima, Savitzky-Golay 0.15s; distance 0.4s; prominence max(0.01,0.04*range)',
        'duration_range_seconds':[.5,2.5],'cycle_min_valid_ratio':.7,'max_missing_gap_seconds':.25,
        'events':events,'cycles':records,
        'note':'新补实现；未声称逐字恢复旧导出源码或复现所有旧周期边界。'
    },ensure_ascii=False,indent=2),encoding='utf8')
    return norm,strides,records
