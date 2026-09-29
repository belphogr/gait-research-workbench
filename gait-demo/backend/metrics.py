"""Geometric displays, separate from classifier features and diagnoses."""
import numpy as np
from .models import JOINTS

EDGES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),(11,15),(11,16),(8,12),(12,13),(13,14),(14,17),(14,18)]

def displays(frame):
    pose = frame.fillna(0).to_numpy().reshape(40,19,3)
    angles = {}
    for side in ['L','R']:
        for label, triple in [('髋',('Neck',side+'Hip',side+'Knee')),('膝',(side+'Hip',side+'Knee',side+'Ankle')),('踝',(side+'Knee',side+'Ankle',side+'Toe'))]:
            a,b,c = [pose[:,JOINTS.index(j)] for j in triple]
            u,v = a-b,c-b
            denominator = np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
            valid = denominator > 1e-8
            values = np.full(40,np.nan)
            values[valid] = np.degrees(np.arccos(np.clip((u[valid]*v[valid]).sum(axis=1)/denominator[valid],-1,1)))
            angles[side+label] = [float(x) if np.isfinite(x) else None for x in values]
    curves = [{'phase':round(i*100/39,1),**{k:v[i] for k,v in angles.items()}} for i in range(40)]
    rom = [{'joint':j,'left':float(np.nanmax(angles['L'+j])-np.nanmin(angles['L'+j])) if any(v is not None for v in angles['L'+j]) else None,'right':float(np.nanmax(angles['R'+j])-np.nanmin(angles['R'+j])) if any(v is not None for v in angles['R'+j]) else None} for j in ['髋','膝','踝']]
    return {'curves':curves,'rom':rom,'skeleton':pose.tolist(),'edges':EDGES,'display_source':'三维坐标的几何夹角；仅用于展示，不代表模型归因。原始缺失坐标按训练规则填 0。','phase_note':'代表性步态周期归一化相位（0–100%），不是实际秒数','radar':None,'radar_note':'原模型未定义统一的雷达评分，暂不生成评分'}
