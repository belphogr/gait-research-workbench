"""Adapters for the models present at the other task's initial handoff.

Never trains, selects on test scores, or loads a person-level tree.
"""
from __future__ import annotations
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'models'
TREE = FRAMEWORK / 'tree_model_w'
JOINTS = ['Head','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist','MidHip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','RHeel','RToe','LHeel','LToe']
COLUMNS = [f'{j}_{a}' for j in JOINTS for a in 'xyz']
CLASSES = ['L0','L1','L2']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def original_definitions(path, names, namespace):
    """Execute exact original definitions without their training-only imports."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('原始模型代码缺少必要定义')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def validate_stride(frame):
    if list(frame.columns) != COLUMNS:
        raise ValueError('步态数据必须使用训练时相同的 19 关节 xyz 列及顺序')
    values = frame.fillna(0).to_numpy(dtype=np.float64)
    if values.shape != (40, 57) or not np.isfinite(values).all():
        raise ValueError('需要一个完整的 40 帧、57 列、无无穷值步态周期')
    if np.max(np.ptp(values, axis=0)) < 1e-7:
        raise ValueError('未检测到有效步态运动，请重新录制')

class ModelRegistry:
    def __init__(self):
        self.loaded = {}
        self.manifest_path = Path(__file__).resolve().parents[1] / 'model_manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8')) if self.manifest_path.exists() else {'routes': {}}

    def load(self, scenario):
        if scenario in self.loaded:
            return self.loaded[scenario]
        route = self.manifest['routes'].get(scenario)
        if not route:
            raise RuntimeError(f'{scenario} 模型尚未核验，请运行 scripts/audit_models.py')
        for source, digest in self.manifest.get('source_sha256',{}).items():
            if sha256(ROOT/source) != digest:
                raise RuntimeError('原始预处理或模型源码已改变，请重新核验后再启动')
        path = ROOT / route['weight']
        if sha256(path) != route['weight_sha256']:
            raise RuntimeError('模型权重已改变，请重新核验，禁止静默更换版本')
        if scenario == 'W':
            import joblib, sklearn
            if sklearn.__version__ != '1.8.0':
                raise RuntimeError('W 原始树模型需要 scikit-learn 1.8.0')
            sys.path.insert(0, str(TREE / 'scripts'))
            import run_w_xy_plus_calibrated_z_validation as prepared
            artifact = joblib.load(path)
            if artifact['model'].n_features_in_ != 10935 or list(artifact['model'].classes_) != [0,1,2]:
                raise RuntimeError('树模型不是交付时的 stride ExtraTrees')
            model = (artifact, prepared)
        else:
            import math, torch
            from torch import nn
            from torch.autograd import Variable
            config_path = ROOT / route['config']
            if sha256(config_path) != route['config_sha256']:
                raise RuntimeError('模型配置已改变，请重新核验')
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
            ns = {'torch':torch,'nn':nn,'math':math,'Variable':Variable}
            original_definitions(FRAMEWORK/'ml_utils/positional_encoding.py', ['PositionalEncoder'], ns)
            original_definitions(FRAMEWORK/'ml_utils/cnn1d_model.py', ['ConvBlock','DenseBlock','CNN1D'], ns)
            params = {k.removeprefix('param_net__module__'):v[0] for k,v in config.items() if k.startswith('param_net__module__')}
            if params['in_chans'] != 243 or params['time_steps'] != 45 or config.get('normalization_mode','per_stride') != 'per_stride':
                raise RuntimeError('当前适配器只支持原始 integrate 45×243 输入')
            network = ns['CNN1D'](**params)
            network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            network.eval()
            ns.update({'np':np,'pd':pd,'KINEMATIC_JOINTS':JOINTS,'KINEMATIC_PAIRS':[(JOINTS[r],JOINTS[l]) for r,l in [(2,5),(3,6),(4,7),(9,12),(10,13),(11,14),(15,17),(16,18)]]})
            original_definitions(FRAMEWORK/'ml_utils/gait_data_loader.py',['add_kinematic_features'],ns)
            model = (network, ns['add_kinematic_features'])
        self.loaded[scenario] = model
        return model

    def predict(self, scenario, frames, source):
        for frame in frames:
            validate_stride(frame)
        model = self.load(scenario)
        if scenario == 'W':
            artifact, prepared = model
            if source not in artifact['z_calibration']['per_source']:
                raise ValueError('坐标来源未经原始树模型校准，不能推理')
            cal = artifact['z_calibration']['per_source'][source]
            x = np.stack([np.concatenate((prepared.xy.xy_features(prepared.xy.canonicalize_xy(f)), (prepared.relative_z_features(f)-cal['center'])/cal['iqr'])) for f in frames])
            probabilities = artifact['model'].predict_proba(x)
        else:
            import torch
            network, features = model
            route = self.manifest['routes'][scenario]
            if route.get('frame_count_contract') != 'constant_40_verified':
                raise RuntimeError('原始帧数标准化参数未确认，禁止猜测模型输入')
            filled = [f.fillna(0) for f in frames]  # Exactly as original GaitDataset.
            x = np.stack([features((f-f.mean())/(f.std()+1e-6),include_z=True).to_numpy(dtype=np.float32) for f in filled])
            with torch.inference_mode():
                probabilities = torch.softmax(network(torch.from_numpy(x), torch.zeros(len(x))),dim=1).numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError('模型产生非有限结果')
        # Original W delivery uses arithmetic mean of stride probabilities.
        # The demo uses the same explicit pooling for each acquired video.
        mean = probabilities.mean(axis=0)
        return {'grade':int(mean.argmax()),'probabilities':mean.tolist(),'stride_probabilities':probabilities.tolist(),'aggregation':'逐步态周期概率算术均值','model':self.manifest['routes'][scenario]}

registry = ModelRegistry()
