"""Original VP3D + explicitly reconstructed exporter. No classifier substitute."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from . import pose_preview
from .models import original_definitions,sha256
from .unified_data_exporter import vp3d_keypoints_to_unified,process_and_export,VERSION

BASE=Path(__file__).resolve().parents[1]
CODE=BASE/'recovered-vp3d/代码'
WEIGHT=BASE/'recovered-vp3d/模型/vp3d_19j_finetuned_best.pth'
EXPECTED_WEIGHT='85180fb26ce0a6f5f6f3a930d9292a6bf5f5dc9a734eddb4f46e6024d89917f1'
EXPECTED_CODE='de24dfa857f5bb1eaa56ecf08bae7cc5bfd0ea100e43583b499d36e029b9435f'

def research_from_cached(output_dir,scenario,progress):
    """Repaired pose/cycle output, explicitly not approved classifier input."""
    import pandas as pd
    from .cycle_detection import detect_cycles
    from .vp3d_contract import infer_training_contract,training_axes_to_unified
    from .unified_data_exporter import normalize_3d,resample_stride
    from .models import COLUMNS
    output_dir=Path(output_dir)
    stored=np.load(output_dir/'landmarks-2d.npz');x=stored['keypoints'];valid=stored['valid']
    if sha256(WEIGHT)!=EXPECTED_WEIGHT or sha256(CODE/'videopose3d_19j.py')!=EXPECTED_CODE:
        raise RuntimeError('VP3D 权重或网络源码发生变化，需重新核验')
    progress('处理双脚运动周期')
    records,notes=detect_cycles(x,valid,float(stored['fps']))
    if not records:raise ValueError('未找到通过质量要求的完整双脚运动周期')
    sys.path.insert(0,str(CODE))
    from videopose3d_19j import TemporalModel19J,mp33_to_19j
    network=TemporalModel19J()
    network.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    progress('三维姿态重建 · 科研预览')
    raw=infer_training_contract(network,mp33_to_19j(x),valid,projection_mode='observed_trc')
    np.save(output_dir/'research-vp3d.npy',raw)
    norm=normalize_3d(training_axes_to_unified(raw,projection_mode='observed_trc'))
    for i,r in enumerate(records,1):
        name=f'research_stride_{i:03d}.csv'
        pd.DataFrame(resample_stride(norm[r['start']:r['end']+1]),columns=COLUMNS).to_csv(output_dir/name,index=False)
        r['file']=name
    notes.update(cycles=records,vp3d_version='vp3d-observed-trc-frame-v4',coordinate_permutation=[2,1,0],
                 input_contract='hip-centered, projected trunk scale, observed TRC u=up/v=-lateral, N×38×1',
                 classification_verified=False,scenario=scenario,weight_sha256=EXPECTED_WEIGHT,
                 limitation='二维投影躯干近似三维训练尺度；单目投影和分级兼容性尚未通过验证')
    (output_dir/'research-cycle-provenance.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf8')
    return {'stride_paths':[output_dir/r['file'] for r in records],'cycle_count':len(records),
            'version':'vp3d-observed-trc-frame-v4','classification_verified':False}

def extract(video_path,output_dir,scenario,progress):
    output_dir=Path(output_dir)
    metadata,overlay=pose_preview.extract(video_path,output_dir,progress)
    if metadata['valid_ratio']<.5:
        raise ValueError('有效帧不足50%，请完整露出身体并改善光线后重拍')
    stored=np.load(output_dir/'landmarks-2d.npz')
    x=stored['keypoints'].copy();valid=stored['valid'];fps=float(stored['fps'])
    good=~np.isnan(x[:,0,0])
    if good.sum()<10:raise ValueError('有效人体帧不足')
    # Same missing-frame interpolation as the delivered batch_19j_ws.py.
    idx=np.arange(len(x))
    for j in range(33):
        for c in range(3):x[:,j,c]=np.interp(idx,idx[good],x[good,j,c])
    hips=(x[:,23,:2]+x[:,24,:2])/2
    shoulders=(x[:,11,:2]+x[:,12,:2])/2
    trunk=float(np.median(np.linalg.norm(shoulders-hips,axis=1)))
    relative=(x[:,[27,28],:2]-hips[:,None,:])/max(trunk,1e-6)
    if np.max(np.ptp(relative[valid],axis=0))<.05:
        raise ValueError('未检测到足够下肢运动，请录制完整行走而非静止站立')
    progress('三维姿态重建')
    if sha256(WEIGHT)!=EXPECTED_WEIGHT or sha256(CODE/'videopose3d_19j.py')!=EXPECTED_CODE:
        raise RuntimeError('VP3D 权重或原始网络源码已改变，请重新核验')
    sys.path.insert(0,str(CODE))
    from videopose3d_19j import TemporalModel19J,mp33_to_19j,normalize_screen_coordinates
    network=TemporalModel19J()
    network.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    network.eval()
    ns={'np':np,'torch':torch,'mp33_to_19j':mp33_to_19j,'normalize_screen_coordinates':normalize_screen_coordinates}
    original_definitions(CODE/'pipeline_19j.py',['infer_3d_19j_pipeline'],ns)
    raw=ns['infer_3d_19j_pipeline'](network,x,metadata['width'],metadata['height'])
    np.save(output_dir/'raw-vp3d.npy',raw)
    progress('处理步态周期')
    _,_,records=process_and_export(vp3d_keypoints_to_unified(raw),'local',scenario,
                                  output_dir=output_dir,fps=fps,valid=valid)
    if not records:raise ValueError('未检测到完整有效步态周期，请全身入镜并连续行走后重拍')
    return {'stride_paths':[output_dir/r['file'] for r in records],
            'coordinate_source':'VP3D-19J','valid_ratio':metadata['valid_ratio'],
            'overlay_path':overlay,'exporter_version':VERSION,
            'preprocessing_note':'原 VP3D 网络与权重 + 新补周期处理 v1；训练输入格式已核验，旧周期边界未完全复现。',
            'cycle_count':len(records)}
