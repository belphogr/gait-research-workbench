"""Training-contract VP3D inference; an explicit replacement for legacy inference.

Delivered TRC training uses hip-centered/trunk-scaled orthographic XY and
collate_sequences produces N×38×1, not 1×38×T. A camera's projected trunk
approximates the training 3D scale; this remaining approximation is documented.
The original TRC axis COMMENTS conflict with measured marker coordinates.
Observed mode follows measured Z-up TRCs; comment mode is retained for audit.
"""
import numpy as np

VERSION='vp3d-trc-frame-contract-v3'

def normalize_image_pose(mapped,valid):
    xy=np.asarray(mapped,float)[...,:2].copy();valid=np.asarray(valid,bool)
    if xy.ndim!=3 or xy.shape[1:]!=(19,2) or valid.shape!=(len(xy),):raise ValueError('19关节二维输入格式错误')
    finite=np.isfinite(xy).all(axis=(1,2));good=finite&valid
    if good.sum()<10:raise ValueError('有效二维人体帧不足')
    idx=np.arange(len(xy));times=np.flatnonzero(good)
    for j in range(19):
        for c in range(2):xy[:,j,c]=np.interp(idx,times,xy[good,j,c])
    scale=float(np.median(np.linalg.norm(xy[good,1]-xy[good,8],axis=1)))
    if scale<1e-6:raise ValueError('投影躯干尺度无效')
    return ((xy-xy[:,8:9])/scale).astype(np.float32)

def infer_training_contract(network,mapped,valid,batch_size=128,projection_mode='comment'):
    import torch
    xy=normalize_image_pose(mapped,valid)
    if projection_mode=='observed_trc':
        # Actual TRC: Z up, Y lateral; prepare_trc_training projects
        # u=TRC Z (up), v=-TRC Y (lateral). Camera XY must follow that.
        xy=-xy[:,:,[1,0]].copy()
    elif projection_mode!='comment':raise ValueError('未知投影坐标版本')
    outputs=[];network.eval()
    with torch.inference_mode():
        for start in range(0,len(xy),batch_size):
            # Exact ordering and singleton time dimension of finetune collate.
            inputs=torch.from_numpy(xy[start:start+batch_size]).reshape(-1,38,1)
            pred=network(inputs)
            if pred.shape!=(len(inputs),57,1):raise RuntimeError('VP3D 训练图输出形状不符')
            outputs.append(pred[:,:,0].numpy().reshape(-1,19,3))
    raw=np.concatenate(outputs)
    if not np.isfinite(raw).all():raise RuntimeError('VP3D 输出非有限坐标')
    return raw

def training_axes_to_unified(raw,projection_mode='comment'):
    # VP internal left-right/up/forward -> CSV forward/left-right/up.
    # This follows TRC preparation, not the old malformed prediction's neck axis.
    if projection_mode=='observed_trc':return np.asarray(raw)[:,:,[2,1,0]].copy()
    return np.asarray(raw)[:,:,[2,0,1]].copy()
