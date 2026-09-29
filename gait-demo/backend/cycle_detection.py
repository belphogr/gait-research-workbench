"""Image-plane gait events, independent of classifier labels and 3D axis guesses.

Alternating foot separation cancels camera/body translation. PCA finds the
visible movement direction. Events are phase extrema, NOT force-plate-confirmed
heel strikes. This module must not supply a diagnosis or alter model outputs.
"""
import numpy as np
from scipy.signal import find_peaks,savgol_filter

VERSION='image-foot-phase-v2'

def detect_cycles(keypoints,valid,fps):
    x=np.asarray(keypoints,float);valid=np.asarray(valid,bool)
    if x.ndim!=3 or x.shape[1:]!=(33,3) or valid.shape!=(len(x),):raise ValueError('二维关节格式错误')
    if not np.isfinite(fps) or fps<=0:raise ValueError('帧率无效')
    visible=valid & np.isfinite(x[:,[11,12,23,24,27,28],:2]).all(axis=(1,2))
    visible &= (x[:,27,2]>.5)&(x[:,28,2]>.5)
    if visible.sum()<fps:raise ValueError('双脚有效可见时间不足一秒')
    hip=x[:,[23,24],:2].mean(axis=1);neck=x[:,[11,12],:2].mean(axis=1)
    trunk=float(np.median(np.linalg.norm(neck[visible]-hip[visible],axis=1)))
    if trunk<=1e-6:raise ValueError('二维躯干尺度无效')
    separation=(x[:,27,:2]-x[:,28,:2])/trunk
    # Fit the axis only on visible frames. Interpolation is used for filtering;
    # intervals with long gaps still fail explicit quality checks below.
    times=np.arange(len(x));good=np.flatnonzero(visible)
    for axis in range(2):separation[:,axis]=np.interp(times,good,separation[good,axis])
    centered=separation[visible]-np.median(separation[visible],axis=0)
    _,_,vectors=np.linalg.svd(centered,full_matrices=False);direction=vectors[0]
    if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    signal=separation@direction
    window=max(3,int(round(fps*.15))|1)
    if len(signal)>=window:signal=savgol_filter(signal,window,2)
    amplitude=float(np.percentile(signal[visible],95)-np.percentile(signal[visible],5))
    if amplitude<.1:raise ValueError('双脚相对运动不足，不能确认行走周期')
    events=[];cycles=[]
    for phase,sign in [('positive',1),('negative',-1)]:
        indices,props=find_peaks(sign*signal,distance=max(1,int(round(fps*.4))),prominence=max(.05,amplitude*.2))
        indices=[int(i) for i in indices if visible[i]]
        events.extend({'frame':i,'phase':phase} for i in indices)
        for start,end in zip(indices,indices[1:]):
            duration=(end-start)/fps
            if duration<.5-1e-4 or duration>2.5+1e-4:continue
            interval=visible[start:end+1];gap=maximum=0
            for value in interval:
                gap=0 if value else gap+1;maximum=max(maximum,gap)
            if interval.mean()<.7 or maximum/fps>.25:continue
            cycles.append({'start':start,'end':end,'phase':phase,'duration':duration,'valid_ratio':float(interval.mean())})
    cycles.sort(key=lambda c:(c['start'],c['phase']))
    return cycles,{'version':VERSION,'events':sorted(events,key=lambda e:e['frame']),
                   'axis_image_xy':direction.tolist(),'robust_amplitude':amplitude,
                   'event_definition':'extrema of PCA-projected left-right ankle separation, not physical heel strike',
                   'prominence':max(.05,amplitude*.2),'fps':float(fps),
                   'quality':'both ankles visibility >0.5; valid interval >=0.7; gap <=0.25s',
                   'duration_seconds':[.5,2.5],'duration_rounding_tolerance_seconds':1e-4}
