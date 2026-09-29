"""Associate pose candidates and select a sustained moving foreground subject."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def select_subject(candidates,fps,width,height):
    tracks=[];active=[]
    for frame,poses in enumerate(candidates):
        live=[i for i in active if frame-tracks[i]['last']<=round(fps*.25)]
        centers=[p[[23,24],:2].mean(axis=0)/[width,height] for p in poses]
        used=set()
        if live and poses:
            cost=np.array([[np.linalg.norm(tracks[i]['center']-center) for center in centers] for i in live])
            rr,cc=linear_sum_assignment(cost)
            for r,c in zip(rr,cc):
                if cost[r,c]>.12:continue
                t=tracks[live[r]];t['frames'][frame]=poses[c];t['center']=centers[c];t['last']=frame;used.add(int(c))
        for c,p in enumerate(poses):
            if c not in used:
                tracks.append({'frames':{frame:p},'center':centers[c],'last':frame});live.append(len(tracks)-1)
        active=live
    ranked=[]
    for ident,track in enumerate(tracks):
        positions=np.stack(list(track['frames'].values()))
        visible=(positions[:,[11,12,27,28],2]>.5).all(axis=1)
        if visible.sum()<fps:continue
        p=positions[visible];hip=p[:,[23,24],:2].mean(axis=1);neck=p[:,[11,12],:2].mean(axis=1)
        trunk=np.linalg.norm(neck-hip,axis=1);good=trunk>5
        if good.sum()<fps:continue
        p=p[good];trunk=trunk[good]
        sep=(p[:,27,:2]-p[:,28,:2])/trunk[:,None]
        motion=float(np.linalg.norm(np.percentile(sep,95,axis=0)-np.percentile(sep,5,axis=0)))
        spans=np.ptp(p[:,:,:2],axis=1);area=float(np.median(spans[:,0]*spans[:,1]/(width*height)))
        score=float(area*motion*np.sqrt(len(p)))
        ranked.append({'track':ident,'visible_frames':len(p),'relative_foot_motion':motion,'median_image_area':area,'score':score,
                       'first_frame':min(track['frames']),'last_frame':max(track['frames'])})
    if not ranked:raise ValueError('没有持续全身可见的人物，请完整露出双脚后重拍')
    ranked.sort(key=lambda r:r['score'],reverse=True)
    selected_tracks=[ranked[0]['track']]
    if len(ranked)>1 and ranked[1]['score']>.75*ranked[0]['score']:
        first,second=sorted(ranked[:2],key=lambda r:r['first_frame'])
        gap=second['first_frame']-first['last_frame']-1
        area_ratio=min(first['median_image_area'],second['median_image_area'])/max(first['median_image_area'],second['median_image_area'])
        motion_ratio=min(first['relative_foot_motion'],second['relative_foot_motion'])/max(first['relative_foot_motion'],second['relative_foot_motion'])
        if 0<=gap<=round(fps*3) and area_ratio>.5 and motion_ratio>.5:
            selected_tracks=[first['track'],second['track']]
        else:
            raise ValueError('检测到多个相近的运动主体，无法确认受试者；请保持单人行走后重拍')
    result=np.full((len(candidates),33,3),np.nan,dtype=np.float32)
    for selected in selected_tracks:
        for frame,p in tracks[selected]['frames'].items():result[frame]=p
    return result,{'method':'cross-frame hip association + visible foreground foot-motion score',
                   'selected_track':selected_tracks[0],'selected_tracks':selected_tracks,'candidate_tracks':ranked,
                   'max_link_distance_image_fraction':.12,'max_track_gap_seconds':.25,
                   'ambiguity_ratio':.75,'confirmed_identity':False,
                   'segment_note':'不重叠的主要行走片段按运动幅度和画面尺度合并；未确认人物身份' if len(selected_tracks)>1 else None}
