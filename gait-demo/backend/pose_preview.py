"""Local MediaPipe extraction with explicit subject tracking; no classifier."""
from pathlib import Path
import sys,json,hashlib
import cv2
import numpy as np
from .subject_tracking import select_subject

BASE=Path(__file__).resolve().parents[1]
ASSET=BASE/'assets/pose_landmarker_full.task'
VENDOR=BASE/'pose_vendor'
EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
       (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32),(27,31),(28,32)]

def available():return ASSET.is_file() and (VENDOR/'mediapipe').is_dir()

def extract(path,output,progress):
    if not available():raise RuntimeError('本地 MediaPipe 视频姿态组件尚未安装')
    sys.path.insert(0,str(VENDOR))
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    output=Path(output);cap=cv2.VideoCapture(str(path))
    fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or width<=0 or height<=0:
        cap.release();raise ValueError('无法解码视频')
    options=vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=ASSET.read_bytes()),
                                        running_mode=vision.RunningMode.IMAGE,num_poses=4)
    candidates=[]
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if len(candidates)/fps>120.1:raise ValueError('实际解码视频超过两分钟，请截取后重试')
                result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                candidates.append([np.array([[p.x*width,p.y*height,p.visibility] for p in pose],dtype=np.float32) for pose in result.pose_landmarks])
                if len(candidates)%30==0:progress(f'提取二维姿态 · 已处理 {len(candidates)} 帧')
    finally:cap.release()
    if not candidates:raise ValueError('视频没有有效帧')
    progress('确认持续行走主体')
    points,selection=select_subject(candidates,fps,width,height)
    valid=np.sum(points[:,[11,12,27,28],2]>.5,axis=1)>=3
    if valid.sum()<10:raise ValueError('选定主体有效人体帧不足，请保持单人全身入镜后重拍')
    padded=np.full((len(candidates),4,33,3),np.nan,dtype=np.float32)
    for i,poses in enumerate(candidates):
        for j,pose in enumerate(poses):padded[i,j]=pose
    np.savez_compressed(output/'pose-candidates.npz',keypoints=padded,fps=fps)
    (output/'subject-selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(output/'landmarks-2d.npz',keypoints=points,valid=valid,fps=fps)
    progress('生成骨架预览')
    overlay=output/'pose-preview.webm';cap=cv2.VideoCapture(str(path))
    writer=cv2.VideoWriter(str(overlay),cv2.VideoWriter_fourcc(*'VP80'),fps,(width,height))
    try:
        if not writer.isOpened():raise RuntimeError('无法生成浏览器可播放的 VP8 骨架视频')
        for frame_index,selected in enumerate(points):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('视频再次解码时帧数不一致')
            if np.isfinite(selected).all():
                for a,b in EDGES:
                    if min(selected[a,2],selected[b,2])>.5:
                        cv2.line(frame,tuple(selected[a,:2].astype(int)),tuple(selected[b,:2].astype(int)),(137,139,22),3,cv2.LINE_AA)
                for x,y,v in selected:
                    if v>.5:cv2.circle(frame,(int(x),int(y)),4,(240,250,250),-1,cv2.LINE_AA)
            if frame_index==len(points)//2:
                encoded,image=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
                if encoded:(output/'pose-still.jpg').write_bytes(image.tobytes())
            writer.write(frame)
    finally:cap.release();writer.release()
    metadata={'kind':'pose_preview','dimension':'2D','coordinate_system':'image pixels',
              'frames':len(points),'valid_frames':int(valid.sum()),'valid_ratio':float(valid.mean()),
              'fps':fps,'width':width,'height':height,'grade':None,'probabilities':None,
              'pose_model':'MediaPipe Pose Landmarker Full float16 v1 · 多候选主体跟踪',
              'pose_weight_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),'subject_selection':selection,
              'quality_note':'持续行走主体跟踪；左肩、右肩、左踝、右踝至少三个点 visibility >0.5；身份及三维输入质量未独立确认',
              'provenance':'本次视频 · 多候选姿态检测与主体跟踪；尚未进入步态分级模型',
              'limitation':'运动科研预览；视频分级兼容性尚未通过验证，不生成等级。'}
    (output/'pose-preview.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    return metadata,overlay
