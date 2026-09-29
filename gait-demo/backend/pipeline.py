"""Video pipeline has an explicit provenance contract; never substitutes 2D for 3D."""
from pathlib import Path
import json
import importlib.util
import cv2

BASE = Path(__file__).resolve().parents[1]
CONTRACT = BASE / 'video_pipeline.json'

def readiness():
    if not CONTRACT.exists():
        if (BASE/'data/recovered-vp3d-verification.json').exists():
            return {'ready':False,'reason':'原 VP3D 网络与权重已恢复并通过三维输出实测；训练坐标统一及步态周期导出尚未核验。','missing':['unified_data_exporter.py','坐标统一、归一化与周期切分一致性验证']}
        return {'ready':False,'reason':'未找到已核验的视频→三维步态处理链路。已有模型与三维数据演示可用；现场视频分类尚未启用。','missing':['videopose3d_19j.py','unified_data_exporter.py','pose_landmarker.task','视频预处理与训练坐标的一致性验证']}
    config = json.loads(CONTRACT.read_text(encoding='utf-8'))
    adapter = Path(config.get('adapter',''))
    if not adapter.is_absolute():adapter=BASE/adapter
    verified = bool(config.get('verified')) and adapter.is_file()
    return {'ready':verified,'research_preview_ready':bool(config.get('research_preview_ready')) and adapter.is_file(),
            'demo_w_grading':bool(config.get('demo_w_grading')) and adapter.is_file(),
            'reason':config.get('note') or '视频预处理适配器尚未通过一致性核验','missing':config.get('missing',[]) if not verified else [],'config':config}

def research_from_cached(output,scenario,progress):
    state=readiness()
    if not state.get('research_preview_ready'):raise RuntimeError('三维科研预览尚未启用')
    adapter=Path(state['config']['adapter'])
    if not adapter.is_absolute():adapter=BASE/adapter
    spec=importlib.util.spec_from_file_location('backend.research_video_adapter',adapter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.research_from_cached(output,scenario,progress)

def probe_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = capture.read()
        if not ok or fps <= 0:
            raise ValueError('无法解码视频，请使用有效的 MP4 / MOV / AVI / WebM 文件')
        if count <= 0:
            count = 1
            while True:
                decoded, _ = capture.read()
                if not decoded:
                    break
                count += 1
                if count/fps > 120.1:
                    raise ValueError('视频超过两分钟，请截取完整行走片段后重试')
        duration = count/fps
        if duration > 120.1:
            raise ValueError('视频超过两分钟，请截取完整行走片段后重试')
        if duration < 1:
            raise ValueError('片段过短，无法包含完整步态周期，请重新录制')
        return {'duration':round(duration,2),'fps':fps,'frames':int(count),'width':frame.shape[1],'height':frame.shape[0]}
    finally:
        capture.release()

def extract(path, output, scenario, progress):
    state = readiness()
    if not state['ready']:
        raise RuntimeError(state['reason'])
    adapter=Path(state['config']['adapter'])
    if not adapter.is_absolute():adapter=BASE/adapter
    spec = importlib.util.spec_from_file_location('backend.verified_video_adapter',adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Adapter returns stride_paths, coordinate_source, valid_ratio, overlay_path.
    result = module.extract(path, output, scenario, progress)
    if not result.get('stride_paths'):
        raise ValueError('未检测到完整有效步态周期，请完整露出下肢并重新录制')
    if not 0 <= result.get('valid_ratio',-1) <= 1:
        raise RuntimeError('姿态质量指标缺失，不能产生结果')
    return result
