from __future__ import annotations
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .models import registry, ROOT, CLASSES
from .metrics import displays
from .gait_profile import build_profile
from . import pipeline, pose_preview

BASE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('GAIT_DEMO_DATA',str(BASE/'data')))
DATA.mkdir(parents=True,exist_ok=True)
MAX_BYTES = 200*1024*1024
SUFFIXES = {'.mp4','.mov','.avi','.webm'}
pool = ThreadPoolExecutor(max_workers=1)
gate = threading.Lock()

@contextmanager
def db():
    connection = sqlite3.connect(DATA/'records.sqlite',timeout=20)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()

def init():
    with db() as c:
        c.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, name TEXT, scenario TEXT, mode TEXT, status TEXT, stage TEXT, error TEXT, created TEXT, started REAL, elapsed REAL, result TEXT)')
        c.execute("UPDATE tasks SET status='failed',error='服务重启中断了分析，请重试' WHERE status IN ('processing','queued')")

@asynccontextmanager
async def lifespan(app):
    init()
    yield

app = FastAPI(title='步态研究 · 本地评估 API',lifespan=lifespan)

@app.middleware('http')
async def local_mutations(request:Request,call_next):
    if request.method not in {'GET','HEAD','OPTIONS'}:
        origin = request.headers.get('origin')
        if origin and origin not in {'http://127.0.0.1:8000','http://localhost:8000','http://127.0.0.1:5173','http://localhost:5173'}:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'仅允许本地工作台操作'},status_code=403)
    return await call_next(request)

def task_row(task_id):
    with db() as c:
        row = c.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
    if not row:
        raise HTTPException(404,'记录不存在')
    item = dict(row)
    item['result'] = json.loads(item['result']) if item['result'] else None
    if item['result'] and item['result'].get('research_version') and 'profile' not in item['result'] and (DATA / task_id / 'research-cycle-provenance.json').is_file():
        try:
            item['result']['profile'] = build_profile(DATA / task_id, item['result']['valid_ratio'])
            update(task_id, result=json.dumps(item['result'], ensure_ascii=False, allow_nan=False))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            logging.warning('profile unavailable for %s: %s', task_id, error)
    if item['status'] in {'queued','processing'}:
        item['elapsed'] = round(time.time()-item['started'],1)
    return item

def update(task_id,**values):
    with db() as c:
        c.execute('UPDATE tasks SET '+','.join(k+'=?' for k in values)+' WHERE id=?',(*values.values(),task_id))

def samples():
    path = BASE/'samples.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

def run_task(task_id,sample=None,video=None):
    started = time.time()
    output = DATA/task_id
    def progress(stage):
        update(task_id,status='processing',stage=stage)
    try:
        item = task_row(task_id)
        progress('读取数据')
        quality, overlay, original, media_info = None,None,None,None
        if sample:
            frames = [pd.read_csv(ROOT/p) for p in sample['strides']]
            source = sample['coordinate_source']
            provenance = '已有三维步态数据 · 当前模型重新推理（不是现场视频）'
        else:
            media_info = pipeline.probe_video(video)
            progress('提取人体姿态')
            video_state = pipeline.readiness()
            if not video_state['ready']:
                preview, overlay_path = pose_preview.extract(video,output,progress)
                result = {**preview,'scenario':item['scenario'],'sample':False,
                          'overlay_url':'/media/'+task_id+'/'+overlay_path.name,
                          'still_url':'/media/'+task_id+'/pose-still.jpg' if (output/'pose-still.jpg').is_file() else None,
                          'video_url':'/media/'+task_id+'/'+video.name,'media':media_info,
                          'limitation':video_state.get('reason') or preview.get('limitation','三维分级尚未核验')}
                if video_state.get('research_preview_ready'):
                    try:
                        research=pipeline.research_from_cached(output,item['scenario'],progress)
                        first=pd.read_csv(research['stride_paths'][0])
                        result.update(displays(first))
                        result.update(stride_count=research['cycle_count'],research_version=research['version'],
                                      phase_note='双脚运动周期相位（0–100%）；不是已验证的足跟接触时刻',
                                      display_source='原 VP3D 权重的三维估计曲线，仅作科研预览；单目投影尺度与分级兼容性未通过验证。')
                        if (output / 'research-cycle-provenance.json').is_file():
                            try:
                                result['profile'] = build_profile(output, preview['valid_ratio'])
                            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
                                logging.warning('profile unavailable for %s: %s', task_id, error)
                        if item['scenario']=='W' and video_state.get('demo_w_grading'):
                            if preview['valid_ratio'] < .5:
                                raise ValueError('有效帧占比不足 50%，请改善光线并完整露出下肢后重拍')
                            progress('调用 W 模型 · 实验性分级')
                            frames=[pd.read_csv(p) for p in research['stride_paths']]
                            prediction=registry.predict('W',frames,'VP3D-19J')
                            result.update(prediction,kind='video_experimental',classes=CLASSES,
                                coordinate_source='VP3D-19J',
                                provenance='本次视频 · MediaPipe 二维姿态 → 原 VP3D 权重 → 原型周期处理 → 指定 W ExtraTrees 实际推理',
                                preprocessing_note='中期演示原型：周期边界与原训练数据尚未一致性验证',
                                probability_note='模型输出概率未经临床校准；本视频链路的准确率尚未验证',
                                quality_note='有效姿态帧 / 总视频帧',
                                classification_status='experimental',
                                limitation='实验性模型输出，仅展示当前流程可运行；不能作为已验证步态等级或疾病诊断。')
                    except Exception as error:
                        logging.warning('research preview unavailable for %s: %s',task_id,error)
                        result['research_error']=str(error)
                stage=('W 模型推理完成 · 实验性结果' if result.get('classification_status')=='experimental' else
                       '运动科研预览完成 · 分级尚未核验' if result.get('research_version') else '二维姿态预览完成 · 分级尚未核验')
                update(task_id,status='completed',stage=stage,
                       elapsed=round(time.time()-started,2),result=json.dumps(result,ensure_ascii=False,allow_nan=False))
                return
            extracted = pipeline.extract(video,output,item['scenario'],progress)
            frames = [pd.read_csv(p) for p in extracted['stride_paths']]
            source = extracted['coordinate_source']
            quality = extracted['valid_ratio']
            if quality < 0.5:
                raise ValueError('有效帧占比不足 50%，请改善光线、完整露出下肢后重拍')
            overlay_path = extracted.get('overlay_path')
            if overlay_path:
                safe = Path(overlay_path).resolve()
                if output.resolve() not in safe.parents:
                    raise RuntimeError('骨架视频输出必须位于当前任务目录')
                overlay = '/media/'+task_id+'/'+safe.name
            original = '/media/'+task_id+'/'+video.name
            provenance = '本次上传或录制视频 · 原模型真实推理 · 重建预处理 v1'
        progress('处理步态周期')
        progress('模型推理')
        prediction = registry.predict(item['scenario'],frames,source)
        progress('生成评估结果')
        result = {**prediction,**displays(frames[0]),'classes':CLASSES,'scenario':item['scenario'],'stride_count':len(frames),'valid_ratio':quality,'coordinate_source':source,'provenance':provenance,'overlay_url':overlay,'video_url':original,'media':media_info,'probability_note':'模型输出概率未经临床校准，不代表诊断准确率','quality_note':'已有步态 CSV 未保存视频有效帧占比' if sample else '有效姿态帧 / 总视频帧','sample':bool(sample)}
        if not sample:
            result['preprocessing_note']=extracted.get('preprocessing_note','')
            result['exporter_version']=extracted.get('exporter_version','')
        update(task_id,status='completed',stage='分析完成',elapsed=round(time.time()-started,2),result=json.dumps(result,ensure_ascii=False,allow_nan=False))
    except Exception as error:
        logging.exception('task %s failed',task_id)
        update(task_id,status='failed',stage='分析失败',error=str(error),elapsed=round(time.time()-started,2))
    finally:
        gate.release()

def create(name,scenario,mode):
    task_id = uuid.uuid4().hex
    (DATA/task_id).mkdir()
    with db() as c:
        c.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)',(task_id,name,scenario,mode,'queued','等待分析',None,datetime.now().isoformat(timespec='seconds'),time.time(),0,None))
    return task_id

@app.get('/api/health')
def health():
    return {'service':'online','routes':registry.manifest.get('routes',{}),'video':{**pipeline.readiness(),'preview_ready':pose_preview.available()},'busy':gate.locked()}

@app.get('/api/samples')
def get_samples():
    return [{k:v for k,v in s.items() if k!='strides'} for s in samples()]

@app.post('/api/samples/{sample_id}/analyze',status_code=202)
def analyze_sample(sample_id:str):
    sample = next((s for s in samples() if s['id']==sample_id),None)
    if not sample:
        raise HTTPException(404,'样例不存在')
    if not gate.acquire(blocking=False):
        raise HTTPException(409,'本机正在分析一个任务，请稍后再试')
    try:
        task_id = create(sample['name'],sample['scenario'],'sample')
        pool.submit(run_task,task_id,sample,None)
    except Exception:
        gate.release()
        raise
    return {'id':task_id}

@app.post('/api/tasks',status_code=202)
async def analyze(file:UploadFile=File(...),scenario:str=Form('W')):
    if scenario not in {'W','OW','WT'}:
        raise HTTPException(400,'未知行走场景')
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415,'支持 MP4、MOV、AVI、WebM 视频')
    if not gate.acquire(blocking=False):
        raise HTTPException(409,'本机正在分析一个任务，请稍后再试')
    task_id = None
    try:
        task_id = create(Path(file.filename or '视频').name,scenario,'video')
        path = DATA/task_id/('original'+suffix)
        total = 0
        with path.open('wb') as target:
            while chunk := await file.read(1024*1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413,'文件超过 200 MB')
                target.write(chunk)
        if total == 0:
            raise HTTPException(400,'视频为空')
        pool.submit(run_task,task_id,None,path)
        return {'id':task_id}
    except Exception as error:
        if task_id:
            update(task_id,status='failed',stage='上传失败',error=str(getattr(error,'detail',error)))
            shutil.rmtree(DATA/task_id)
        gate.release()
        raise
    finally:
        await file.close()

@app.get('/api/tasks/{task_id}')
def get_task(task_id:str):
    return task_row(task_id)

@app.get('/api/history')
def history():
    with db() as c:
        rows = c.execute('SELECT id,name,scenario,mode,status,stage,error,created,elapsed FROM tasks ORDER BY created DESC LIMIT 200').fetchall()
    return [dict(r) for r in rows]

@app.delete('/api/tasks/{task_id}')
def delete_task(task_id:str):
    item = task_row(task_id)
    if item['status'] in {'queued','processing'}:
        raise HTTPException(409,'分析中不能删除')
    if not re.fullmatch('[0-9a-f]{32}',task_id):
        raise HTTPException(400,'无效记录 ID')
    folder = (DATA/task_id).resolve()
    if folder.parent != DATA.resolve():
        raise HTTPException(400,'无效路径')
    if folder.exists():
        shutil.rmtree(folder)
    with db() as c:
        c.execute('DELETE FROM tasks WHERE id=?',(task_id,))
    return {'ok':True}

@app.get('/media/{task_id}/{filename}')
def media(task_id:str,filename:str):
    task_row(task_id)
    if not re.fullmatch('[0-9a-f]{32}',task_id) or Path(filename).name != filename:
        raise HTTPException(404,'文件不存在')
    path = DATA/task_id/filename
    if not path.is_file():
        raise HTTPException(404,'文件不存在')
    return FileResponse(path)

DIST = BASE/'frontend/dist'
if DIST.exists():
    app.mount('/',StaticFiles(directory=DIST,html=True),name='frontend')
