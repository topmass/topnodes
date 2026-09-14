"""Run from ComfyUI root: python custom_nodes/topnodes/check_video_loader.py"""
import ast
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
import torch

ffmpeg_path = 'ffmpeg'
ENCODE_ARGS = ('utf-8', 'replace')
class ProgressBar:
    def __init__(self, *args): pass
    def update(self, *args): pass
    def update_absolute(self, *args): pass

path = Path('custom_nodes/comfyui-videohelpersuite/videohelpersuite/load_video_nodes.py')
if not path.exists():
    path = Path('custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py')
for node in ast.parse(path.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('target_size', 'ffmpeg_frame_generator'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'))
with tempfile.TemporaryDirectory() as directory:
    video = str(Path(directory) / 'offset.mkv')
    subprocess.run(['ffmpeg', '-v', 'error', '-itsoffset', '0.0625', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=24:duration=1', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', '1.0625', '-fps_mode', 'passthrough', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', video], check=True)
    frames = ffmpeg_frame_generator(video, 24, 0, 0, 64, 64, fps_rounding='near')
    next(frames)
    frames = list(frames)
    assert len(frames) == 24, f'Loader duplicated frames: {len(frames)} instead of 24'
    assert np.abs(frames[0] - frames[1]).mean() > .001
print('PASS: offset timestamps do not create extra opening frames')
