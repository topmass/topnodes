"""Run with Python; no ComfyUI or GPU required."""
from pathlib import Path
import runpy
import shutil
import tempfile
from install import install_timing_fix, patch_video_loader

original = '''def ffmpeg_frame_generator(video, force_rate, frame_load_cap, start_time,
                           custom_width, custom_height, downscale_ratio=8,
                           meta_batch=None, unique_id=None):
    vfilters = []
    vfilters.append("fps=fps="+str(force_rate))
    args_all_frames = []
    args_all_frames += ["-f", "rawvideo", "-"]

class LoadVideoFFmpegUpload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"meta_batch": ("VHS_BatchManager",)}}
'''
patched = patch_video_loader(original)
assert 'fps_rounding="near"' in patched and ':round=' in patched
assert '"-fps_mode", "passthrough"' in patched
assert patch_video_loader(patched) == patched
partial = patched.replace('"-fps_mode", "passthrough", ', '')
assert patch_video_loader(partial) == patched
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    loader = root / 'ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py'
    loader.parent.mkdir(parents=True)
    loader.write_text(original)
    install_timing_fix(root)
    assert loader.read_text() == patched
    backups = list(loader.parent.glob('*.bak'))
    assert len(backups) == 1 and backups[0].read_text() == original
    timestamp = loader.stat().st_mtime_ns
    install_timing_fix(root)
    assert loader.stat().st_mtime_ns == timestamp
    incompatible = original.replace('args_all_frames +=', 'args_all_frames = args_all_frames +')
    loader.write_text(incompatible)
    try:
        install_timing_fix(root)
    except (ValueError, SyntaxError):
        pass
    else:
        raise AssertionError('Unsupported source was accepted')
    assert loader.read_text() == incompatible
    assert len(list(loader.parent.glob('*.bak'))) == 1
    loader.write_text(original)
    pack = root / 'topnodes'
    pack.mkdir()
    for name in ('install.py', 'prestartup_script.py'):
        shutil.copyfile(Path(__file__).with_name(name), pack / name)
    runpy.run_path(str(pack / 'prestartup_script.py'))
    assert loader.read_text() == patched
print('PASS: fresh install, old patch upgrade, repeat startup, exact backup, and unsupported-source protection')
