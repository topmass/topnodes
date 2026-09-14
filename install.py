"""Apply the supported VideoHelperSuite timing fix without changing other code."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile


def patch_video_loader(source):
    tree = ast.parse(source)
    replacements = []
    lines = source.splitlines(keepends=True)
    for name in ('ffmpeg_frame_generator', 'LoadVideoFFmpegUpload'):
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name), None)
        if node is None:
            raise ValueError(f'Unsupported VideoHelperSuite: missing {name}. No files changed.')
        block = ''.join(lines[node.lineno - 1:node.end_lineno])
        updated = block
        if name == 'ffmpeg_frame_generator':
            edits = [
                ('meta_batch=None, unique_id=None):', 'meta_batch=None, unique_id=None, fps_rounding="near"):'),
                ('vfilters.append("fps=fps="+str(force_rate))', 'vfilters.append("fps=fps="+str(force_rate)+":round="+fps_rounding)'),
                ('args_all_frames += ["-f", "rawvideo", "-"]', 'args_all_frames += ["-fps_mode", "passthrough", "-f", "rawvideo", "-"]'),
            ]
            for old, new in edits:
                if new in updated:
                    continue
                if updated.count(old) != 1:
                    raise ValueError('VideoHelperSuite code differs from the supported version. No files changed; see TopNodes README.')
                updated = updated.replace(old, new, 1)
        elif '"fps_rounding"' not in updated:
            marker = '"optional": {'
            if updated.count(marker) != 1:
                raise ValueError('Unsupported VideoHelperSuite loader inputs. No files changed.')
            updated = updated.replace(marker, marker + '\n                    "fps_rounding": (["near", "up", "down"], {"default": "near"}),', 1)
        replacements.append((block, updated))
    for old, new in replacements:
        source = source.replace(old, new, 1)
    compile(source, 'VideoHelperSuite timing patch', 'exec')
    return source


def install_timing_fix(custom_nodes=None):
    root = Path(__file__).resolve().parent
    custom_nodes = Path(custom_nodes) if custom_nodes is not None else root.parent
    loaders = [p / 'videohelpersuite' / 'load_video_nodes.py' for p in custom_nodes.iterdir() if p.is_dir() and p.name.lower() in ('comfyui-videohelpersuite', 'comfyui_videohelpersuite')]
    if not loaders:
        print('[TopNodes] Install ComfyUI-VideoHelperSuite, then restart ComfyUI to enable the timing fix.')
    for path in loaders:
        original = path.read_bytes()
        updated = patch_video_loader(original.decode('utf-8')).encode('utf-8')
        if updated == original:
            continue
        backup = path.with_name(path.name + '.topnodes-' + hashlib.sha256(original).hexdigest()[:12] + '.bak')
        if not backup.exists():
            backup.write_bytes(original)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.topnodes-', delete=False) as pending:
            pending.write(updated)
            temporary = Path(pending.name)
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
        print(f'[TopNodes] Applied VideoHelperSuite timing fix. Original saved as {backup.name}.')


if __name__ == '__main__':
    try:
        install_timing_fix()
    except (OSError, ValueError, SyntaxError) as error:
        print(f'[TopNodes] Timing fix could not be applied: {error}')
