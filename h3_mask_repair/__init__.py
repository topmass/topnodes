import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
import torch
import torch.nn.functional as F

import folder_paths
from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack, SAM3_TrackToMask


def repair_signature(images, masks):
    digest = hashlib.sha256()
    for batch in (images, masks):
        digest.update(str((tuple(batch.shape), batch.dtype)).encode())
        for frame in batch:
            digest.update(memoryview(frame.detach().cpu().contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def apply_mask_repairs(images, masks, steps, segment, track):
    result = masks.clone()
    count, height, width = masks.shape
    for step in steps:
        first, last = int(step['frame']), int(step['end'])
        if not 0 <= first <= last < count:
            raise ValueError('Repair range is outside the loaded video. Clear the old repairs.')
        if step.get('restore'):
            if first != last:
                raise ValueError('Reset applies to one frame only.')
            result[first] = masks[first]
            continue
        positive, negative = step.get('positive', []), step.get('negative', [])
        erase_points = step.get('erase_points', [])
        erase_only = step.get('erase_only', False)
        if erase_only and (positive or negative or step.get('replace') or any(s['mode'] != 'erase' for s in step.get('strokes', []))):
            raise ValueError('Use Add and Erase as separate repairs.')
        for point in positive + negative + erase_points:
            if not (0 <= point['x'] <= 1 and 0 <= point['y'] <= 1):
                raise ValueError('Repair points must be inside the image.')
        if negative and not positive:
            raise ValueError('Add a green point inside the target before using red exclude points, or use Erase.')
        seed = torch.zeros_like(result[first]) if step.get('replace') else result[first].clone()
        if erase_only and not seed.any():
            continue
        if positive:
            coords = lambda points: [{'x': min(width - 1, round(p['x'] * width)), 'y': min(height - 1, round(p['y'] * height))} for p in points]
            seed = segment(images[first:first + 1], coords(positive), coords(negative))[0].to(result)
        if erase_points:
            if not erase_only:
                raise ValueError('Mask removal points require Erase mode.')
            regions, _ = label((seed > 0).detach().cpu().numpy(), structure=np.ones((3, 3)))
            selected = {regions[min(height - 1, round(p['y'] * height)), min(width - 1, round(p['x'] * width))] for p in erase_points} - {0}
            if not selected:
                raise ValueError('Click inside a visible mask to erase it, or drag across it with the brush.')
            seed[torch.from_numpy(np.isin(regions, list(selected))).to(seed.device)] = 0
        strokes = step.get('strokes', [])
        if strokes:
            canvas = Image.new('L', (width, height), 255) if erase_only else Image.fromarray((seed.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(canvas)
            for stroke in strokes:
                if stroke['mode'] not in ('paint', 'erase') or not 0 < stroke['width'] <= 1:
                    raise ValueError('Invalid repair brush.')
                if any(not (0 <= p['x'] <= 1 and 0 <= p['y'] <= 1) for p in stroke['points']):
                    raise ValueError('Brush points must be inside the image.')
                points = [(round(p['x'] * (width - 1)), round(p['y'] * (height - 1))) for p in stroke['points']]
                radius = max(1, round(stroke['width'] * width)) / 2
                fill = 255 if stroke['mode'] == 'paint' else 0
                if len(points) > 1:
                    draw.line(points, fill=fill, width=max(1, round(radius * 2)))
                for x, y in points:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)
            painted = torch.from_numpy(np.asarray(canvas).copy()).to(result) / 255
            seed = seed * painted if erase_only else painted
        if erase_only:
            removed = (result[first] > 0) & (seed == 0)
            if not removed.any():
                raise ValueError('No mask was touched. Click inside a mask or drag across it.')
            if last > first:
                removal = track(images[first:last + 1], removed.to(result).unsqueeze(0)).to(result)
                if len(removal) != last - first + 1:
                    raise ValueError('SAM returned the wrong erase frame count.')
                if erase_points and not strokes:
                    for offset, current in enumerate(result[first:last + 1]):
                        regions, _ = label((current > 0).detach().cpu().numpy(), structure=np.ones((3, 3)))
                        selected = np.unique(regions[(removal[offset] > 0.5).detach().cpu().numpy()])
                        removal[offset] = torch.from_numpy(np.isin(regions, selected[selected != 0])).to(removal)
                result[first:last + 1] = result[first:last + 1].masked_fill(removal > 0.5, 0)
            result[first] = seed
            continue
        if step.get('replace') and not seed.any():
            result[first:last + 1] = 0
            continue
        if last > first:
            propagated = track(images[first:last + 1], seed.unsqueeze(0)).to(result)
            if len(propagated) != last - first + 1:
                raise ValueError('SAM returned the wrong repair frame count.')
            result[first:last + 1] = propagated
        result[first] = seed
    return result


class H3MaskRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',),
                             'repairs': ('STRING', {'default': '{}', 'multiline': True}),
                             'enabled': ('BOOLEAN', {'default': True})},
                'optional': {'masks': ('MASK',), 'model': ('MODEL', {'lazy': True}), 'visible_frames': ('INT', {'default': 0, 'min': 0})}}

    RETURN_TYPES = ('MASK',)
    RETURN_NAMES = ('repaired_mask',)
    FUNCTION = 'repair'
    CATEGORY = 'MiniMax H3'
    OUTPUT_NODE = False
    DESCRIPTION = 'Preview and repair the selected subject mask. Open the editor after running the mask stage. Repairs are stored in the workflow; save the workflow to keep them. No H3 sampling is required to apply mask repairs.'

    def check_lazy_status(self, images, masks=None, enabled=True, repairs='{}', model=None, **kwargs):
        if masks is None:
            masks = images.new_zeros(images.shape[:3])
        spec = json.loads(repairs)
        steps = spec.get('steps', []) if enabled else []
        if steps and spec.get('signature') != repair_signature(images, masks):
            return []
        needs_model = any(s.get('positive') or int(s['end']) > int(s['frame']) for s in steps)
        return ['model'] if needs_model and model is None else []

    def repair(self, images, masks=None, repairs='{}', enabled=True, model=None, visible_frames=0):
        if masks is None:
            masks = images.new_zeros(images.shape[:3])
        if images.shape[0] != masks.shape[0] or tuple(images.shape[1:3]) != tuple(masks.shape[1:3]):
            raise ValueError('Connect source frames and their full-size masks with matching frame counts.')
        spec = json.loads(repairs)
        signature = repair_signature(images, masks)
        steps = spec.get('steps', []) if enabled else []
        repairs_skipped = bool(steps) and spec.get('signature') != signature
        if repairs_skipped:
            steps = []
        def segment(frame, positive, negative):
            if model is None:
                raise ValueError('Connect the SAM model to use point repairs.')
            return SAM3_Detect.execute(model, frame, positive_coords=json.dumps(positive), negative_coords=json.dumps(negative)).result[0]
        def track(frames, mask):
            if model is None:
                raise ValueError('Connect the SAM model to track a repair range.')
            data = SAM3_VideoTrack.execute(frames, model, initial_mask=mask, max_objects=1).result[0]
            return SAM3_TrackToMask.execute(data, '0').result[0]
        result = apply_mask_repairs(images, masks, steps, segment, track) if steps else masks
        relative = 'h3-mask-repair/' + uuid.uuid4().hex
        directory = Path(folder_paths.get_temp_directory()) / relative
        directory.mkdir(parents=True)
        height, width = images.shape[1:3]
        preview_w = min(768, width)
        preview_h = max(1, round(height * preview_w / width))
        visible_frames = visible_frames or len(images)
        if not 1 <= visible_frames <= len(images):
            raise ValueError('Invalid visible frame count.')
        if steps and visible_frames < len(result):
            result[visible_frames:] = result[visible_frames - 1]
        mask_pixels = []
        for i in range(visible_frames):
            frame = (images[i].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).resize((preview_w, preview_h)).save(directory / f'frame-{i}.jpg', quality=85)
            mask_pixels.append(int(torch.count_nonzero(result[i] > 0)))
            mask = F.adaptive_max_pool2d(result[i:i+1, None].float(), (preview_h, preview_w))[0, 0]
            Image.fromarray(np.ceil(mask.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)).save(directory / f'mask-{i}.png')
        return {'ui': {'h3_mask_repair': [{'directory': relative, 'frames': visible_frames, 'width': preview_w, 'height': preview_h, 'signature': signature, 'repairs_skipped': repairs_skipped, 'mask_pixels': mask_pixels, 'source_width': width, 'source_height': height}]}, 'result': (result,)}


NODE_CLASS_MAPPINGS = {'H3MaskRepair': H3MaskRepair}
NODE_DISPLAY_NAME_MAPPINGS = {'H3MaskRepair': 'H3 Mask Repair - Scrub / Click / Paint'}
WEB_DIRECTORY = './web'


class H3MaskFrameGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'masks': ('MASK',)}}

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'gate'
    CATEGORY = 'MiniMax H3'
    DESCRIPTION = 'Allow the generated crop only on frames with a selected mask. Leave empty frames unchanged.'

    def gate(self, masks):
        return ((masks > 0).flatten(1).any(1).to(masks.dtype)[:, None, None],)


NODE_CLASS_MAPPINGS['H3MaskFrameGate'] = H3MaskFrameGate
NODE_DISPLAY_NAME_MAPPINGS['H3MaskFrameGate'] = 'H3 Preserve Unmasked Frames'
