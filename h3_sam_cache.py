import hashlib
import os
from pathlib import Path

import torch

import folder_paths


class H3SAMTrackCache:
    SLOTS = ['Single', 'Duo A', 'Duo B']

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'slot': (cls.SLOTS,), 'mode': (['Track + save', 'Reuse saved'],)}, 'optional': {'track_data': ('SAM3_TRACK_DATA', {'lazy': True})}}

    RETURN_TYPES = ('SAM3_TRACK_DATA',)
    FUNCTION = 'load_or_save'
    CATEGORY = 'MiniMax H3'
    DESCRIPTION = 'Track + save stores SAM results. Reuse saved skips SAM and checks that the source frames match. Switch back to Track + save after changing the SAM prompt or tracking settings.'

    @classmethod
    def cache_path(cls, slot):
        if slot not in cls.SLOTS:
            raise ValueError('Select a valid SAM cache slot.')
        return Path(folder_paths.get_user_directory()) / 'h3_sam_cache' / (slot.replace(' ', '_') + '.pt')

    @classmethod
    def IS_CHANGED(cls, slot, mode, **kwargs):
        path = cls.cache_path(slot)
        return path.stat().st_mtime_ns if path.exists() else 'missing'

    def check_lazy_status(self, images, slot, mode, track_data=None):
        return ['track_data'] if mode == 'Track + save' and track_data is None else []

    def load_or_save(self, images, slot, mode, track_data=None):
        path = self.cache_path(slot)
        digest = hashlib.sha256(str((tuple(images.shape), images.dtype)).encode())
        for frame in images:
            digest.update(memoryview(frame.detach().cpu().contiguous().numpy()).cast('B'))
        signature = digest.hexdigest()
        if mode == 'Track + save':
            if track_data is None:
                raise ValueError('Connect the SAM tracker to track_data before saving.')
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix('.tmp')
            torch.save({'source': signature, 'track_data': track_data}, pending)
            os.replace(pending, path)
        elif mode == 'Reuse saved':
            if not path.exists():
                raise ValueError(f'No saved masks for {slot}. Run Track + save first.')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['source'] != signature:
                raise ValueError(f'The source frames changed for {slot}. Run Track + save again.')
            track_data = saved['track_data']
        else:
            raise ValueError('Select Track + save or Reuse saved.')
        return (track_data,)


NODE_CLASS_MAPPINGS = {'H3SAMTrackCache': H3SAMTrackCache}
NODE_DISPLAY_NAME_MAPPINGS = {'H3SAMTrackCache': 'H3 SAM - Save / Reuse Tracking'}


if __name__ == '__main__':
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory, patch.object(folder_paths, 'get_user_directory', return_value=directory):
        node = H3SAMTrackCache()
        frames = torch.zeros(3, 4, 4, 3)
        data = {'packed_masks': torch.ones(3, 1, 2, 1, dtype=torch.uint8), 'n_frames': 3, 'orig_size': (4, 4), 'scores': [0.9]}
        assert node.check_lazy_status(frames, 'Duo A', 'Track + save') == ['track_data']
        assert node.check_lazy_status(frames, 'Duo A', 'Reuse saved') == []
        node.load_or_save(frames, 'Duo A', 'Track + save', data)
        restored = node.load_or_save(frames, 'Duo A', 'Reuse saved')[0]
        assert torch.equal(restored['packed_masks'], data['packed_masks'])
        assert restored['orig_size'] == data['orig_size']
        for images, slot in [(frames + 1, 'Duo A'), (frames, 'Duo B')]:
            try:
                node.load_or_save(images, slot, 'Reuse saved')
            except ValueError:
                pass
            else:
                raise AssertionError('Changed source or missing cache was accepted.')
    print('SAM save/reuse, lazy tracking, slot isolation, and changed-source checks passed.')
