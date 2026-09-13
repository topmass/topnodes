from . import h3_swap_timing, h3_sam_cache, h3_video_resolution, h3_mask_repair
from .sol_attn_minimax_v5 import SolAttnMiniMax

NODE_CLASS_MAPPINGS = {'SolAttnMiniMax': SolAttnMiniMax}
NODE_DISPLAY_NAME_MAPPINGS = {'SolAttnMiniMax': 'Sol-Attn MiniMax H3'}
for module in (h3_swap_timing, h3_sam_cache, h3_video_resolution, h3_mask_repair):
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = './h3_mask_repair/web'
