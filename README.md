# TopNodes

ComfyUI nodes and MiniMax H3 workflows for replacing one or two video characters with reference images. Includes SAM tracking reuse, mask repair, full-clip audio timing, resolution selection, and Sol-Attn/VSA.

## Install

Use a current [ComfyUI](https://github.com/Comfy-Org/ComfyUI) with native MiniMax H3, SAM3, and subgraph support. Clone this repository into `ComfyUI/custom_nodes/topnodes`:

```sh
git clone https://github.com/topmass/topnodes.git
```

Run that command from `ComfyUI/custom_nodes`. Restart ComfyUI and hard-refresh the browser. Do not install this alongside separate copies of `h3_swap_timing.py`, `h3_sam_cache.py`, `h3_video_resolution.py`, `h3_mask_repair`, or `sol_attn_minimax_v5.py`: they register the same node names. Move the old copies outside `custom_nodes` when migrating. Saved workflows retain their existing node names.

Install these public packs separately:

- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments)
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

Install their requirements in ComfyUI's Python environment. TopNodes itself uses ComfyUI's existing torch, numpy, and Pillow dependencies. FFmpeg must be installed. The supplied SageAttention selection requires a compatible SageAttention installation. Sol-Attn/VSA requires a compatible CUDA build of [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) with `sol_attn`; installing this Python node alone does not install GPU kernels. This package does not download models or make outbound network requests.

### Required FPS patch

These workflows expose `fps_rounding` on VideoHelperSuite's FFmpeg loader. If your installed loader does not offer it, apply the included patch from the VideoHelperSuite directory:

```sh
git apply --check ../topnodes/patches/vhs-fps-rounding.patch
git apply ../topnodes/patches/vhs-fps-rounding.patch
```

The patch modifies only FPS rounding and its input. It was checked against public upstream source when packaged. If the check fails, inspect your installed version instead of forcing the patch. Skip it if your loader already supports `fps_rounding`. Restart after applying. VideoHelperSuite updates may require reapplying it.

## Workflows and models

The `workflows` directory contains single-character and two-character FullClip workflows, each with a MaskRepair version. Use the MaskRepair versions for the editor and preservation of unmasked frames. Upload your own video and replacement images; example media and saved tracking are not included.

All four workflows are saved with:

- Main model: `Minimax-h3_Singularity_ref2va_pruned_v1.3_int8.safetensors`
- Turbo enabled at **0.6**: `H3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors`
- Speech/talking-face LoRA present in Power Lora Loader but **off**: `natural_face_speech_h3_lora_v1_500.safetensors`

Models are not redistributed. Place the main model in `models/diffusion_models`, and LoRAs in `models/loras` with the paths above. The workflows also expect:

- `models/checkpoints/sam3.1_multiplex_fp16.safetensors`
- `models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`
- The background-removal model selected by the workflow's native model loader.

Select equivalent compatible files if your filenames differ. The supplied quantizations and attention settings are hardware-specific; this is not a universal low-VRAM preset.

## Use

1. Upload the source clip, choose output size, and provide reference A (and B in the duo).
2. Set each SAM prompt and maximum tracks. Run the mask preview stage with caches set to **Track + save**.
3. Choose object indices separately: `0` selects the first tracked object, `1` the second, blank combines all tracks. A and B can use the same saved slot with different indices.
4. Set good tracking slots to **Reuse saved**. Slots hold their latest save only. Changing video, resolution, or frame conversion invalidates the saved masks. After fresh tracking, old repairs from a different source/mask are skipped without blocking generation. The editor shows a notice. Saved edits remain in the workflow; save a copy before clearing them to start new corrections.
5. Optionally open each mask repair editor. Scrub to a failure, add a target point, exclude another area, or paint/erase. Choose this frame, this frame through the end, or a custom endpoint. Apply queues mask processing only.
6. Save the workflow to keep corrections. Reset restores one frame's original selected SAM mask; Undo removes the latest correction. Later repairs affect only their specified range.
7. Enable video generation. Keep repair nodes enabled to use corrections; set their `enabled` input false to use selected SAM masks without repairs. Preview-stage switches do not control whether saved corrections feed generation.

In repair workflows, cleanup is bypassed to preserve brief masks. Empty frames retain the incoming image exactly during compositing; B preserves A's existing result on frames without a B mask. Each active crop branch still needs at least one usable mask somewhere in the clip. Masking every frame is not required.

Output is 24 FPS. Model-required padding is trimmed from the final video and audio. Original audio is the default; regenerated audio remains optional. FPS rounding is source-dependent: single defaults to `near`, duo to `up`, matching the source examples used during development. Inspect cadence when using a different source.

## Included nodes

| Source | Nodes |
|---|---|
| `h3_swap_timing.py` | H3SwapTimeline24, H3SwapPrepare, H3SwapFinish |
| `h3_video_resolution.py` | H3VideoResolution |
| `h3_sam_cache.py` | H3SAMTrackCache |
| `h3_mask_repair/` | H3MaskRepair, H3MaskFrameGate; browser editor |
| `sol_attn_minimax_v5.py` | SolAttnMiniMax |

## Checks

From the ComfyUI root, using its Python environment:

```sh
python custom_nodes/topnodes/h3_swap_timing.py
PYTHONPATH=. python custom_nodes/topnodes/h3_sam_cache.py
PYTHONPATH=. python custom_nodes/topnodes/h3_mask_repair/test_repair.py
PYTHONPATH=. python custom_nodes/topnodes/check_package.py
```

Mask tests cover repair ranges, point coordinates, painting, reset, source changes, visible frame counts, and native crop/uncrop with isolated masked frames. Packaging checks cover node registration, workflow settings, and removal of saved repairs. These checks do not run a complete H3 generation or certify every GPU configuration.
