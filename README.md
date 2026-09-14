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

Install their requirements in ComfyUI's Python environment. TopNodes itself uses ComfyUI's existing torch, numpy, Pillow, and scipy dependencies. FFmpeg must be installed. The supplied SageAttention selection requires a compatible SageAttention installation. Sol-Attn/VSA requires a compatible CUDA build of [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) with `sol_attn`; installing this Python node alone does not install GPU kernels. This package does not download models or make outbound network requests.

### Automatic video timing fix

TopNodes includes `install.py` for ComfyUI-Manager installation and `prestartup_script.py` for normal ComfyUI startup. After installing or updating TopNodes, restart ComfyUI. The startup helper checks the installed VideoHelperSuite loader and applies the supported timing changes automatically, including after a VideoHelperSuite update.

The helper adds the FPS-rounding input and prevents raw-video output from duplicating opening frames on clips with offset timestamps. It creates an exact `.topnodes-<hash>.bak` backup alongside the loader before changing it. Already-patched files are left untouched. Unsupported versions or file-permission errors produce a `[TopNodes]` console message; the helper does not force a patch or download dependencies. Install VideoHelperSuite first, or install it later and restart.

A manual Git pull does not run installers by itself; the next ComfyUI startup runs this check. To run the same helper directly, use `python install.py` from the TopNodes directory. The patch file remains available under `patches/` for inspection, but normal installation does not require applying it manually.

## Workflows and models

The `workflows` directory contains single-character and two-character FullClip workflows, each with a MaskRepair version. Use the MaskRepair versions for the editor and preservation of unmasked frames. Upload your own video and replacement images; example media and saved tracking are not included.

All workflows are saved with:

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

### Click-first single-character workflow

`TopNodes-Vid-Input-Swap-FullClip-ClickMask.json` starts without a SAM text prompt or existing masks. Restart ComfyUI and refresh the browser after updating the nodes. Choose the clip and output size, then open the mask suite. Opening it queues clip preparation automatically. Click the target, select **From this frame to the end**, and click **Apply**. Scrub to any failure and apply another correction from there, or edit one frame.

Click **Done** and save the workflow. Turn **STEP 1 - Mask suite preview** off and **STEP 2** on to generate video. Keep the mask suite node itself enabled. ComfyUI can reuse its cached result; after a restart, it replays the saved edits with SAM. This is not a separate saved mask file. The editor keeps separate edit sessions for each source and size, so switching clips does not require clearing another clip's edits. Point tracking can stop at a scene cut. Check each new shot and add a target point if its mask is missing; text-based SAM re-detection is not part of this workflow.

### Prompt-based workflows

1. Upload the source clip, choose output size, and provide reference A (and B in the duo).
2. Set each SAM prompt and maximum tracks. Run the mask preview stage with caches set to **Track + save**.
3. Choose object indices separately: `0` selects the first tracked object, `1` the second, blank combines all tracks. A and B can use the same saved slot with different indices.
4. Set good tracking slots to **Reuse saved**. Slots hold their latest save only. Changing video, resolution, or frame conversion invalidates the saved masks. After fresh tracking, old repairs from a different source/mask are skipped without blocking generation. The editor shows a notice. Saved edits remain in the workflow; save a copy before clearing them to start new corrections.
5. Optionally click **Open mask suite**. Previews show masks over the source video. The suite offers source-only and mask-only views, 2x/4x zoom, faint-pixel highlighting, and full-resolution masked-pixel counts. **Replace entire frame mask** clears the old mask before applying your new selection or paint. **Empty this frame** removes all mask pixels from that frame. Scrub to a failure and choose **Add mask** or **Erase mask**. In Add, click to segment a target or drag to paint. In Erase, click inside a connected mask region to remove that region, or drag to brush away part of it. Forward Erase tracks the removed region and subtracts it from saved masks; it does not regenerate the remaining mask. The other controls are under **More options**. Choose this frame, this frame through the end, or a custom endpoint. Apply queues mask processing only.
6. Save the workflow to keep corrections. Reset restores one frame's original selected SAM mask; Undo removes the latest correction. Later repairs affect only their specified range.
7. Enable video generation. Keep repair nodes enabled to use corrections; set their `enabled` input false to use selected SAM masks without repairs. Preview-stage switches do not control whether saved corrections feed generation.

Single-character editor workflows use the original SAM workflow cleanup, crop, latent-mask expansion, and sampling. The editor replaces the mask source. The final edited mask repeats across H3's hidden repeated video frames, so padding cannot expose the original face as unmasked conditioning. Empty-mask frames retain the source during compositing, preventing changes from a neighboring masked latent group from appearing across a cut. Cleanup can remove small or brief manual marks, just as it does with SAM masks. The two-character repair workflow still bypasses cleanup and preserves incoming frames where a subject mask is empty; B preserves A's existing result on those frames. Each active crop branch still needs at least one usable mask somewhere in the clip. Masking every frame is not required.

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
python custom_nodes/topnodes/check_video_loader.py
python custom_nodes/topnodes/check_install.py
```

Mask tests cover repair ranges, point coordinates, painting, reset, source changes, visible frame counts, and native crop/uncrop with isolated masked frames. Packaging checks cover node registration, workflow settings, and removal of saved repairs. These checks do not run a complete H3 generation or certify every GPU configuration.
