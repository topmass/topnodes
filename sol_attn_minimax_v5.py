"""Sol-Attn (arXiv 2607.24027) for MiniMax-H3, via comfy-kitchen's CUDA kernels.

Single-file node: installs an ``optimized_attention_override`` on the model
(per-model patch, sigma-scheduled dense warm-up), with H3's conditioning sink
and override chaining. Requires comfy_kitchen with ``sol_attn`` (bf16, head_dim
128, sm_80+); everything else falls back to the existing attention backend.
"""

import logging
import sys
from functools import partial

import torch

from comfy_api.latest import ComfyExtension, io

try:
    import comfy_kitchen as _ck
    _CK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    _ck = None
    _CK_IMPORT_ERROR = exc

HEAD_DIM = 128
BLOCK_SIZE = 64

_stats = {"sparse": 0, "producer": 0, "dense_fallback": 0, "outside_range": 0,
          "errors": 0}
_seen = set()




def sol_attn_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_stats)


def reset_sol_attn_stats():
    for key in _stats:
        _stats[key] = 0
    _seen.clear()
    _PRODUCER_STATS.clear()


def _log_once(key, message):
    if key not in _seen:
        _seen.add(key)
        logging.info(f"[sol_attn] {message}")


def _log_kernel_failure(exc):
    # Full traceback on the first distinct failure, short line on repeats.
    key = ("kernel_failure", type(exc).__name__, str(exc))
    first = key not in _seen
    _seen.add(key)
    logging.error(f"[sol_attn] kernel failed ({exc}); falling back", exc_info=first)


# ---------------------------------------------------------------------------
# H3 segment layout: publish the packed video / target-audio spans of the
# current call so the attention override can keep the conditioning rows exact.
# ---------------------------------------------------------------------------

def _h3_log_once(message):
    _log_once(("h3", message), f"H3 layout: {message}")


_INSTALLED = set()
_PATCHED_LAYOUTS = set()
# id(position_ids) -> (layout, video bounds, audio bounds). The layout is kept
# alive deliberately so the id cannot be recycled underneath us; there is one
# entry per distinct shape.
_SPANS = {}


def _patch_packed_layout(module):
    """Register the segment bounds of every PackedLayout built, without mutating it."""
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        segs = getattr(self, "segments", []) or []
        video = next(((a, b) for a, b, kind in segs if kind == "video"), None)
        # Target audio is the segment immediately before video; sink_q only
        # needs THOSE query rows dense, not the (possibly huge) reference rows.
        audio = next(((a, b) for a, b, kind in segs if kind == "audio"), None)
        if torch.is_tensor(getattr(self, "position_ids", None)) and video is not None:
            _SPANS[id(self.position_ids)] = (self, video, audio)

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def install_h3_layout(model):
    """Idempotently hook the model so each forward publishes its segment spans
    into transformer_options (read by the override's sink logic)."""
    if id(model) in _INSTALLED:
        return
    for attr in ("rope_freqs", "_forward"):
        if not hasattr(model, attr):
            raise RuntimeError(f"MiniMax-H3 layout hook needs .{attr} on the diffusion model")

    _patch_packed_layout(sys.modules[type(model).__module__])

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        model._sol_transformer_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_transformer_options = None
            transformer_options.pop("sol_h3_video_span", None)
            transformer_options.pop("sol_h3_audio_span", None)
            transformer_options.pop("sol_h3_layout", None)

    def rope_freqs(position_ids, device):
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            _h3_log_once("no layout registered; the conditioning sink is inactive")
        else:
            layout, video, audio = entry
            options = getattr(model, "_sol_transformer_options", None)
            if options is not None:
                options["sol_h3_video_span"] = video
                options["sol_h3_audio_span"] = audio
                options["sol_h3_layout"] = layout
        return original_rope_freqs(position_ids, device)

    model._forward = _forward
    model.rope_freqs = rope_freqs
    _INSTALLED.add(id(model))


def _gate(transformer_options, tokens, min_tokens, sigma_start, sigma_end):
    """Why this call stays dense regardless of its tensors -- too short, or
    outside the sampling window (the paper's dense warm-up) -- as a
    (stats counter, reason) pair, or None."""
    if tokens < min_tokens:
        return "dense_fallback", f"seq {tokens} < {min_tokens}"
    sigmas = (transformer_options or {}).get("sigmas")
    if sigmas is not None:
        sigma = float(sigmas[0])
        if (sigma_start is not None and sigma > sigma_start) or \
           (sigma_end is not None and sigma < sigma_end):
            return "outside_range", f"sigma {sigma:.3g} outside the sparse window"
    return None


def _ineligible(q, k, dim_head):
    """Why these tensors can't go through the kernel, or None. q/k are BTHD."""
    if q.device.type != "cuda":
        return "not cuda"
    if q.dtype != torch.bfloat16:
        return f"dtype {q.dtype} (kernel is bf16-only)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != 128"
    if q.shape[1] != k.shape[1]:
        return "cross-attention (kept dense)"
    if q.shape != k.shape:
        # GQA or any other q/k mismatch would silently index wrong.
        return f"q/k shape mismatch {tuple(q.shape)} vs {tuple(k.shape)}"
    return None


def _run(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
         tau, verbose, sink_blocks=(0, 0), sink_q=(0, 0), topk_ratio=0.0):
    """Returns the attention output, or None if this call should stay dense."""
    if skip_reshape:
        b, _, _, dim_head = q.shape          # BHND
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    else:
        b, _, dim_head = q.shape             # B, N, heads*dim_head
        dim_head //= heads
        qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

    reason = _ineligible(qs, ks, dim_head)
    if reason is not None:
        _stats["dense_fallback"] += 1
        if verbose:
            _log_once((tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
        return None

    out = _ck.sol_attn(
        qs, ks, vs, tau=tau, scale=scale,
        sink_blocks=list(sink_blocks), sink_q=list(sink_q),
        topk_ratio=topk_ratio,
    )  # BTHD
    _stats["sparse"] += 1
    if verbose:
        sel = f"topk={topk_ratio:.3f}" if topk_ratio else f"tau={tau}"
        _log_once((tuple(qs.shape), "sparse"),
                  f"sparse {tuple(qs.shape)} {sel} cuda-int8")

    if skip_output_reshape:
        return out.transpose(1, 2)           # BHND
    return out.reshape(b, -1, heads * dim_head)


_PRODUCER_STATS = {}
_PRODUCER_CHUNK = 4096

# ---------------------------------------------------------------------------
# VSA (FastVideo) mode: the H3 sequence re-tiled the way VSA-H3 trains --
# prefix segments in their own zero-padded 64-row tiles, the video span in
# 4x4x4 cubes (partial edge cubes zero-padded), prefix tiles always attended
# and prefix queries dense, top-k over video tiles, no pooled tail, plus the
# gated coarse branch from the checkpoint's to_gate_compress weights.
# ---------------------------------------------------------------------------
_VSA_PLANS = {}       # (signature, segments, device) -> plan; small LRU, no layout refs
_VSA_PLANS_MAX = 4
_VSA_ROPE = {}        # (id(rope_freqs), id(plan)) -> (rope_freqs, plan, permuted copy)
_VSA_CUBE = (4, 4, 4)


def _vsa_plan(layout, device):
    """Padded tile order for one PackedLayout: `src` maps each padded row to
    its source row (-1 = pad, live rows first within a tile), `inv` maps each
    source row to its padded position, `block_len` counts live rows per tile."""
    key = (tuple(layout.signature), tuple(layout.segments), str(device))
    plan = _VSA_PLANS.get(key)
    if plan is not None:
        return plan
    text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    tiles, n_prefix = [], 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind != "video":
            m = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            seg = torch.full((m * BLOCK_SIZE,), -1, dtype=torch.int64)
            seg[:n] = torch.arange(a, b)
            tiles.append(seg.view(m, BLOCK_SIZE))
            n_prefix += m
            continue
        if grid[0] * grid[1] * grid[2] != n:
            raise RuntimeError(f"VSA: video segment {n} rows does not match grid {grid}")
        ct, ch, cw = _VSA_CUBE
        pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, _VSA_CUBE))
        padded = torch.full((pt, ph, pw), -1, dtype=torch.int64)
        padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b).view(*grid)
        cubes = (padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw)
                 .permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK_SIZE))
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)   # live first
        tiles.append(torch.gather(cubes, 1, order))
    tiles = torch.cat(tiles)
    src = tiles.reshape(-1)
    live = src >= 0
    inv = torch.empty(layout.seq_len, dtype=torch.int64)
    inv[src[live]] = torch.nonzero(live).flatten()
    plan = {"n": int(src.numel()), "n_orig": int(layout.seq_len), "n_prefix": n_prefix,
            "src": src.to(device), "inv": inv.to(device),
            "block_len": (tiles >= 0).sum(1).to(torch.int32).to(device)}
    while len(_VSA_PLANS) >= _VSA_PLANS_MAX:
        del _VSA_PLANS[next(iter(_VSA_PLANS))]
    _VSA_PLANS[key] = plan
    return plan


def _vsa_rope(rope_freqs, plan):
    key = (id(rope_freqs), id(plan))
    hit = _VSA_ROPE.get(key)
    if hit is not None and hit[0] is rope_freqs and hit[1] is plan:
        return hit[2]
    padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
    padded[0, plan["inv"]] = rope_freqs[0]
    _VSA_ROPE.clear()
    _VSA_ROPE[key] = (rope_freqs, plan, padded)
    return padded


def _vsa_chunk(x, plan, i, m):
    """Padded rows [i, i+m) of x: source rows gathered, pad rows zero."""
    idx = plan["src"][i:i + m]
    return x[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(x.dtype)


def _make_producer_forward(module, stock_forward, opts):
    """Chunked-producer replacement for H3 Attention.forward: projects qkv in
    4K-token slices straight into comfy_kitchen's int8 carriers (norm+rope
    fused, full bf16 Q/K/V never materialised), then runs the sparse core.
    Anything ineligible falls back to the stock forward, whose attention call
    still reaches the normal override."""
    import comfy.model_management

    def forward(x, rope_freqs=None, transformer_options={}):
        def fallback():
            # The stock forward's attention call still reaches the override,
            # which applies the same gates and falls through to dense.
            return stock_forward(x, rope_freqs=rope_freqs,
                                 transformer_options=transformer_options)

        try:
            if (rope_freqs is None or x.dtype != torch.bfloat16 or x.dim() != 2
                    or x.device.type != "cuda"):
                return fallback()
            s = x.shape[0]
            if _gate(transformer_options, s, opts["min_tokens"],
                     opts["sigma_start"], opts["sigma_end"]) is not None:
                return fallback()   # the override counts and logs it
            tau = opts["tau"]
            topk = opts.get("topk_ratio", 0.0)
            from comfy_kitchen.backends import cuda as _ck_cuda
            h, hd = module.heads, module.head_dim
            qw = comfy.model_management.cast_to(module.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(module.k_norm.weight, device=x.device)
            extra = {}
            vsa = bool(opts.get("vsa"))
            freqs, n = rope_freqs, s
            if vsa:
                layout = (transformer_options or {}).get("sol_h3_layout")
                if layout is None or layout.seq_len != s:
                    _h3_log_once("no layout for this call; VSA tiling inactive")
                    return fallback()
                plan = _vsa_plan(layout, x.device)
                n = plan["n"]
                freqs = _vsa_rope(rope_freqs, plan)
                sink, sink_q = (0, plan["n_prefix"]), (0, plan["n_prefix"])
                extra = {"tail": False, "block_len": plan["block_len"]}
                gate = getattr(module, "to_gate_compress", None)   # a normal model layer
                if gate is not None:
                    # filled chunk by chunk below, alongside the projection
                    extra["coarse_gate"] = x.new_empty(n, h * hd).view(1, n, h, hd)
            else:
                sink, sink_q = _sink_blocks(transformer_options, s,
                                            opts["sink_conditioning"])
            stats = _PRODUCER_STATS.get((id(module), n))

            def chunks():
                for i in range(0, n, _PRODUCER_CHUNK):
                    if not vsa:
                        yield module.qkv_proj(x[i:i + _PRODUCER_CHUNK])
                        continue
                    xc = _vsa_chunk(x, plan, i, _PRODUCER_CHUNK)   # padded order, 44 MB at 4K rows
                    if gate is not None:
                        extra["coarse_gate"].view(n, h * hd)[i:i + xc.shape[0]] = gate(xc)
                    yield module.qkv_proj(xc)
            out, km, vs = _ck_cuda.sol_attn_chunked(
                chunks, n, h, freqs, (qw, kw),
                kmean=None if stats is None else stats[0],
                vscale=None if stats is None else stats[1],
                tau=tau, topk_ratio=topk,
                sink_blocks=list(sink), sink_q=list(sink_q),
                rope_eps=module.q_norm.eps, **extra)
            _PRODUCER_STATS[(id(module), n)] = (km, vs)
            _stats["producer"] += 1
            if opts["verbose"]:
                sel = f"topk={topk:.3f}" if topk else f"tau={tau}"
                mode = f"VSA tiles ({n} padded rows, {sink[1]} prefix tiles)" if vsa else "chunked qkv"
                _log_once(("producer", n), f"producer path: {s} tokens, {mode}, {sel}")
            out = out.view(n, h * hd)
            if vsa:
                out = out[plan["inv"]]
            return module.out_proj(out)
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
            return fallback()

    return forward


def _sink_blocks(transformer_options, tokens, mode):
    """(exact-KV blocks, dense-query blocks) for MiniMax-H3's conditioning rows.

    H3 packs [text][cond][ref][audio][video] into one sequence; sparsifying the
    conditioning rows costs sync and prompt adherence. exact_kv measures ~3%,
    exact_kv_and_rows ~17%, so exact-KV is the default and rows are opt-in.
    """
    if mode == "off":
        return (0, 0), (0, 0)
    span = (transformer_options or {}).get("sol_h3_video_span")
    if span is None:
        return (0, 0), (0, 0)
    video_start, video_stop = span
    if tokens < video_stop or video_start <= 0:
        return (0, 0), (0, 0)
    blocks = (0, (video_start + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if mode != "exact_kv_and_rows":
        return blocks, (0, 0)
    # Dense-query protection exists for the TARGET AUDIO rows; reference rows
    # only need the exact-KV side. Fall back to the whole conditioning range
    # when the layout did not publish an audio span.
    audio = (transformer_options or {}).get("sol_h3_audio_span")
    if audio is None:
        return blocks, blocks
    audio_start, _audio_stop = audio
    return blocks, (audio_start // BLOCK_SIZE, blocks[1])


def make_override(tau=1.0, min_tokens=4096,
                  sigma_start=None, sigma_end=None, verbose=False,
                  sink_conditioning="exact_kv", previous=None, topk_ratio=0.0,
                  vsa=False):
    """Build an optimized_attention_override callable.

    ``previous`` chains any override already installed on the model: every path
    that declines hands off to it first, falling through to ``func`` only if
    there is none. In VSA mode the override is dense-only: a VSA-trained
    checkpoint must never run plain block-sparse attention, so anything the
    producer patch declines runs dense here.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None or vsa:
            _stats["dense_fallback"] += 1
            return dense()
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        gated = _gate(kwargs.get("transformer_options"), tokens, min_tokens,
                      sigma_start, sigma_end)
        if gated is not None:
            counter, reason = gated
            _stats[counter] += 1
            if verbose:
                _log_once((tokens, reason), f"dense {tokens} tokens: {reason}")
            return dense()
        sink, sink_q = _sink_blocks(kwargs.get("transformer_options"), tokens,
                                    sink_conditioning)
        if verbose and sink != (0, 0):
            _log_once((tokens, sink, sink_q),
                      f"conditioning sink: KV blocks {sink} exact, dense query blocks {sink_q}")

        try:
            out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                       kwargs.get("scale", None), tau, verbose,
                       sink, sink_q, topk_ratio)
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
            return dense()
        return dense() if out is None else out

    return override




def _apply_patch(model, *, tau, start_percent, end_percent, min_tokens,
                 sink_conditioning, verbose, topk_ratio=0.0, vsa=False):
    diffusion_model = model.get_model_object("diffusion_model")
    is_h3 = hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")

    if is_h3 and (sink_conditioning != "off" or vsa):
        install_h3_layout(diffusion_model)
    blocks = getattr(diffusion_model, "blocks", None)
    if vsa and not is_h3:
        logging.warning("[sol_attn] VSA tiling needs MiniMax-H3; running plain top-k")
    elif vsa and blocks and not hasattr(blocks[0].attn, "to_gate_compress"):
        logging.warning("[sol_attn] VSA: checkpoint has no to_gate_compress weights; "
                        "running the fine stage only (no coarse branch)")

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = float(model_sampling.percent_to_sigma(start_percent))
    sigma_end = float(model_sampling.percent_to_sigma(end_percent))

    m = model.clone()
    previous = m.model_options["transformer_options"].get("optimized_attention_override")
    if previous is not None:
        logging.info("[sol_attn] chaining onto an existing attention override")

    # Chunked-producer path: patch each H3 self-attention forward so qkv is
    # projected in 4K slices straight into the int8 carriers (the full bf16
    # Q/K/V never exists). Both selection modes: top-k derives its threshold
    # from the producer's own pooled outputs in the workspace.
    if is_h3 and blocks is not None:
        opts = {"tau": tau, "topk_ratio": topk_ratio,
                "min_tokens": min_tokens,
                "sigma_start": sigma_start, "sigma_end": sigma_end,
                "sink_conditioning": sink_conditioning, "verbose": verbose,
                "vsa": vsa}
        installed = 0
        for i, blk in enumerate(blocks):
            attn = getattr(blk, "attn", None)
            if attn is None or not hasattr(attn, "qkv_proj"):
                continue
            key = f"diffusion_model.blocks.{i}.attn.forward"
            if key in m.object_patches:
                continue   # someone else patched it; the override handles it
            m.add_object_patch(
                key, _make_producer_forward(attn, attn.forward, opts))
            installed += 1
        if installed:
            _PRODUCER_STATS.clear()
            logging.info(f"[sol_attn] chunked qkv producer on {installed} blocks")

    m.model_options["transformer_options"]["optimized_attention_override"] = \
        make_override(tau=tau, min_tokens=min_tokens,
                      sigma_start=sigma_start, sigma_end=sigma_end,
                      verbose=verbose, sink_conditioning=sink_conditioning,
                      previous=previous, topk_ratio=topk_ratio, vsa=vsa)
    reset_sol_attn_stats()
    return io.NodeOutput(m)


class SolAttnMiniMax(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SolAttnMiniMax",
            display_name="Patch Sol-Attn (MiniMax)",
            is_experimental=True,
            category="sol_attn",
            description="Training-free block-sparse attention (Sol-Attn, arXiv "
                        "2607.24027) for MiniMax-H3, using comfy_kitchen.sol_attn. "
                        "bf16 + head_dim 128 only; ineligible calls fall back to the "
                        "existing attention backend. The win grows with sequence "
                        "length; below ~12k tokens dense is usually faster, so leave "
                        "min_tokens high.",
            inputs=[
                io.Model.Input("model"),
                io.DynamicCombo.Input("selection", options=[
                    io.DynamicCombo.Option("adaptive tau", [
                        io.Float.Input("tau", default=1.3, min=0.0, max=4.0,
                                       step=0.05,
                                       tooltip="Threshold beta. Higher is sparser: "
                                               "1.0 ~ 16% of blocks kept exact, "
                                               "1.5 ~ 7%, 2.0 ~ 2.7%."),
                    ]),
                    io.DynamicCombo.Option("top-k (SLA)", [
                        io.Float.Input("keep_percent", default=10.0, min=0.5,
                                       max=95.0, step=0.5,
                                       tooltip="Percent of key blocks each query "
                                               "block keeps exactly (sinks and the "
                                               "diagonal ride on top). With the "
                                               "lightx2v SLA turbo LoRA: 15 is the "
                                               "value it was distilled against, 10 "
                                               "is community-validated and faster. "
                                               "Without the LoRA, higher = closer "
                                               "to dense."),
                    ]),
                    io.DynamicCombo.Option("VSA (FastVideo)", [
                        io.Float.Input("vsa_keep_percent", default=10.0, min=0.5,
                                       max=95.0, step=0.5,
                                       tooltip="Percent of VIDEO cubes each query cube "
                                               "keeps; the FastH3-VSA checkpoints are "
                                               "trained at 10 (90% sparsity). The coarse "
                                               "branch uses the checkpoint's "
                                               "to_gate_compress layers when present."),
                    ]),
                ], tooltip="How exact key blocks are chosen per query block. "
                           "'adaptive tau': threshold at tau sigmas of the score "
                           "distribution (density varies per head/block). "
                           "'top-k (SLA)': a fixed keep_percent everywhere -- the "
                           "selection the lightx2v SLA LoRAs were distilled "
                           "against. 'VSA (FastVideo)': the FastH3-VSA recipe -- "
                           "4x4x4 video cubes, conditioning always attended, no "
                           "pooled tail, gated coarse branch; for checkpoints "
                           "trained with VSA. sink_conditioning is implied, and "
                           "anything outside the start/end window runs DENSE "
                           "(set start 0 / end 1 for a few-step checkpoint)."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Run dense before this point. The paper uses 0.2."),
                io.Float.Input("end_percent", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_tokens", default=12288, min=0, max=1 << 20, step=512,
                             tooltip="Sequences shorter than this stay dense."),
                io.Combo.Input("sink_conditioning",
                               options=["exact_kv", "exact_kv_and_rows", "off"],
                               default="exact_kv_and_rows",
                               tooltip="exact_kv: every query sees the packed "
                                       "text/audio/reference rows exactly (~3% cost). "
                                       "exact_kv_and_rows: additionally runs the TARGET "
                                       "AUDIO query rows dense (what keeps generated "
                                       "audio intact); reference rows stay sparse, so "
                                       "the cost is independent of reference size."),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, selection, start_percent, end_percent, min_tokens,
                sink_conditioning, verbose) -> io.NodeOutput:
        if _ck is None:
            raise RuntimeError(f"comfy_kitchen unavailable: {_CK_IMPORT_ERROR}")
        if not hasattr(_ck, "sol_attn"):
            raise RuntimeError(
                "comfy_kitchen has no sol_attn; rebuild the extension "
                "(python setup.py build_ext --inplace)")
        mode = selection["selection"]
        vsa = mode == "VSA (FastVideo)"
        keep = selection.get("vsa_keep_percent" if vsa else "keep_percent")
        return _apply_patch(
            model, tau=selection.get("tau", 1.3),
            start_percent=start_percent, end_percent=end_percent,
            min_tokens=min_tokens, sink_conditioning=sink_conditioning,
            verbose=verbose,
            topk_ratio=keep / 100.0 if mode != "adaptive tau" else 0.0,
            vsa=vsa)


class SolAttnMiniMaxExtension(ComfyExtension):
    async def get_node_list(self):
        return [SolAttnMiniMax]


async def comfy_entrypoint() -> SolAttnMiniMaxExtension:
    return SolAttnMiniMaxExtension()
