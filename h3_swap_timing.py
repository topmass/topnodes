import math

import torch
import torch.nn.functional as F


class H3SwapTimeline24:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "audio": ("AUDIO",), "video_info": ("VHS_VIDEOINFO",)}}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("aligned_audio", "output_frames")
    FUNCTION = "align_timeline"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = "The video loader converts to 24 FPS. This node locks output to that frame count and trims or pads audio to match. Normalize the whole source before splitting for seamless joins."

    def align_timeline(self, images, audio, video_info):
        if video_info["loaded_fps"] != 24:
            raise ValueError("Set the video loader force_rate to 24.")
        if len(images) == 0:
            raise ValueError("The input video has no frames.")
        waveform = audio["waveform"]
        samples = round(len(images) * audio["sample_rate"] / 24)
        aligned = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))
        return {**audio, "waveform": aligned}, len(images)


class H3SwapPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_info": ("VHS_VIDEOINFO",), "image": ("IMAGE",), "audio": ("AUDIO",), "regenerate_audio": ("BOOLEAN", {"default": False})}, "optional": {"output_frames": ("INT", {"default": 0, "min": 0, "forceInput": True})}}

    RETURN_TYPES = ("FLOAT", "INT", "IMAGE", "AUDIO", "INT", "MASK", "BOOLEAN")
    RETURN_NAMES = ("model_duration", "model_frames", "padded_images", "padded_audio", "output_frames", "audio_noise_mask", "regenerate_audio")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3"

    def prepare(self, video_info, image, audio, regenerate_audio=False, output_frames=0):
        if video_info["loaded_fps"] != 24:
            raise ValueError("Set the video loader force_rate to 24.")
        if len(image) == 0:
            raise ValueError("The input video has no frames.")
        waveform, rate = audio["waveform"], audio["sample_rate"]
        output_frames = output_frames or max(len(image), math.ceil(video_info["source_duration"] * 24), math.ceil(waveform.shape[-1] * 24 / rate))
        model_frames = max(5, output_frames)
        model_frames += (5 - model_frames % 17) % 17
        padded_images = torch.cat((image, image[-1:].expand(model_frames - len(image), -1, -1, -1)))
        samples = round(model_frames * rate / 24)
        padded_audio = {**audio, "waveform": F.pad(waveform, (0, samples - waveform.shape[-1]))}
        mask = image.new_full((1, 32, 32), float(regenerate_audio))
        return model_frames / 24, model_frames, padded_images, padded_audio, output_frames, mask, regenerate_audio


class H3SwapFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "original_audio": ("AUDIO",), "decoded_audio": ("AUDIO",), "output_frames": ("INT", {"forceInput": True}), "regenerate_audio": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "finish"
    CATEGORY = "MiniMax H3"

    def finish(self, images, original_audio, decoded_audio, output_frames, regenerate_audio):
        if len(images) < output_frames:
            raise ValueError("H3 returned fewer frames than the source clip. Do not export a shortened clip.")
        audio = decoded_audio if regenerate_audio else original_audio
        waveform = audio["waveform"]
        samples = round(output_frames * audio["sample_rate"] / 24)
        waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))
        return images[:output_frames].clone(), {**audio, "waveform": waveform}


NODE_CLASS_MAPPINGS = {"H3SwapTimeline24": H3SwapTimeline24, "H3SwapPrepare": H3SwapPrepare, "H3SwapFinish": H3SwapFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SwapTimeline24": "H3 Swap - Lock 24 FPS Timeline", "H3SwapPrepare": "H3 Swap - Full Clip / Audio Mode", "H3SwapFinish": "H3 Swap - Remove Padding / Select Audio"}


if __name__ == "__main__":
    for frames in (1, 5, 22, 56, 73, 74, 107):
        images = torch.arange(frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 2, 2, 3)
        original = {"sample_rate": 48000, "waveform": torch.ones(1, 2, frames * 2000 + 100)}
        for regenerate in (False, True):
            duration, count, padded, audio, target, mask, mode = H3SwapPrepare().prepare({"loaded_fps": 24, "source_duration": frames / 24}, images, original, regenerate)
            assert count % 17 == 5 and target == frames + 1 and count >= target
            assert torch.equal(padded[:frames], images) and torch.equal(padded[-1], images[-1])
            assert mask.max().item() == float(regenerate)
            decoded = {"sample_rate": 32000, "waveform": torch.zeros(1, 2, round(duration * 32000))}
            result, sound = H3SwapFinish().finish(padded, original, decoded, target, mode)
            assert len(result) == target and sound["waveform"].shape[-1] == round(target * sound["sample_rate"] / 24)
            if not regenerate:
                assert torch.equal(sound["waveform"][..., :original["waveform"].shape[-1]], original["waveform"])
            else:
                assert sound["waveform"].count_nonzero() == 0
    for rate in (32000, 44100, 48000):
        for frames in (5, 22, 73, 107):
            images = torch.zeros(frames, 2, 2, 3)
            info = {"loaded_fps": 24, "source_duration": frames / 24}
            source = {"sample_rate": rate, "waveform": torch.ones(1, 2, round(frames * rate / 24) + 100)}
            aligned, target = H3SwapTimeline24().align_timeline(images, source, info)
            for mode in (False, True):
                prepared = H3SwapPrepare().prepare(info, images, aligned, mode, target)
                result, sound = H3SwapFinish().finish(prepared[2], aligned, prepared[3], prepared[4], mode)
                assert len(result) == frames and sound["waveform"].shape[-1] == round(frames * rate / 24)
    print("Full-clip padding, trimming, original audio, and regenerated audio checks passed.")
