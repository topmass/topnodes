RESOLUTIONS = {
    "Landscape - 1280 x 720 (720p)": (1280, 720),
    "Landscape - 1920 x 1080 (1080p)": (1920, 1080),
    "Landscape - 960 x 544": (960, 544),
    "Landscape - 736 x 416": (736, 416),
    "Portrait - 720 x 1280 (720p)": (720, 1280),
    "Portrait - 1080 x 1920 (1080p)": (1080, 1920),
    "Portrait - 544 x 960": (544, 960),
    "Portrait - 416 x 736": (416, 736),
}


class H3VideoResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"resolution": (list(RESOLUTIONS),)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select_resolution"
    CATEGORY = "MiniMax H3"

    def select_resolution(self, resolution):
        return RESOLUTIONS[resolution]


NODE_CLASS_MAPPINGS = {"H3VideoResolution": H3VideoResolution}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VideoResolution": "H3 Video Output Size"}


if __name__ == "__main__":
    node = H3VideoResolution()
    for preset, expected in zip(RESOLUTIONS, [(1280, 720), (1920, 1080), (960, 544), (736, 416), (720, 1280), (1080, 1920), (544, 960), (416, 736)], strict=True):
        assert node.select_resolution(preset) == expected
