"""Writing and joining video segments. Model-agnostic.

Long generated videos cannot be held in memory as one IMAGE batch: at
1280x704 a float32 frame is about 10.8 MB, so three and a half minutes is
roughly 54 GB. These nodes write each segment as it finishes and join them
afterwards, which keeps peak memory at a single segment.
"""
import os

import av
import numpy as np
import torch

import folder_paths


class SaveVideoSegment:
    """Write one segment losslessly and append it to a manifest.

    The manifest threads through a chain of these so segments render and are
    written one at a time, letting the cache release each one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "segment"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "drop_head": ("INT", {"default": 0, "min": 0, "max": 1000,
                              "tooltip": "Drop this many frames from the "
                              "start: the carried-over seed frames the "
                              "previous segment already showed."}),
                "target_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                                  "tooltip": "Trim to exactly this many frames. "
                                  "0 disables. Generators that emit 8k+1 frames "
                                  "otherwise run ahead of the audio."}),
            },
            "optional": {
                "manifest_in": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest",)
    FUNCTION = "save"
    CATEGORY = "bolding/video"
    DESCRIPTION = ("Write a segment to a lossless FFV1 file so its frames can "
                   "leave memory, and thread a manifest through the chain.")

    def save(self, images, filename_prefix, index, drop_head, target_frames,
             manifest_in=""):
        seg_dir = os.path.join(folder_paths.get_output_directory(), "segments")
        os.makedirs(seg_dir, exist_ok=True)
        path = os.path.join(seg_dir, f"{filename_prefix}_{index:03d}.mkv")

        arr = images
        if drop_head:
            arr = arr[drop_head:]
        # Video generators commonly emit 8k+1 frames, one more than the audio
        # window they were conditioned on. Dropping the seed frame corrects a
        # chained segment but not an unchained one, and the surplus then
        # accumulates: every unchained segment puts the picture 1/fps further
        # ahead of the sound for the rest of the video.
        if target_frames and arr.shape[0] > target_frames:
            arr = arr[:target_frames]
        n, h, w, _ = arr.shape
        if target_frames and n != target_frames:
            print(f"[bolding] WARNING segment {index}: {n} frames, expected "
                  f"{target_frames}; audio will drift")

        container = av.open(path, "w")
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            frame = (arr[i].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            container.mux(stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())
        container.close()

        entry = f"{n}\t{path}"
        return (f"{manifest_in}\n{entry}" if manifest_in else entry,)


class StitchSegments:
    """Join saved segments and mux an audio track over the whole thing.

    Audio arrives as a tensor, so there is no file decode and none of the
    packed/planar, int/float confusion that muxing from a file invites.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("STRING", {"forceInput": True}),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "tooltip":
                                "x264 quality, lower is better. 0 is lossless."}),
                "fade_out_secs": ("FLOAT", {"default": 0.0, "min": 0.0,
                                  "max": 30.0, "step": 0.5, "tooltip":
                                  "Cosine fade on the audio tail. Generated "
                                  "tracks often stop rather than resolve."}),
                "filename_prefix": ("STRING", {"default": "video"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "stitch"
    CATEGORY = "bolding/video"
    DESCRIPTION = ("Concatenate the segments in a manifest and mux an audio "
                   "track across the joins. The picture is joined, the sound "
                   "never is.")

    def stitch(self, manifest, audio, fps, crf, fade_out_secs, filename_prefix):
        paths = [ln.split("\t", 1)[1] for ln in manifest.strip().splitlines()
                 if ln.strip()]
        if not paths:
            raise ValueError("empty manifest, nothing to stitch")
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise ValueError(f"missing segment files: {missing[:3]}")

        # ComfyUI's own helper, so the file lands where the UI looks for it and
        # gets the same counter and subfolder handling as the core save nodes
        full_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_folder, exist_ok=True)
        file = f"{name}_{counter:05d}.mp4"
        dst = os.path.join(full_folder, file)

        probe = av.open(paths[0])
        f0 = next(probe.decode(video=0))
        h, w = f0.height, f0.width
        probe.close()

        wave = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wave.dim() == 3:
            wave = wave[0]
        wave = wave.detach().float().cpu().numpy()
        if wave.ndim == 1:
            wave = wave[None, :]

        # Both streams must exist before anything is muxed: the container
        # header is written on the first mux and a stream added afterwards
        # breaks the time bases.
        out = av.open(dst, "w")
        vs = out.add_stream("libx264", rate=int(round(fps)))
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        vs.options = {"crf": str(crf), "preset": "slow"}
        aus = out.add_stream("aac", rate=sr)
        aus.bit_rate = 320000
        aus.layout = "stereo" if wave.shape[0] > 1 else "mono"

        total = 0
        for p in paths:
            for frame in av.open(p).decode(video=0):
                # rebuilt from the array so it carries no timestamps of its
                # own; every source file restarts its pts at zero
                out.mux(vs.encode(av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format="rgb24"), format="rgb24")))
                total += 1
        out.mux(vs.encode())

        need = int(total / fps * sr)
        wave = wave[:, :need]
        if wave.shape[1] < need:
            wave = np.pad(wave, ((0, 0), (0, need - wave.shape[1])))
        if fade_out_secs > 0:
            k = min(int(fade_out_secs * sr), wave.shape[1])
            ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
            wave[:, -k:] = wave[:, -k:] * ramp
        wave = np.ascontiguousarray(wave.astype(np.float32))
        for i in range(0, wave.shape[1], 1024):
            fr = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(wave[:, i:i + 1024]),
                format="fltp", layout=aus.layout.name)
            fr.sample_rate = sr
            out.mux(aus.encode(fr))
        out.mux(aus.encode())
        out.close()

        print(f"[bolding] wrote {dst}: {total} frames, {total/fps:.3f}s, {w}x{h}")
        # The shape core video nodes use. A {"text": ...} payload has no
        # renderer, so the result never appears in the media assets pane.
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder,
                                   "type": "output"}],
                       "animated": (True,)}}
