"""Audio-driven segment loop for LTX-2.3.

Core ComfyUI has no loop node, so a video longer than one generation has to be
unrolled by hand or driven from an external script. This uses the engine's
subgraph expansion to unroll N chained segments at run time, with N as an
ordinary widget.

The node is LTX-2.3 specific: it emits LTXVConcatAVLatent, LTXVSeparateAVLatent
and the two-stage upscale that model expects. The saving and stitching it feeds
are not, and live in video_io.
"""
from comfy_execution.graph_utils import GraphBuilder

from .shots import parse_shots, shot_at

# the schedules the shipped LTX-2.3 distilled-LoRA templates use
SIGMAS_S1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_S2 = "0.85, 0.7250, 0.4219, 0.0"


class LTXSegmentLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "audio": ("AUDIO", {"tooltip":
                          "The track the video is generated against. Each "
                          "segment pins its own window of it."}),
                "shots": ("STRING", {"multiline": True, "default":
                          "[0:00]\nA subject in a room, static camera, "
                          "photorealistic.", "tooltip":
                          "[mm:ss] keeps the frame chain, [mm:ss cut] breaks "
                          "it, [mm:ss back] resumes from before the cutaway."}),
                "negative": ("STRING", {"multiline": True, "default":
                             "cartoon, ugly, still frame, blurry"}),
                "segments": ("INT", {"default": 3, "min": 1, "max": 500}),
                "seg_secs": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0,
                                       "step": 0.5}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 3840,
                                  "step": 64}),
                "height": ("INT", {"default": 704, "min": 256, "max": 2160,
                                   "step": 64}),
                # control_after_generate must be declared: the ComfyUI
                # frontend adds that widget to any INT named "seed" whether
                # the backend asks for it or not. Without it the saved widget
                # array is one slot shorter than the UI expects and every
                # value after the seed lands in the wrong field.
                "seed": ("INT", {"default": 1234, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True}),
                "handoff_frames": ("INT", {"default": 9, "min": 1, "max": 33,
                                   "step": 8, "tooltip":
                                   "How many frames of the previous segment "
                                   "to hand to the next one. 1 pins position "
                                   "only, so the camera re-picks its "
                                   "direction and speed at every join. 9 pins "
                                   "two latent frames, which carries motion "
                                   "across the seam. Must be 8k+1."}),
                "filename_prefix": ("STRING", {"default": "segment"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip":
                                    "Optional still of the scene or character "
                                    "to start from. It seeds the first segment, "
                                    "and the frame chain carries it forward "
                                    "through the rest."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest",)
    FUNCTION = "expand"
    CATEGORY = "bolding/video"
    DESCRIPTION = ("Unroll N chained LTX-2.3 segments across an audio track. "
                   "Each segment pins its own window of audio as latents the "
                   "sampler never denoises, so the audio constrains the "
                   "picture and a speaking or singing subject syncs to it.")

    def expand(self, model, clip, vae, audio_vae, upscale_model, audio, shots,
               negative, segments, seg_secs, fps, width, height, seed,
               filename_prefix, handoff_frames=9, reference_image=None):
        frames = int(round(seg_secs * fps))
        # The VAE compresses time 8:1, so only 8k+1 pixel frames encode to a
        # whole number of latent frames. 1 gives a single pinned latent frame
        # (position); 9 gives two (position and motion).
        if (handoff_frames - 1) % 8:
            raise ValueError(
                f"handoff_frames must be 8k+1 (1, 9, 17, 25, 33), got "
                f"{handoff_frames}")
        # seconds of the previous segment replayed at the head of this one
        overlap = (handoff_frames - 1) / fps
        if (seg_secs * fps) % 8:
            raise ValueError(
                f"seg_secs * fps must be a multiple of 8 (got {seg_secs * fps}); "
                "LTX rounds its frame count to 8k+1")
        for name, v in (("width", width), ("height", height)):
            if v % 64:
                raise ValueError(f"{name} must be a multiple of 64 (got {v}); "
                                 "each stage halves then doubles it")

        shot_list = parse_shots(shots)
        w2, h2 = width // 2, height // 2

        g = GraphBuilder()
        neg_node = g.node("CLIPTextEncode", text=negative, clip=clip)
        samp = g.node("KSamplerSelect", sampler_name="euler")
        sig1 = g.node("ManualSigmas", sigmas=SIGMAS_S1)
        sig2 = g.node("ManualSigmas", sigmas=SIGMAS_S2)
        noise2 = g.node("RandomNoise", noise_seed=42)
        mask = g.node("SolidMask", value=0.0, width=width, height=height)
        # one text encode per distinct shot rather than per segment
        pos_nodes = {i: g.node("CLIPTextEncode", text=s["prompt"], clip=clip)
                     for i, s in enumerate(shot_list)}

        prev_dec = None      # the preceding segment
        anchor_dec = None    # last segment before a cutaway, for 'back'
        manifest = None

        for i in range(segments):
            start = i * seg_secs
            si = shot_at(shot_list, start)
            new_shot = i > 0 and si != shot_at(shot_list, start - seg_secs)

            # Where this segment's opening frame comes from. A reference
            # still stands in for "the previous segment" on the very first one,
            # which is the only place there is nothing to carry forward from.
            src = seed_img = None
            if i == 0:
                seed_img = reference_image
            elif new_shot and shot_list[si]["cut"]:
                pass                      # a real cut starts from nothing
            elif new_shot and shot_list[si]["back"]:
                if anchor_dec is None:
                    raise ValueError(
                        f"the shot at {shot_list[si]['at']:g}s uses 'back' but "
                        "nothing earlier was on screen to return to")
                src = anchor_dec
            else:
                src = prev_dec

            # A seeded segment regenerates the handed-over frames at its
            # head, so it must be that much longer and its audio window must
            # reach back over the same span, or picture and sound slide apart.
            # A previous segment hands over handoff_frames; a reference
            # still is one frame and pins position only, so it drops one.
            head = handoff_frames if src is not None else (1 if seed_img is not None else 0)
            seeded = head > 0
            gen_len = frames + (head if seeded else 1)
            ov = (head - 1) / fps if head > 1 else 0.0
            trim = g.node("TrimAudioDuration", audio=audio,
                          start_index=max(0.0, start - ov),
                          duration=seg_secs + ov)
            aenc = g.node("LTXVAudioVAEEncode", audio=trim.out(0),
                          audio_vae=audio_vae)
            # mask value 0 means the sampler never denoises the audio latents,
            # so the track constrains the video instead of being generated
            apin = g.node("SetLatentNoiseMask", samples=aenc.out(0),
                          mask=mask.out(0))
            empty = g.node("EmptyLTXVLatentVideo", width=w2, height=h2,
                           length=gen_len, batch_size=1)
            cond = g.node("LTXVConditioning", positive=pos_nodes[si].out(0),
                          negative=neg_node.out(0), frame_rate=fps)

            pre = None
            if src is not None or seed_img is not None:
                if src is not None:
                    # The last frames the source segment KEPT, not the last it
                    # generated: it dropped its own replayed head, so the kept
                    # range starts at that offset. Reading from the generated
                    # tail instead would hand over frames the viewer never saw.
                    src_dec, src_head = src
                    start_img = g.node(
                        "ImageFromBatch", image=src_dec.out(0),
                        batch_index=src_head + frames - handoff_frames,
                        length=handoff_frames).out(0)
                else:
                    start_img = seed_img
                rs = g.node("ResizeImagesByLongerEdge", images=start_img,
                            longer_edge=1536)
                pre = g.node("LTXVPreprocess", image=rs.out(0),
                             img_compression=18)
                stage1 = g.node("LTXVImgToVideoInplace", vae=vae,
                                image=pre.out(0), latent=empty.out(0),
                                strength=0.7, bypass=False).out(0)
            else:
                stage1 = empty.out(0)

            cat1 = g.node("LTXVConcatAVLatent", video_latent=stage1,
                          audio_latent=apin.out(0))
            guide1 = g.node("CFGGuider", model=model, positive=cond.out(0),
                            negative=cond.out(1), cfg=1.0)
            run1 = g.node("SamplerCustomAdvanced",
                          noise=g.node("RandomNoise", noise_seed=seed + i).out(0),
                          guider=guide1.out(0), sampler=samp.out(0),
                          sigmas=sig1.out(0), latent_image=cat1.out(0))
            split1 = g.node("LTXVSeparateAVLatent", av_latent=run1.out(0))

            up = g.node("LTXVLatentUpsampler", samples=split1.out(0),
                        upscale_model=upscale_model, vae=vae)
            if pre is not None:
                stage2 = g.node("LTXVImgToVideoInplace", vae=vae,
                                image=pre.out(0), latent=up.out(0),
                                strength=1.0, bypass=False).out(0)
            else:
                stage2 = up.out(0)

            cat2 = g.node("LTXVConcatAVLatent", video_latent=stage2,
                          audio_latent=split1.out(1))
            crop = g.node("LTXVCropGuides", positive=cond.out(0),
                          negative=cond.out(1), latent=split1.out(0))
            guide2 = g.node("CFGGuider", model=model, positive=crop.out(0),
                            negative=crop.out(1), cfg=1.0)
            run2 = g.node("SamplerCustomAdvanced", noise=noise2.out(0),
                          guider=guide2.out(0), sampler=samp.out(0),
                          sigmas=sig2.out(0), latent_image=cat2.out(0))
            split2 = g.node("LTXVSeparateAVLatent", av_latent=run2.out(0))
            dec = g.node("VAEDecodeTiled", samples=split2.out(0), vae=vae,
                         tile_size=768, overlap=64, temporal_size=4096,
                         temporal_overlap=4)

            save = g.node("BoldingSaveVideoSegment", images=dec.out(0),
                          filename_prefix=filename_prefix, index=i,
                          drop_head=head,
                          target_frames=frames,
                          manifest_in="" if manifest is None else manifest)
            manifest = save.out(0)

            prev_dec = (dec, head)
            # the anchor deliberately does not advance through a cutaway
            if not shot_list[si]["cut"]:
                anchor_dec = (dec, head)

        return {"expand": g.finalize(), "result": (manifest,)}
