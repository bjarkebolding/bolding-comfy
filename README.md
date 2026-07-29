# bolding-comfy

Personal ComfyUI nodes.

## Install

Clone into ComfyUI's `custom_nodes/` and restart:

    cd ComfyUI/custom_nodes
    git clone https://github.com/bjarkebolding/bolding-comfy

With [spark-comfyui](https://github.com/bjarkebolding/spark-comfyui) that is
`data/custom_nodes/`, which is already bind-mounted into the container, so no
extra configuration is needed:

    cd data/custom_nodes
    git clone https://github.com/bjarkebolding/bolding-comfy
    ./spark-comfyui.sh stop && ./spark-comfyui.sh service --no-bf16-vae

Installing it there also means `backup` records the pack's origin URL and
commit in its manifest, and `restore` re-clones it at that commit. A checkout
kept elsewhere and bind-mounted is invisible to both, so a restored machine
would come back without these nodes.

If you do want the source outside the ComfyUI tree, for instance to point
several installs at one checkout, bind-mount it instead by adding a line to
`spark-mounts.conf`, and keep your own backups of it:

    mount = /path/to/bolding-comfy:/opt/ComfyUI/custom_nodes/bolding-comfy

Requires `av` and `numpy`, both already present in a standard ComfyUI
install.

## The VAE precision flag

The segment loop node cannot run under `--bf16-vae`, hence the
`--no-bf16-vae` above. This is not a limitation of these nodes. LTX-2.3's
audio VAE never casts the incoming waveform to the VAE dtype, so any workflow
that feeds it audio fails with:

    Input type (float) and bias type (c10::BFloat16) should be the same

The stock `LoadAudio` node hits it too, and so do the LTX-2.3 audio templates
ComfyUI ships. On a plain ComfyUI install nothing is affected unless you pass
`--bf16-vae` yourself. spark-comfyui adds it by default as part of the GB10
fast path, so it needs turning off for these workflows, which
`--no-bf16-vae` does without giving up the unet and text encoder speedups.
Adding `--fp32-vae` instead does not help, because ComfyUI's VAE precision
flags are mutually exclusive and one has already been added by then.

## Nodes

**LTX Audio-Driven Segment Loop** (`bolding/video`)

Unrolls N chained LTX-2.3 segments across an audio track, using the engine's
subgraph expansion, so the segment count is a widget rather than a hand-built
graph. Each segment encodes its own window of the audio and pins those latents
with a zero noise mask, so the sampler never denoises them: the audio becomes a
constraint on the picture rather than something generated alongside it. A
speaking or singing subject syncs to it.

Continuity across segments comes from handing each segment's last frame to the
next through `LTXVImgToVideoInplace`. A shot list decides what happens when:

    [0:00]        change the prompt, keep the frame chain (a move within a shot)
    [0:32 cut]    break the chain, a real scene change
    [0:48 back]   resume from the anchor, the last frame before the cutaway

`back` is what makes cutaways affordable: the anchor deliberately does not
advance through a `cut`, so a subject returns looking like itself.

**`handoff_frames`** decides how much of the previous segment is handed to
the next. `LTXVImgToVideoInplace` pins as many latent frames as its input
encodes to, and the VAE compresses time 8:1, so the value must be `8k+1`.
`1` pins position only: the camera decelerates into a still frame and picks a
fresh direction and speed on the far side of every join. `9` pins two latent
frames, which carries motion across the seam. Seeded segments generate
`handoff_frames` extra frames and drop them again, and their audio window
reaches back by the same amount, so the frame count and sync are unchanged.

An optional `reference_image` seeds the first segment with a still of your
scene or character, and the frame chain carries it forward. Leave it
unconnected to let the model invent everything from the shot list alone.

This node is LTX-2.3 specific. The two below are not.

**Save Video Segment** and **Stitch Segments + Audio** (`bolding/video`)

Write each segment to lossless FFV1 as it finishes, then join them and mux an
audio track across the joins. Segments are written rather than accumulated
because a float32 frame at 1280x704 is about 10.8 MB, so three and a half
minutes in one IMAGE batch is roughly 54 GB. Audio is muxed from the `AUDIO`
tensor, never from a file.

The stitched file reports itself to ComfyUI the way the core video nodes do,
so it appears and plays in the media assets pane rather than only on disk.

## Two things worth knowing

**Frame counts.** Video generators commonly emit `8k+1` frames, one more than
the audio window they were conditioned on. `target_frames` trims each segment
to exactly the right length. Without it the surplus accumulates and the picture
runs ahead of the sound: eleven cutaways cost 500 ms by the end of a
three-minute video. Expected total is always `segments * seg_secs * fps`.

**Generated audio often stops rather than resolves.** `fade_out_secs` on the
stitch node applies a cosine fade to the tail.

## Example

`example_workflows/audio_driven_music_video.json` is a complete, runnable
graph: a song generated with ACE-Step 1.5 driving a lip-synced video, laid out
left to right in four labelled groups.

## Writing nodes for this pack

Any INT widget named `seed` **must** declare `control_after_generate`. The
ComfyUI frontend adds that widget on its own for anything called `seed`,
whether the backend asks for it or not, so a node that omits it builds one
fewer widget slot than the UI expects and every saved value after the seed
lands in the wrong field. The API path ignores the control widget, so this
breaks only in the browser and looks like a corrupt workflow file.

## Requirements

Sizes must satisfy `seg_secs * fps` divisible by 8, and width and height
divisible by 64, since each stage halves then doubles them. The nodes raise
rather than silently produce a misaligned video.
