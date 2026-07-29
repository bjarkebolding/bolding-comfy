"""Node registry for the bolding-comfy pack.

Add a module here and list its classes below. Class-ID keys are prefixed
with "Bolding" so they cannot collide with another pack's nodes; the display
names are what appear in the menu.
"""
from .ltx_segment_loop import LTXSegmentLoop
from .video_io import SaveVideoSegment, StitchSegments

NODE_CLASS_MAPPINGS = {
    "BoldingLTXSegmentLoop": LTXSegmentLoop,
    "BoldingSaveVideoSegment": SaveVideoSegment,
    "BoldingStitchSegments": StitchSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoldingLTXSegmentLoop": "LTX Audio-Driven Segment Loop",
    "BoldingSaveVideoSegment": "Save Video Segment",
    "BoldingStitchSegments": "Stitch Segments + Audio",
}
