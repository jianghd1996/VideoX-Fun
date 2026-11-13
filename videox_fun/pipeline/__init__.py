"""Minimal Wan2.2 pipeline exports for the SingleRecon training branch.

The matching ``videox_fun.models`` package intentionally exports only Wan
components.  Importing every optional CogVideoX, Flux, Qwen, LTX, and audio
pipeline here would therefore fail before the Wan training entrypoint starts.
"""

from .pipeline_wan2_2 import Wan2_2Pipeline
from .pipeline_wan2_2_fun_inpaint import Wan2_2FunInpaintPipeline
from .pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline
from .pipeline_wan2_2_livephoto_frame_cat import Wan2_2LivePhotoFrameCatPipeline

Wan2_2FunPipeline = Wan2_2Pipeline
Wan2_2I2VPipeline = Wan2_2FunInpaintPipeline
