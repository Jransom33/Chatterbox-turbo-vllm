from .t3 import T3VllmModel, SPEECH_TOKEN_OFFSET
from vllm import ModelRegistry

ModelRegistry.register_model("ChatterboxT3", T3VllmModel)
