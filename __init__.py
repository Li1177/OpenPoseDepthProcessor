from .nodes import OpenPoseDepthProcessorNode

NODE_CLASS_MAPPINGS = {
    "OpenPoseDepthProcessor": OpenPoseDepthProcessorNode,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
