import cv2
import numpy as np
from custom_nodes.nodes import NODE_CLASS_MAPPINGS, ImageType, IntType

class OpenPoseDepthProcessorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "openpose_image": ("IMAGE",),
                "depth_image": ("IMAGE",),
                "margin": ("INT", {"default": 20, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "Custom Nodes"

    def process(self, openpose_image, depth_image, margin):
        depth_img_resized = cv2.resize(depth_image, (openpose_image.shape[1], openpose_image.shape[0]))
        gray_img = cv2.cvtColor(openpose_image, cv2.COLOR_BGR2GRAY)
        non_black_points = np.argwhere(gray_img > 0)
        mask = np.zeros_like(gray_img)
        mask[gray_img > 0] = 255

        kernel = np.ones((margin, margin), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        processed_depth_img = cv2.bitwise_and(depth_img_resized, depth_img_resized, mask=mask_dilated)

        return (processed_depth_img,)

NODE_CLASS_MAPPINGS.update({
    "OpenPoseDepthProcessor": OpenPoseDepthProcessorNode
})
