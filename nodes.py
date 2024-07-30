import cv2
import numpy as np
from comfyui import Node, ImageType, IntType

class OpenPoseDepthProcessorNode(Node):
    def __init__(self):
        super().__init__()
        self.add_input("openpose_image", ImageType)
        self.add_input("depth_image", ImageType)
        self.add_output("processed_depth_image", ImageType)
        self.add_param("margin", IntType, 20)  # 添加自定义边距参数，默认值为20

    def process(self, openpose_image, depth_image, margin):
        # 调整深度图的分辨率以匹配OpenPose图像
        depth_img_resized = cv2.resize(depth_image, (openpose_image.shape[1], openpose_image.shape[0]))

        # 创建基于OpenPose图像的掩膜
        gray_img = cv2.cvtColor(openpose_image, cv2.COLOR_BGR2GRAY)
        non_black_points = np.argwhere(gray_img > 0)
        mask = np.zeros_like(gray_img)
        mask[gray_img > 0] = 255

        # 添加边距
        kernel = np.ones((margin, margin), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        # 应用掩膜到调整后的深度图
        processed_depth_img = cv2.bitwise_and(depth_img_resized, depth_img_resized, mask=mask_dilated)

        return processed_depth_img

# 注册自定义Node
def register():
    Node.register_node(OpenPoseDepthProcessorNode)
