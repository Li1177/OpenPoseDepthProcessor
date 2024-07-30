from setuptools import setup, find_packages

setup(
    name='OpenPoseDepthProcessor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python-headless',
        'numpy'
    ],
    entry_points={
        'comfyui.nodes': [
            'OpenPoseDepthProcessor = OpenPoseDepthProcessor'
        ]
    }
)
