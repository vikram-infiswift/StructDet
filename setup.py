from setuptools import setup, find_packages

setup(
    name="StructDet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "opencv-python",
        "numpy",
        "Pillow",
    ],
    python_requires=">=3.8",
)
