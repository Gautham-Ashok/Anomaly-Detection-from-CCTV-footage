from setuptools import setup, find_packages

setup(
    name="anomaly-detection-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
    ],
    python_requires=">=3.8",
)