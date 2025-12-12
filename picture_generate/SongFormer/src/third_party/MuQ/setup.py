from setuptools import setup, find_packages

setup(
    name='muq',  # Name of the package
    version='0.1.0',  # Version of the package
    packages=find_packages(where='src'),  # Automatically discover packages under the 'src' directory
    package_dir={'': 'src'},  # Specify the root directory for packages as 'src'
    include_package_data=True,  # Include additional files, such as static files
    install_requires=[  # List of dependencies
        "einops",
        "librosa",
        "nnAudio", 
        "numpy", 
        "soundfile", 
        "torch", 
        "torchaudio", 
        "tqdm", 
        "transformers", 
        "easydict",
        "x_clip", 
    ],
    author='Haina Zhu',  # Author name
    author_email='juhayna@qq.com',  # Author email address
    description='MuQ: A deep learning model for music and text',  # Short description of the package
    long_description=open('README.md', encoding='utf-8').read(),  # Long description from the README file
    long_description_content_type='text/markdown',  # Format of the long description (Markdown)
    url='https://github.com/tencent-ailab/MuQ',  # Project URL
    classifiers=[
        'Programming Language :: Python :: 3',  # Python 3 support
        'License :: OSI Approved :: MIT License',  # License type
        'Operating System :: OS Independent',  # Supports all operating systems
    ],
    python_requires='>=3.8',  # Supported Python version
)
