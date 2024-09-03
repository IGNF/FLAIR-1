from setuptools import setup, find_packages

setup(
    name='flair',
    version='0.2.0',  # Change this as needed or implement dynamic version reading from VERSION file
    author='Anatol Garioud & Samy Khelifi',
    author_email='ai-challenge@ign.fr',
    description='baseline and demo code from the flair #1 challenge',
    long_description='French Land-cover from Arospace ImageRy',
    long_description_content_type='French Land-cover from Arospace ImageRy',
    url='https://github.com/IGNF/FLAIR-1',
    project_urls={
        'Bug Tracker': 'https://github.com/IGNF/FLAIR-1'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    package_dir={'detect': '.'},
    packages=find_packages(where='.'),
    python_requires='>=3.10',
    install_requires=[
        'geopandas>=0.10',
        'rasterio>=1.1.5',
        'omegaconf',
        'jsonargparse',
        "matplotlib>=3.8.2",
        "pandas>=2.1.4",
        "scikit-image>=0.22.0",
        "pillow>=9.3.0",
        "torchmetrics==1.2.0",
        "pytorch-lightning==2.1.1",
        "segmentation-models-pytorch==0.3.3",
        "albumentations==1.3.1",
        "tensorboard==2.15.1",
        "transformers>=4.41.2"
    ],
    include_package_data=True,
    package_data={
        '': ['*.yml']
    },
    entry_points={
        'console_scripts': [
            'flair-detect=src.zone_detect.main:main',
            'flair=src.flair.main:main'
        ]
    }
)

