import os, sys
# sys.path.append(os.path.dirname(__file__))
from pathlib import Path
from setuptools import setup, find_packages

current_dir = Path(__file__).parent
readme_path = current_dir / 'README.md'
readme = readme_path.read_text()
requirements_path = current_dir / 'requirements.txt'
requirements = requirements_path.read_text().splitlines()
setup(
    name='flair',
    version='0.1.0',  # Change this as needed or implement dynamic version reading from VERSION file
    author='anatol garioud & samy khelifi',
    author_email='ai-challenge@ign.fr',
    description='baseline and demo code for flair 1 challenge',
    long_description='French Land-cover from Arospace ImageRy',
    long_description_content_type=readme,
    url='https://github.com/IGNF/FLAIR-1-AI-Challenge',
    project_urls={
        'Bug Tracker': 'https://github.com/IGNF/FLAIR-1-AI-Challenge'
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
        'jsonargparse'
    ],
    include_package_data=True,
    package_data={
        '': ['*.yml']
    },
    entry_points={
        'console_scripts': [
            'flair-detect=src.detect.main:main',
            'flair-train=main:main'
        ]
    }
)

# Configure flake8 and pytest as needed, possibly through other means
# as setup.py doesn't directly support these configurations.
