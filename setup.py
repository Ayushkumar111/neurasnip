from setuptools import setup, find_packages

setup(
    name='NeuraGallery',
    version='0.1.0',
    author='Ayush Kumar',
    author_email='workmailayush04@gmail.com',
    description='AI-powered image search platform',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'streamlit',
        'faiss-cpu',
        'pytesseract',
        'paddleocr',
        'numpy',
        'opencv-python',
        'Pillow',
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)