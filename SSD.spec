# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SSD.

Build with:
    pyinstaller SSD.spec --clean --noconfirm

Requires:
    pip install spacy-pkuseg sudachipy sudachidict-core
"""

import sys
import glob
from pathlib import Path

block_cipher = None

# Get the directory containing this spec file
SPEC_DIR = Path(SPECPATH) if 'SPECPATH' in dir() else Path('.')

# Collect MKL and runtime DLLs required by numpy/scipy (Anaconda)
mkl_binaries = []
for mkl_dir in [
    Path(sys.prefix) / 'Library' / 'bin',
    Path(sys.prefix) / 'DLLs',
]:
    if mkl_dir.exists():
        for pattern in ['mkl*.dll', 'libiomp*.dll', 'libblas*.dll', 'liblapack*.dll', 'vcomp*.dll']:
            for dll in mkl_dir.glob(pattern):
                mkl_binaries.append((str(dll), '.'))

a = Analysis(
    ['ssdiff_gui/main.py'],
    pathex=[str(SPEC_DIR)],
    binaries=mkl_binaries,
    datas=[
        # Include resources folder if it exists
        ('ssdiff_gui/resources', 'resources'),
    ],
    hiddenimports=[
        # Core packages
        'ssdiff',
        'ssdiff.core',
        'ssdiff.preprocess',
        'ssdiff.clusters',
        'ssdiff.crossgroup',
        'ssdiff.io_utils',
        'ssdiff.lexicon',
        'ssdiff.snippets',
        'ssdiff.sweep',
        'ssdiff.utils',

        # GUI
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',

        # Data processing
        'pandas',
        'numpy',
        'numpy.core._methods',
        'numpy.linalg',
        'scipy',
        'scipy.sparse',
        'scipy.spatial',
        'scipy.linalg',
        'scipy.stats',
        'scipy.special',
        'scipy.optimize',
        'scipy._lib',

        # NLP
        'gensim',
        'gensim.models',
        'gensim.models.keyedvectors',
        'spacy',

        # Asian language tokenizer backends
        'spacy_pkuseg',       # Chinese
        'sudachipy',          # Japanese
        'sudachidict_core',   # Japanese

        # ML
        'sklearn',
        'sklearn.cluster',
        'sklearn.decomposition',
        'sklearn.metrics',
        'sklearn.preprocessing',
        'sklearn.utils',
        'sklearn.utils._cython_blas',

        # Export
        'docx',

        # Standard library that might be missed
        'pickle',
        'json',
        'pathlib',
        'datetime',
        'dataclasses',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'IPython',
        'jupyter',
        'pytest',
        # Conflicting Qt bindings
        'PyQt5',
        'PyQt6',
        # Heavy packages pulled in transitively
        'torch',
        'torchvision',
        'torchaudio',
        'sphinx',
        'docutils',
        'babel',
        'pygments',
        'numba',
        'llvmlite',
        'dask',
        'distributed',
        'xarray',
        'botocore',
        'boto3',
        'zmq',
        'pygame',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SSD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ssdiff_gui/resources/icon.ico',
)
