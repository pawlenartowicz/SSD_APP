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

# Collect MKL and runtime DLLs required by numpy/scipy (Anaconda, Windows only)
mkl_binaries = []
if sys.platform == 'win32':
    for mkl_dir in [
        Path(sys.prefix) / 'Library' / 'bin',
        Path(sys.prefix) / 'DLLs',
    ]:
        if mkl_dir.exists():
            for pattern in ['mkl*.dll', 'libiomp*.dll', 'libblas*.dll', 'liblapack*.dll', 'vcomp*.dll']:
                for dll in mkl_dir.glob(pattern):
                    mkl_binaries.append((str(dll), '.'))

# On Linux, bundle libexpat.so from the build environment so pyexpat.so gets
# the same version it was compiled against (≥2.6.0) rather than the system one.
if sys.platform == 'linux':
    for pattern in ['libexpat.so.1*', 'libexpat.so.1']:
        for lib in (Path(sys.prefix) / 'lib').glob(pattern):
            mkl_binaries.append((str(lib), '.'))

# Icon: .ico on Windows, .icns on macOS, None on Linux
if sys.platform == 'win32':
    app_icon = 'ssdiff_gui/resources/icon.ico'
elif sys.platform == 'darwin':
    app_icon = 'ssdiff_gui/resources/icon.icns'
else:
    app_icon = None

# Windows-only Analysis kwargs
analysis_kwargs = {}
if sys.platform == 'win32':
    analysis_kwargs['win_no_prefer_redirects'] = False
    analysis_kwargs['win_private_assemblies'] = False

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

        # Networking (used by ssdiff.preprocess for spaCy model downloads)
        'requests',
        'requests.adapters',
        'requests.auth',
        'requests.exceptions',
        'certifi',
        'charset_normalizer',
        'idna',
        'urllib3',
        'urllib3.util',

        # Threading
        'concurrent.futures',
        'concurrent.futures.thread',

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
    cipher=block_cipher,
    noarchive=False,
    **analysis_kwargs,
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
    strip=(sys.platform != 'win32'),
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == 'darwin'),
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=app_icon,
)

# macOS: create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='SSD.app',
        icon=app_icon,
        bundle_identifier='com.ssd.app',
        info_plist={
            'CFBundleShortVersionString': '1.1.0',
            'NSHighResolutionCapable': True,
        },
    )
