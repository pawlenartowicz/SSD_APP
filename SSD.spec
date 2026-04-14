# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SSD.

Build with:
    pyinstaller SSD.spec --clean --noconfirm

Requires:
    (no extra pip installs needed)
"""

import re
import sys
import glob
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# ── Version (single source of truth: pyproject.toml) ──
_pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
_ver_match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject, re.MULTILINE)
_version = _ver_match.group(1) if _ver_match else "0.0.0"

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

# Collect data files for NLP packages that ship dictionaries/lookup tables
nlp_datas = []
nlp_datas += collect_data_files('spacy_lookups_data')

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
    ] + nlp_datas,
    hiddenimports=[
        # Core packages — ssdiff v1.0.0 module structure
        'ssdiff',
        'ssdiff.ssd',
        'ssdiff.corpus',
        'ssdiff.embeddings',
        'ssdiff.results',
        'ssdiff.lang_config',
        'ssdiff.backends',
        'ssdiff.backends.pls',
        'ssdiff.backends.pca_sweep',
        'ssdiff.backends._sweep_math',
        'ssdiff.backends.group',
        'ssdiff.utils',
        'ssdiff.utils.text',
        'ssdiff.utils.vectors',
        'ssdiff.utils.neighbors',
        'ssdiff.utils.snippets',
        'ssdiff.utils.lexicon',
        'ssdiff.utils.math',

        # GUI
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtNetwork',

        # Data processing
        'pandas',
        'numpy',
        'numpy.core._methods',
        'numpy.linalg',

        # NLP
        'spacy',

        # NLP lookup data
        'spacy_lookups_data',


        # Export (CSV via pandas + openpyxl for xlsx reading)
        'openpyxl',
        'openpyxl.cell',
        'openpyxl.styles',
        'openpyxl.utils',
        'et_xmlfile',

        # Threading
        'concurrent.futures',
        'concurrent.futures.thread',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthooks/rthook_openblas.py'] if sys.platform == 'darwin' else [],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'IPython',
        'jupyter',
        'pytest',
        # Conflicting Qt bindings
        'PyQt5',
        'PyQt6',
        # Unused PySide6/Qt modules (app only needs QtCore, QtGui, QtWidgets, QtNetwork)
        'PySide6.QtWebEngine',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebChannel',
        'PySide6.QtQuick',
        'PySide6.QtQuick3D',
        'PySide6.QtQuickWidgets',
        'PySide6.QtQml',
        'PySide6.Qt3DCore',
        'PySide6.Qt3DRender',
        'PySide6.Qt3DExtras',
        'PySide6.QtOpenGL',
        'PySide6.QtOpenGLWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.QtDesigner',
        'PySide6.QtCharts',
        'PySide6.QtGraphs',
        'PySide6.QtDataVisualization',
        'PySide6.QtBluetooth',
        'PySide6.QtLocation',
        'PySide6.QtPositioning',
        'PySide6.QtSensors',
        'PySide6.QtSerialPort',
        'PySide6.QtSql',
        'PySide6.QtSvg',
        'PySide6.QtSvgWidgets',
        'PySide6.QtPdf',
        'PySide6.QtPdfWidgets',
        'PySide6.QtRemoteObjects',
        'PySide6.QtShaderTools',
        'PySide6.QtSpatialAudio',
        'PySide6.QtNfc',
        'PySide6.QtTest',
        # Heavy packages from system Python or transitive deps
        'scipy',
        'matplotlib',
        'PIL',
        'Pillow',
        'kiwisolver',
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

# ── Strip unused Qt shared libraries that hooks pull in anyway ──
_qt_keep = {
    'Core', 'Gui', 'Widgets', 'Network',
    'DBus',         # QtWidgets runtime dep on Linux
    'XcbQpa',       # X11 platform plugin
    'WaylandClient', 'WlShellIntegration',  # Wayland support
    'OpenGL',       # needed by QtGui on Linux
    'Svg',          # icon rendering
    'PrintSupport', # QWidget print dialogs
    'Concurrent',   # used by some spaCy pipelines internally
}

def _is_unwanted_qt(name):
    """Return True for Qt .so/.dll/.dylib files we don't need."""
    basename = Path(name).name
    # Match libQt6<Module>.so.6  /  Qt6<Module>.dll  /  Qt<Module>.abi3.so
    for pat in (r'libQt6(\w+)\.so', r'Qt6(\w+)\.dll', r'Qt(\w+)\.abi3\.so'):
        m = re.match(pat, basename)
        if m and m.group(1) not in _qt_keep:
            return True
    # avcodec / avformat / avutil — only needed by WebEngine / Multimedia
    # NOTE: ICU libs (libicu*) must NOT be excluded — QtCore depends on them
    if re.match(r'lib(avcodec|avformat|avutil|swresample)', basename):
        return True
    return False

a.binaries = [b for b in a.binaries if not _is_unwanted_qt(b[0])]
a.datas    = [d for d in a.datas    if not _is_unwanted_qt(d[0])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == 'darwin':
    # macOS: onedir mode — files live inside the .app bundle alongside the binary.
    # Faster launch, better Gatekeeper/security compatibility.
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name='SSD',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=True,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=app_icon,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=True,
        upx=True,
        upx_exclude=[],
        name='SSD',
    )

    app = BUNDLE(
        coll,
        name='SSD.app',
        icon=app_icon,
        bundle_identifier='com.ssd.app',
        info_plist={
            'CFBundleShortVersionString': _version,
            'NSHighResolutionCapable': True,
        },
    )

else:
    # Windows / Linux: onefile mode — single executable, no separate folder needed.
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
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=app_icon,
    )
