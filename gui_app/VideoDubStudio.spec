# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import shutil
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path.cwd()
NUMPY_CORE_SUBMODULES = collect_submodules("numpy._core")
EJS_DATAS = collect_data_files("yt_dlp_ejs", includes=["**/*.js"])
BUNDLED_BINARIES = []
for tool in ("ffmpeg", "ffprobe", "deno"):
    path = shutil.which(tool)
    if path:
        BUNDLED_BINARIES.append((path, "bin"))
if len(BUNDLED_BINARIES) < 3:
    raise RuntimeError("ffmpeg/ffprobe/deno not found on build machine; install them before packaging")

a = Analysis(
    [str(ROOT / "gui_app" / "main.py")],
    pathex=[str(ROOT), str(ROOT / "gui_app")],
    binaries=BUNDLED_BINARIES,
    datas=EJS_DATAS,
    hiddenimports=NUMPY_CORE_SUBMODULES,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "tensorflow",
        "whisperx",
        "pyannote",
        "speechbrain",
        "faster_whisper",
        "lightning",
        "pytorch_lightning",
        "av",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoDubStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VideoDubStudio",
)
app = BUNDLE(
    coll,
    name="VideoDubStudio.app",
    icon=None,
    bundle_identifier="com.codex.videodubstudio",
)
