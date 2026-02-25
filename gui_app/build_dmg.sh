#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GUI_DIR="$ROOT/gui_app"
DIST_DIR="$ROOT/dist"
DMG_STAGING="$GUI_DIR/dmg_staging"
APP_NAME="VideoDubStudio"

cd "$ROOT"

if [ ! -d "$ROOT/.venv312" ]; then
  echo "missing venv: $ROOT/.venv312"
  exit 1
fi

source "$ROOT/.venv312/bin/activate"
pip install -r "$GUI_DIR/requirements-gui.txt"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  pyinstaller --noconfirm --clean "$GUI_DIR/VideoDubStudio.spec"
fi

APP_BIN_DIR="$DIST_DIR/$APP_NAME.app/Contents/Resources/bin"
if [ ! -x "$APP_BIN_DIR/ffmpeg" ] || [ ! -x "$APP_BIN_DIR/ffprobe" ] || [ ! -x "$APP_BIN_DIR/deno" ]; then
  echo "missing bundled ffmpeg/ffprobe/deno under: $APP_BIN_DIR"
  exit 1
fi

rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$DIST_DIR/$APP_NAME.app" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

DMG_PATH="$DIST_DIR/$APP_NAME.dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_PATH"

echo "dmg built: $DMG_PATH"
