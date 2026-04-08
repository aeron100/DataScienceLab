#!/bin/bash
set -e

# ─────────────────────────────────────────────
# DataScienceLab — DMG Builder
# Usage: bash scripts/build-dmg.sh
# Output: dist/DataScienceLab-1.0.0.dmg
# ─────────────────────────────────────────────

APP_NAME="DataScienceLab"
VERSION="1.0.0"
SCHEME="DataScienceLab"
PROJECT="DataScienceLab.xcodeproj"
CONFIGURATION="Release"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DIST_DIR="$ROOT_DIR/dist"
APP_PATH="$BUILD_DIR/$APP_NAME.app"
DMG_NAME="$APP_NAME-$VERSION.dmg"
DMG_PATH="$DIST_DIR/$DMG_NAME"
STAGING_DIR="$BUILD_DIR/dmg-staging"

echo "==> DataScienceLab DMG Builder"
echo "    Version : $VERSION"
echo "    Output  : $DMG_PATH"
echo ""

# 1. Regenerate Xcode project
echo "==> Regenerating Xcode project..."
cd "$ROOT_DIR"
~/bin/xcodegen generate

# 2. Clean previous build
echo "==> Cleaning previous build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# 3. Build Release
echo "==> Building $CONFIGURATION..."
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "platform=macOS" \
  CONFIGURATION_BUILD_DIR="$BUILD_DIR" \
  build

# 4. Verify .app was produced
if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: Build succeeded but $APP_PATH not found."
  echo "       Check CONFIGURATION_BUILD_DIR output."
  exit 1
fi

echo "==> App built at: $APP_PATH"

# 5. Create staging folder for DMG contents
echo "==> Staging DMG contents..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -R "$APP_PATH" "$STAGING_DIR/"
# Symlink to /Applications so users can drag-and-drop
ln -s /Applications "$STAGING_DIR/Applications"

# 6. Create DMG
echo "==> Creating DMG..."
hdiutil create \
  -volname "$APP_NAME $VERSION" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

# 7. Cleanup staging
rm -rf "$STAGING_DIR"

echo ""
echo "==> Done!"
echo "    DMG: $DMG_PATH"
echo ""
echo "Next steps:"
echo "  - Notarize: xcrun notarytool submit \"$DMG_PATH\" --wait"
echo "  - Staple:   xcrun stapler staple \"$DMG_PATH\""
