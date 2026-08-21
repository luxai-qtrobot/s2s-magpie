#!/bin/bash
# Build the thin arm64 Debian package for LuxAI S2S MAGPIE.
#
# The package contains the application wheel, the qualified Jetson dependency
# lock, configuration, launchers, model provisioning metadata, and the uv
# executable from the build Jetson. The virtual environment and model files
# are deliberately created/downloaded on the target by postinst.
#
# Usage:
#   bash packaging/build_deb.sh

set -euo pipefail

PACKAGE_NAME="luxai-s2s-magpie"
INSTALL_BASE="/opt/luxai/s2s-magpie"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_VERSION="$({
    sed -n '/^\[project\]/,/^\[/s/^version[[:space:]]*=[[:space:]]*"\([^"]*\)"/\1/p' \
        "$PROJECT_DIR/pyproject.toml"
} | head -n 1)"
PACKAGE_VERSION="$PROJECT_VERSION"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

if [[ -z "$PACKAGE_VERSION" ]]; then
    echo "Could not read the project version from pyproject.toml" >&2
    exit 1
fi

for command_name in dpkg dpkg-deb sed uv; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required build command not found: $command_name" >&2
        exit 1
    }
done

ARCH="$(dpkg --print-architecture)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "This package must be built on the target arm64 Jetson (detected: $ARCH)" >&2
    exit 1
fi

LINUX_VERSION="$(lsb_release -rs 2>/dev/null || true)"
if [[ -n "$LINUX_VERSION" ]]; then
    RELEASE="1deb${LINUX_VERSION}"
else
    RELEASE="1"
fi
DEB_FILENAME="${PACKAGE_NAME}_${PACKAGE_VERSION}-${RELEASE}_${ARCH}.deb"

BUILD_DIR="$(mktemp -d /tmp/build-s2s-magpie-XXXXXX)"
STAGING="$BUILD_DIR/staging"
INSTALL_STAGING="$STAGING$INSTALL_BASE"
DIST_DIR="$BUILD_DIR/dist"

cleanup() {
    rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

echo "=== Building $DEB_FILENAME ==="
echo "    project : $PROJECT_DIR"
echo "    staging : $STAGING"

mkdir -p \
    "$INSTALL_STAGING/bin" \
    "$INSTALL_STAGING/config" \
    "$INSTALL_STAGING/packages" \
    "$INSTALL_STAGING/voices" \
    "$STAGING/lib/systemd/system" \
    "$STAGING/DEBIAN" \
    "$DIST_DIR"

echo "--- Building application wheel ---"
uv build --wheel --out-dir "$DIST_DIR" "$PROJECT_DIR"
WHEEL_PATH="$(find "$DIST_DIR" -maxdepth 1 -type f -name 'luxai_s2s_magpie-*.whl' -print -quit)"
if [[ -z "$WHEEL_PATH" ]]; then
    echo "Application wheel was not produced" >&2
    exit 1
fi

echo "--- Copying package payload ---"
cp "$WHEEL_PATH" "$INSTALL_STAGING/packages/"
cp "$PROJECT_DIR/requirements-jetson.lock.txt" "$INSTALL_STAGING/packages/"
cp "$PROJECT_DIR/config/config.yaml" "$INSTALL_STAGING/config/"
cp -a "$PROJECT_DIR/voices/." "$INSTALL_STAGING/voices/"
chmod 644 \
    "$INSTALL_STAGING/packages/$(basename "$WHEEL_PATH")" \
    "$INSTALL_STAGING/packages/requirements-jetson.lock.txt" \
    "$INSTALL_STAGING/config/config.yaml"
find "$INSTALL_STAGING/voices" -type f -exec chmod 644 {} +

UV_PATH="$(command -v uv)"
cp -L "$UV_PATH" "$INSTALL_STAGING/bin/uv"
cp "$SCRIPT_DIR/luxai-s2s-magpie" "$INSTALL_STAGING/bin/"
cp "$SCRIPT_DIR/luxai-s2s-magpie-provision" "$INSTALL_STAGING/bin/"
chmod 755 \
    "$INSTALL_STAGING/bin/uv" \
    "$INSTALL_STAGING/bin/luxai-s2s-magpie" \
    "$INSTALL_STAGING/bin/luxai-s2s-magpie-provision"

cp "$SCRIPT_DIR/luxai-s2s-magpie.service" "$STAGING/lib/systemd/system/"
chmod 644 "$STAGING/lib/systemd/system/luxai-s2s-magpie.service"

cat > "$STAGING/DEBIAN/control" <<EOF
Package: $PACKAGE_NAME
Version: ${PACKAGE_VERSION}-${RELEASE}
Architecture: $ARCH
Maintainer: LuxAI <support@luxai.com>
Depends: python3 (>= 3.12), python3 (<< 3.13), ca-certificates, systemd
Section: misc
Priority: optional
Description: LuxAI MAGPIE-native speech-to-speech service
 Runs the speech-to-speech pipeline over native MAGPIE transports on QTrobot.
 The installer creates an isolated virtual environment and provisions pinned
 model assets without placing Python packages in the system interpreter.
EOF

cat > "$STAGING/DEBIAN/conffiles" <<EOF
$INSTALL_BASE/config/config.yaml
EOF
chmod 644 "$STAGING/DEBIAN/control" "$STAGING/DEBIAN/conffiles"

for maintainer_script in postinst prerm postrm; do
    cp "$SCRIPT_DIR/$maintainer_script" "$STAGING/DEBIAN/$maintainer_script"
    chmod 755 "$STAGING/DEBIAN/$maintainer_script"
done

echo "--- Building Debian archive ---"
dpkg-deb --build --root-owner-group "$STAGING" "$PROJECT_DIR/$DEB_FILENAME"

echo
echo "=== Done ==="
echo "    $PROJECT_DIR/$DEB_FILENAME"
