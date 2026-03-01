#!/usr/bin/env bash
# =============================================================================
# Create a Cerebras virtualenv for CSZoo (Model Zoo CLI).
#
# Docs baseline:
#   https://training-docs.cerebras.ai/rel-2.4.0/getting-started/setup-and-installation
# (clone modelzoo -> create venv -> pip install --editable ./modelzoo)
#
# Defaults:
#   - Model Zoo ref: Release_2.9.0 (latest stable tag as of 2026-02-25)
#   - Venv path     : ~/venv_cerebras_r290
#
# Usage:
#   bash scripts/cs3/setup_r290_venv.sh
#   MODELZOO_REF=Release_2.8.0 bash scripts/cs3/setup_r290_venv.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VENV_DIR="${VENV_DIR:-$HOME/venv_cerebras_r290}"
MODELZOO_REPO="${MODELZOO_REPO:-https://github.com/Cerebras/modelzoo.git}"
MODELZOO_REF="${MODELZOO_REF:-Release_2.9.0}"
MODELZOO_DIR="${MODELZOO_DIR:-$PROJECT_ROOT/external/modelzoo_r290}"
SHARED_VENV="${SHARED_VENV:-/software/cerebras/shared_venv/venv-38243-3067542}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 2>/dev/null || command -v python3 || true)}"
HAVE_GIT=0
if command -v git >/dev/null 2>&1; then
    HAVE_GIT=1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: python3.11/python3 not found on PATH."
    exit 1
fi

# ALCF proxy defaults (kept only if caller has not already set them).
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

echo "================================================================"
echo " Setting up Cerebras CSZoo venv"
echo "================================================================"
echo "  Venv dir     : $VENV_DIR"
echo "  Modelzoo repo: $MODELZOO_REPO"
echo "  Modelzoo ref : $MODELZOO_REF"
echo "  Modelzoo dir : $MODELZOO_DIR"
echo "  Shared venv  : $SHARED_VENV"
echo "  Python       : $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo "================================================================"

if [[ -d "$VENV_DIR" ]]; then
    echo "ERROR: venv already exists at $VENV_DIR"
    echo "Remove it first to recreate: rm -rf $VENV_DIR"
    exit 1
fi

echo ""
echo "[1/6] Preparing Model Zoo checkout..."
if [[ -d "$MODELZOO_DIR/.git" ]]; then
    if [[ "$HAVE_GIT" -eq 1 ]]; then
        git -C "$MODELZOO_DIR" fetch --tags --quiet || true
        if git -C "$MODELZOO_DIR" rev-parse --verify --quiet "$MODELZOO_REF" >/dev/null; then
            git -C "$MODELZOO_DIR" checkout --quiet "$MODELZOO_REF"
        else
            echo "WARNING: Could not resolve '$MODELZOO_REF' in $MODELZOO_DIR; keeping current checkout."
        fi
    else
        echo "WARNING: git is not available; using existing checkout at $MODELZOO_DIR."
    fi
elif [[ -d "$MODELZOO_DIR" ]]; then
    echo "WARNING: $MODELZOO_DIR exists but is not a git checkout; using it as-is."
else
    if [[ "$HAVE_GIT" -eq 0 ]]; then
        echo "ERROR: git is required to clone Model Zoo into $MODELZOO_DIR."
        echo "       Install git or set MODELZOO_DIR to an existing checkout."
        exit 1
    fi
    mkdir -p "$(dirname "$MODELZOO_DIR")"
    git clone --branch "$MODELZOO_REF" --depth 1 "$MODELZOO_REPO" "$MODELZOO_DIR"
fi

MODELZOO_VERSION="$(sed -n 's/^[[:space:]]*__version__ = "\(.*\)"/\1/p' "$MODELZOO_DIR/setup.py" | head -n1 || true)"
if [[ -z "$MODELZOO_VERSION" ]]; then
    MODELZOO_VERSION="unknown"
fi
echo "Resolved Model Zoo version: $MODELZOO_VERSION"

echo ""
echo "[2/6] Creating virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "[3/6] Installing Model Zoo editable package (no-deps)..."
python -m pip install --editable "$MODELZOO_DIR" --no-deps

echo ""
echo "[4/6] Installing Python dependencies..."
REQ_FILE="$MODELZOO_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    FILTERED_REQ="$(mktemp)"
    grep -Ev '^(cerebras_pytorch==|[[:space:]]*$|#)' "$REQ_FILE" > "$FILTERED_REQ"
    python -m pip install -r "$FILTERED_REQ" \
        --extra-index-url https://download.pytorch.org/whl/cpu || \
        echo "WARNING: Some dependency installs failed. Continuing with best-effort setup."
    rm -f "$FILTERED_REQ"
else
    echo "WARNING: requirements.txt not found at $REQ_FILE; skipping dependency sync."
fi

echo ""
echo "[5/6] Resolving cerebras_pytorch runtime..."
if ! python -c "import cerebras_pytorch" >/dev/null 2>&1; then
    if [[ "$MODELZOO_VERSION" != "unknown" ]]; then
        python -m pip install "cerebras_pytorch==$MODELZOO_VERSION" >/dev/null 2>&1 || true
    fi
fi

if ! python -c "import cerebras_pytorch" >/dev/null 2>&1; then
    SHARED_SP=""
    if [[ -d "$SHARED_VENV" ]]; then
        SHARED_SP="$(find "$SHARED_VENV" -maxdepth 5 -type d -name site-packages | head -n1 || true)"
    fi

    if [[ -n "$SHARED_SP" ]]; then
        VENV_SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
        echo "Linking selected Cerebras packages from shared venv: $SHARED_SP"

        # Do NOT add the whole shared site-packages path to avoid importing
        # incompatible py3.8-only binaries (e.g. tensorflow) into this venv.
        for item in \
            cerebras \
            cerebras_appliance \
            cerebras_pytorch \
            cerebras_torch \
            cerebras_cloud \
            cerebras_cloud_sdk; do
            if [[ -e "$SHARED_SP/$item" ]]; then
                ln -sfn "$SHARED_SP/$item" "$VENV_SP/$item"
                echo "  linked: $item"
            fi
        done

        for meta in \
            cerebras_appliance-*.dist-info \
            cerebras_pytorch-*.dist-info \
            cerebras_torch-*.dist-info \
            cerebras_cloud_sdk-*.dist-info; do
            for found in "$SHARED_SP"/$meta; do
                [[ -e "$found" ]] || continue
                ln -sfn "$found" "$VENV_SP/$(basename "$found")"
                echo "  linked: $(basename "$found")"
            done
        done
    else
        echo "WARNING: Could not import cerebras_pytorch and no shared venv was found."
        echo "         cszoo commands that require Cerebras runtime may fail."
    fi
fi

echo ""
echo "[6/6] Writing activation helper..."
ACTIVATE_HELPER="$VENV_DIR/bin/activate_cerebras"
cat > "$ACTIVATE_HELPER" << 'HELPER'
# Source this file to activate the Cerebras CSZoo environment.
# Usage: source ~/venv_cerebras_r290/bin/activate_cerebras

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_dir/bin/activate"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
# Prevent optional TF backend import through transformers.
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
HELPER

echo ""
echo "================================================================"
echo " Setup complete"
echo "================================================================"
echo "Activate:"
echo "  source $VENV_DIR/bin/activate_cerebras"
echo ""
echo "Smoke test:"
echo "  python -c \"from cerebras.modelzoo.cli.main import main; print('cszoo import OK')\""
echo "  cszoo --help"
echo "================================================================"
