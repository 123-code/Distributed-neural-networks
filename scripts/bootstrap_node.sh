#!/bin/sh
# Install the node's only dependency (NumPy) and start a node.
# Works in Termux (Android / Fire TV), Debian/Ubuntu/Raspberry Pi OS/JetPack, macOS.
#
#   sh scripts/bootstrap_node.sh                # node on port 7070
#   sh scripts/bootstrap_node.sh --port 7071 --name firetv --memory-limit 600
set -e
cd "$(dirname "$0")/.."

if [ -n "$TERMUX_VERSION" ] || [ -d /data/data/com.termux ]; then
    # Termux: NumPy comes prebuilt from Termux's repo (pip would try to compile it).
    python3 -c "import numpy" 2>/dev/null || { pkg update -y && pkg install -y python python-numpy; }
    command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock || true  # keep running when the screen sleeps
else
    command -v python3 >/dev/null 2>&1 || { echo "install Python 3 first"; exit 1; }
    python3 -c "import numpy" 2>/dev/null || \
        python3 -m pip install --user numpy 2>/dev/null || \
        python3 -m pip install --user --break-system-packages numpy || \
        { echo "could not install numpy; try: sudo apt install python3-numpy"; exit 1; }
fi

exec python3 -m dnn.node "$@"
