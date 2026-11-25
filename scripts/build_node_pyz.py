#!/usr/bin/env python3
"""Bundle the node into ONE file you can copy to any device with Python 3 + NumPy:

    python scripts/build_node_pyz.py            # writes dist/dnn-node.pyz
    adb push dist/dnn-node.pyz /sdcard/          # e.g. to a Fire TV / Android phone
    python dnn-node.pyz --port 7070              # on the device
"""

import os
import shutil
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = os.path.join(ROOT, "dist", "dnn-node.pyz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(os.path.join(ROOT, "dnn"), os.path.join(tmp, "dnn"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        zipapp.create_archive(tmp, out, interpreter="/usr/bin/env python3", main="dnn.node:main",
                              compressed=True)
    print("wrote %s (%d KB)" % (out, os.path.getsize(out) // 1024))


if __name__ == "__main__":
    sys.exit(main())
