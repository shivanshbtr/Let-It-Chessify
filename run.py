"""
Let It Chessify — Backend Launcher
==============================
Run this from the project root instead of uvicorn directly.
Handles all path setup correctly on Windows/Mac/Linux.

Usage:
    python run.py
    python run.py --port 8001
    python run.py --no-reload
"""
import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in path BEFORE uvicorn spawns subprocesses
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set env var so subprocess workers inherit the path
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn

def parse_args():
    p = argparse.ArgumentParser(description="Chess OCR Backend")
    p.add_argument("--host",      default="0.0.0.0",   help="Host (default 0.0.0.0)")
    p.add_argument("--port",      type=int, default=8000, help="Port (default 8000)")
    p.add_argument("--no-reload", action="store_true",  help="Disable auto-reload")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting Chess OCR backend on http://localhost:{args.port}")
    print(f"Swagger UI: http://localhost:{args.port}/docs")
    print(f"Project root: {project_root}")
    uvicorn.run(
        "backend.main:app",
        host    = args.host,
        port    = args.port,
        reload  = False,
    )
