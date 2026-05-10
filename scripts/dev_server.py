"""Start the local FastAPI dev server on an available port."""

from __future__ import annotations

import socket
import sys
import os
from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
PREFERRED_PORTS = [7860, 5000, 8010, 8020, 8888]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ["PYTHONPATH"] = (
    str(PROJECT_ROOT)
    if not os.environ.get("PYTHONPATH")
    else str(PROJECT_ROOT) + os.pathsep + os.environ["PYTHONPATH"]
)


def can_bind(host: str, port: int) -> tuple[bool, str | None]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False, str(exc)
    return True, None


def choose_port() -> int:
    for port in PREFERRED_PORTS:
        ok, _ = can_bind(HOST, port)
        if ok:
            return port

    for port in range(9000, 9101):
        ok, _ = can_bind(HOST, port)
        if ok:
            return port

    raise RuntimeError("No available local port was found. Check firewall or port reservations.")


def main() -> int:
    port = choose_port()
    print(f"Starting Skill Agent Lab at http://{HOST}:{port}")
    print("Tip: Windows may reserve port 8000, so this script chooses a usable port automatically.")
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=port,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
