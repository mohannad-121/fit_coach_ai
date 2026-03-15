from __future__ import annotations

"""
Week-3 backend entrypoint.

This file exposes the FastAPI app required in the deliverables.
All route definitions live in main.py and are imported here.
"""

from config import API_HOST, API_PORT
from main import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False)
