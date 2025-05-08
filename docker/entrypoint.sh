#!/bin/sh

uv run src/mcp_server.py --port 8001 &
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --app-dir src
