"""One command to start the demo:  python demo/run.py   (then the browser opens by itself)."""
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"

if __name__ == "__main__":
    print(f"Loading models... the demo will open at {URL}")
    from server import app  # noqa: E402  (loads both models)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
