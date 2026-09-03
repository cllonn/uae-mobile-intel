"""
Minimal local server for demo day: serves the dashboard's static files exactly like
`python -m http.server`, plus one JSON API route the AI Copilot chat panel calls.

Why this exists (replacing the plain `python -m http.server` used up to this point): the
dashboard is a static, self-contained HTML file with every map/score number embedded at build
time -- and that is still true, nothing about the map changed. The Copilot panel is different
in kind: it answers open-ended and zone-scoped questions chosen at runtime by whoever is
running the demo, which cannot be precomputed into the page the way the map data is. Answering
for real means running the real `src/copilot.py` -- Python -- which a static HTML file cannot
do on its own. Faking the chat box client-side (a canned-response widget with no real routing,
grounding, or refusal logic behind it) was explicitly ruled out; this server is the smallest
thing that closes the gap honestly: stdlib only (`http.server`), no new dependency, no
framework. It still serves every static file exactly as before, so the demo-day workflow barely
changes -- one command instead of `python -m http.server 8000`, same port, same URLs.

Run from the repo root: `python -m src.serve_dashboard [port]` (default 8000). Then open, in an
incognito window, the URL `python -m src.build_dashboard` printed at the end of its own run:
    http://localhost:8000/data/processed/uae_dashboard.html?v=<build id>

API:
    POST /api/copilot   {"question": str, "zone_id": str|null, "quarter": str|null}
      -> 200 {"question", "tool", "tool_args", "tool_result", "answer", "mode"}
         (exactly `src.copilot.answer_question`'s return value -- this file adds no routing,
         grounding, or narration logic of its own, it only exposes the existing one over HTTP)
      -> 400 on a malformed request body, 500 with the exception message on an internal error --
         never a silent failure; the chat panel shows whatever this returns, verbatim.

`src/copilot_tools.py` caches `zone_priority.parquet` in-process (unchanged by this file) -- if
you re-run `run_pipeline.py` or `build_dashboard.py` while this server is running, restart it
so the Copilot picks up the new data, the same way you would already reopen the dashboard URL
with a fresh `?v=` after any rebuild.
"""
import datetime as dt
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src import copilot

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8000


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the repo root regardless of the caller's cwd -- `python -m src.X` always
        # resolves relative to wherever python was invoked from otherwise, which is exactly
        # the kind of "which copy of the file am I looking at" surprise the caching fix
        # (earlier this session) was about avoiding.
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        # SimpleHTTPRequestHandler logs every static asset request to stderr by default --
        # fine for a handful of files, noisy in a live demo terminal. Keep API calls visible.
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/copilot":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "malformed_request_body"})
            return

        question = body.get("question")
        if not question or not isinstance(question, str):
            self._send_json(400, {"error": "missing_or_invalid_question"})
            return

        try:
            # The one and only call into the copilot -- everything the response contains
            # (tool chosen, structured result, narrated text) is exactly what
            # `src/copilot.answer_question` itself produces. Nothing computed here.
            result = copilot.answer_question(
                question,
                zone_id=(body.get("zone_id") or None),
                quarter=(body.get("quarter") or None),
            )
        except Exception as exc:  # keep the demo alive -- surface the error, don't crash the server
            self._send_json(500, {"error": "copilot_error", "message": str(exc)})
            return

        self._send_json(200, result)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("", port), Handler)
    print(f"Serving {REPO_ROOT} at http://localhost:{port}  (static files + POST /api/copilot)")

    # Cache-bust with the dashboard file's own current mtime -- same reasoning as
    # build_dashboard.py's own printed URL (see its "Demo-day serving" docstring): a plain,
    # un-cache-busted URL is exactly the thing that showed a stale build in the browser-caching
    # investigation earlier. Printing that here too would silently reintroduce it, so this
    # server computes its own correct value from disk rather than ever printing a bare link.
    dashboard_path = REPO_ROOT / "data" / "processed" / "uae_dashboard.html"
    if dashboard_path.exists():
        cache_bust = dt.datetime.fromtimestamp(
            dashboard_path.stat().st_mtime, dt.timezone.utc
        ).strftime("%Y-%m-%dT%H-%M-%SZ")
        print(f"Dashboard: http://localhost:{port}/data/processed/uae_dashboard.html?v={cache_bust}")
    else:
        print(f"Dashboard: (not built yet -- run `python -m src.build_dashboard` first)")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
