"""HTTP transport adapter for the browser GUI."""

from __future__ import annotations

import copy
import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ...config import PROJECT_ROOT
from .web_state import WebGuiState
from .workers import (
    run_beam_smoke_task,
    run_field_bake_task,
    run_field_diagnostics_task,
    run_report_task,
    run_simulation_task,
)


class WebGuiHandler(BaseHTTPRequestHandler):
    """Route browser requests into a locked :class:`WebGuiState`."""

    state: WebGuiState
    index_html = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _send_text(
        self,
        text: str,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe_artifact_path(self, path_text: str) -> Path:
        path = Path(path_text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise PermissionError(
                f"Artifact path is outside the workspace: {resolved}"
            ) from exc
        return resolved

    def _send_artifact(self, query: str) -> None:
        try:
            params = parse_qs(query)
            path = self._safe_artifact_path(params.get("path", [""])[0])
            data = path.read_bytes()
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        content_type = (
            mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(self.index_html)
        elif parsed.path == "/api/state":
            with self.state.lock:
                self._send_json(self.state.snapshot())
        elif parsed.path == "/artifact":
            self._send_artifact(parsed.query)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _start_report(self) -> None:
        if self.state.last_report_snapshot is None:
            self.state.append(
                "warnings",
                "No completed simulation is available for report generation.",
            )
            return
        self.state.start_task(
            "report",
            run_report_task,
            copy.deepcopy(self.state.config),
            self.state.last_report_snapshot,
        )

    def _dispatch_post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> bool:
        if path == "/api/new-session":
            self.state.__init__()
        elif path == "/api/apply-config":
            self.state.append("logs", "Applied web form configuration.")
        elif path == "/api/save-config":
            self.state.save_config()
        elif path == "/api/load-config":
            self.state.load_config(Path(str(payload.get("path", ""))))
        elif path == "/api/start/beam-smoke":
            self.state.start_task(
                "beam_smoke", run_beam_smoke_task,
                copy.deepcopy(self.state.config),
            )
        elif path == "/api/start/field-bake":
            self.state.status = "FIELDS_BAKING"
            self.state.start_task(
                "field_bake", run_field_bake_task,
                copy.deepcopy(self.state.config),
            )
        elif path == "/api/start/simulation":
            self.state.status = "RUNNING"
            self.state.start_task(
                "simulation", run_simulation_task,
                copy.deepcopy(self.state.config),
            )
        elif path == "/api/field-diagnostics":
            self.state.status = "FIELD_DIAGNOSTICS"
            self.state.start_task(
                "field_diagnostics",
                run_field_diagnostics_task,
                copy.deepcopy(self.state.config),
            )
        elif path == "/api/stop":
            self.state.stop()
        elif path == "/api/report":
            self._start_report()
        else:
            return False
        return True

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            with self.state.lock:
                if parsed.path not in {"/api/load-config", "/api/new-session"}:
                    self.state.apply_payload(payload)
                if not self._dispatch_post(parsed.path, payload):
                    self._send_json(
                        {"error": "not found"}, HTTPStatus.NOT_FOUND
                    )
                    return
                self._send_json({"ok": True})
        except Exception as exc:
            self._send_json(
                {"error": str(exc), "traceback": traceback.format_exc()},
                HTTPStatus.BAD_REQUEST,
            )


__all__ = ["WebGuiHandler"]
