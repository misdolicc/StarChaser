"""极简只读 Web 服务：在局域网浏览报表、图片、视频 (python 标准库，无需 nginx)。"""
from __future__ import annotations

import functools
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".h264": "video/h264", ".jsonl": "text/plain",
                      ".json": "application/json; charset=utf-8", ".csv": "text/csv; charset=utf-8"}

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def end_headers(self):
        if self.path.split("?")[0].endswith((".json", ".html")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def serve(data_dir, host="0.0.0.0", port=8080):
    handler = functools.partial(_Handler, directory=str(data_dir))
    with ThreadingHTTPServer((host, port), handler) as httpd:
        log.info("报表服务: http://%s:%d/  (目录 %s)", host, port, data_dir)
        httpd.serve_forever()
