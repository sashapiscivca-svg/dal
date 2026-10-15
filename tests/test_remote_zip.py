import http.server
import io
import re
import threading
import zipfile

from dal.datasets.remote_zip import RangeFile


class _H(http.server.BaseHTTPRequestHandler):
    blob = b""

    def do_GET(self):
        m = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        a, b = (int(m.group(1)), int(m.group(2))) if m else (0, len(self.blob) - 1)
        b = min(b, len(self.blob) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.blob)}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(self.blob[a:b + 1])

    def log_message(self, *_):
        pass


def test_reads_single_member_from_remote_zip_with_few_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i in range(200):
            z.writestr(f"d/{i}.bin", bytes([i % 256]) * 20000)       # 4 MB archive
    _H.blob = buf.getvalue()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        f = RangeFile(f"http://127.0.0.1:{srv.server_port}/x.zip", block=1 << 16)
        z = zipfile.ZipFile(f)
        assert z.read("d/77.bin") == bytes([77]) * 20000
        assert f.bytes < 0.35 * len(_H.blob)                           # only the directory and one member were read
    finally:
        srv.shutdown()


def test_read_member_matches_zipfile_for_stored_and_deflated():
    from dal.datasets.remote_zip import read_member
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.bin", bytes(range(256)) * 400, compress_type=zipfile.ZIP_STORED)
        z.writestr("b.bin", b"hello world " * 5000, compress_type=zipfile.ZIP_DEFLATED)
    _H.blob = buf.getvalue()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        f = RangeFile(f"http://127.0.0.1:{srv.server_port}/y.zip", block=1 << 16)
        z = zipfile.ZipFile(f)
        for name in ("a.bin", "b.bin"):
            assert read_member(f, z.getinfo(name)) == zipfile.ZipFile(io.BytesIO(_H.blob)).read(name)
    finally:
        srv.shutdown()
