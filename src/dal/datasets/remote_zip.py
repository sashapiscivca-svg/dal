"""Read single members of a remote zip archive over HTTP Range requests, without downloading the whole archive.

`RangeFile` behaves like a seekable file, so `zipfile.ZipFile` reads the central directory from the end of the archive and
extracts only the members you ask for. This makes it practical to take a few hundred frames out of a 30 GB dataset archive.
Authentication is HTTP Basic (a public Nextcloud share id as the user name, no password). Standard library only.
"""
from __future__ import annotations

import base64
import http.client
import io
import struct
import urllib.request
import zlib


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, user: str = "", password: str = "", block: int = 1 << 20):
        self.url, self.block = url, block
        self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self._cache: tuple[int, bytes] = (-1, b"")
        self.requests = 0
        self.bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def fetch(self, start: int, end: int) -> bytes:
        """Bytes `[start, end]` (inclusive), with three retries."""
        req = urllib.request.Request(self.url, headers={"Authorization": self.auth, "Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.requests += 1
                self.bytes += len(data)
                return data
            except (OSError, http.client.HTTPException):
                if attempt == 3:
                    raise
        return b""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while len(out) < n:
            blk = self.pos // self.block
            if self._cache[0] != blk:
                s = blk * self.block
                self._cache = (blk, self.fetch(s, min(s + self.block, self.size) - 1))
            off = self.pos - blk * self.block
            chunk = self._cache[1][off:off + (n - len(out))]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def read_member(f: RangeFile, info) -> bytes:
    """One Range request per archive member (local header + data) instead of block reads: 3-4x less traffic for ~0.5 MB files."""
    head = f.fetch(info.header_offset, min(info.header_offset + 30 + 512 + info.compress_size, f.size - 1))
    n, m = struct.unpack("<HH", head[26:30])
    start = 30 + n + m
    raw = head[start:start + info.compress_size]
    if len(raw) < info.compress_size:                       # the 512-byte margin was not enough (large extra field)
        raw += f.fetch(info.header_offset + start + len(raw), info.header_offset + start + info.compress_size - 1)
    return raw if info.compress_type == 0 else zlib.decompress(raw, -15)
