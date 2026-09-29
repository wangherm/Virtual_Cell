"""Bounded-memory HTTPS range reader for a pinned public GCS object.

No whole-object checksum is claimed: object generation and each byte-range
response are checked. Aggregated outputs have independent content hashes.
"""
import io
import time
from collections import OrderedDict
from urllib.parse import parse_qs, urlparse

import requests
from urllib3.exceptions import HTTPError as StreamError


class RangeFile(io.RawIOBase):
    def __init__(self, spec, block_size=8 * 1024**2, cache_blocks=8, session=None, sleep=time.sleep):
        super().__init__()
        if block_size < 1 or cache_blocks < 1 or spec["size"] < 1:
            raise ValueError("Invalid range-reader limits")
        url = urlparse(spec["url"])
        if url.scheme != "https" or parse_qs(url.query).get("generation") != [spec["generation"]]:
            raise ValueError("Remote source must use HTTPS and the pinned object generation")
        self.spec, self.position = spec, 0
        self.block_size, self.cache_blocks = block_size, cache_blocks
        self.cache = OrderedDict()
        self.session, self.sleep = session or requests.Session(), sleep
        self.bytes_received, self.requests_made = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        pos = offset + (0 if whence == 0 else self.position if whence == 1 else self.spec["size"])
        if pos < 0:
            raise ValueError("Negative seek")
        self.position = pos
        return pos

    def block(self, start):
        if start in self.cache:
            self.cache.move_to_end(start)
            return self.cache[start]
        end = min(start + self.block_size, self.spec["size"]) - 1
        for attempt in range(4):
            try:
                with self.session.get(self.spec["url"], headers={"Range": f"bytes={start}-{end}",
                       "Accept-Encoding": "identity"}, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError("Server ignored Range; refusing to download the entire h5ad")
                    expected = f"bytes {start}-{end}/{self.spec['size']}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("Remote Content-Range/size differs from pinned source")
                    generation = response.headers.get("x-goog-generation")
                    if generation is not None and generation != self.spec["generation"]:
                        raise ValueError("Remote object generation changed")
                    value = response.raw.read(end - start + 2)
                    if len(value) != end - start + 1:
                        raise requests.exceptions.ChunkedEncodingError("Incomplete range response")
                break
            except (requests.RequestException, OSError, StreamError) as exc:
                if attempt == 3:
                    raise
                print(f"VCC RANGE RETRY {attempt+1}/3 offset={start}: {type(exc).__name__}", flush=True)
                self.sleep(2**(attempt+1))
        self.bytes_received += len(value)
        self.requests_made += 1
        self.cache[start] = value
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return value

    def read(self, size=-1):
        end = self.spec["size"] if size < 0 else min(self.spec["size"], self.position + size)
        parts = []
        while self.position < end:
            start = self.position // self.block_size * self.block_size
            value = self.block(start)
            part = value[self.position - start:min(end - start, len(value))]
            parts.append(part)
            self.position += len(part)
        return b"".join(parts)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)

    def close(self):
        if not self.closed:
            self.session.close()
            self.cache.clear()
        super().close()
