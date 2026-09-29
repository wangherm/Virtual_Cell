import io

import pytest

from vcell.remote_h5 import RangeFile


class Response:
    def __init__(self, data, start, end, *, status=206, generation="123"):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}", "x-goog-generation": generation}
        self.raw = io.BytesIO(data[start:end+1])
    def raise_for_status(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Session:
    def __init__(self, data, **options):
        self.data, self.options, self.calls = data, options, []
    def get(self, url, headers, **kwargs):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        self.calls.append((start, end))
        return Response(self.data, start, end, **self.options)
    def close(self): pass


def spec(data):
    return {"url": "https://example.test/counts.h5ad?generation=123", "generation": "123", "size": len(data)}


def test_range_file_matches_local_seek_read_and_bounds_cache():
    data = bytes(range(53))
    session = Session(data)
    with RangeFile(spec(data), block_size=8, cache_blocks=2, session=session) as f:
        assert f.read(11) == data[:11]
        assert f.seek(-3, 1) == 8
        assert f.read(6) == data[8:14]
        f.seek(-5, 2)
        b = bytearray(10)
        assert f.readinto(b) == 5 and b[:5] == data[-5:]
        assert len(f.cache) <= 2
        assert f.read(2) == b""
        assert session.calls.count((8, 15)) == 1


@pytest.mark.parametrize("options", [{"status": 200}, {"generation": "999"}])
def test_reject_ignored_range_or_changed_generation(options):
    data = b"abcdefgh"
    with RangeFile(spec(data), block_size=4, session=Session(data, **options)) as f:
        with pytest.raises(ValueError):
            f.read(1)
