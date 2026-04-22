"""Unit tests for the /trim-video helper.

We only exercise _trim_with_ffmpeg directly -- it's a thin wrapper around
subprocess.run, so we patch subprocess.run and assert we build the right
argv + translate failures into RuntimeError. The FastAPI route itself
pulls in httpx and the blob upload path, which isn't worth mocking here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow imports from the railway-service package
sys.path.insert(0, str(Path(__file__).parent.parent))

# main.py reads SUPABASE_URL / SUPABASE_SERVICE_KEY at import time. Provide
# dummies before importing main so the module loads cleanly in CI.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")
os.environ.setdefault("EXTRACT_API_KEY", "test-api-key")

_mock_supabase_client = MagicMock()
patch("supabase.create_client", return_value=_mock_supabase_client).start()


class TestTrimWithFfmpeg:
    def test_builds_correct_ffmpeg_command(self):
        from main import _trim_with_ffmpeg

        fake_result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            _trim_with_ffmpeg("/tmp/in.mp4", "/tmp/out.mp4", 1000, 4500)

        assert mock_run.call_count == 1
        argv = mock_run.call_args.args[0]
        assert argv[0] == "ffmpeg"
        # Stream-copy avoids a CPU-bound re-encode; the spec calls for
        # `-c copy` so keep this rigid.
        assert "-c" in argv
        assert argv[argv.index("-c") + 1] == "copy"
        # -ss in seconds
        assert "-ss" in argv
        assert argv[argv.index("-ss") + 1] == "1.000"
        # -t is duration (end - start) in seconds
        assert "-t" in argv
        assert argv[argv.index("-t") + 1] == "3.500"
        assert argv[-1] == "/tmp/out.mp4"

    def test_raises_on_nonzero_exit(self):
        from main import _trim_with_ffmpeg

        fake_result = MagicMock(returncode=1, stderr="ffmpeg: bad input")
        with patch("subprocess.run", return_value=fake_result):
            try:
                _trim_with_ffmpeg("/tmp/in.mp4", "/tmp/out.mp4", 0, 1000)
            except RuntimeError as e:
                assert "ffmpeg trim failed" in str(e)
                assert "bad input" in str(e)
            else:
                raise AssertionError("expected RuntimeError on ffmpeg failure")

    def test_raises_on_nonpositive_duration(self):
        from main import _trim_with_ffmpeg

        # end_ms <= start_ms is guarded at the route layer too, but the
        # helper rejects it defensively so we can't accidentally ship a
        # zero-length clip.
        try:
            _trim_with_ffmpeg("/tmp/in.mp4", "/tmp/out.mp4", 5000, 4000)
        except RuntimeError as e:
            assert "end_ms" in str(e)
        else:
            raise AssertionError("expected RuntimeError for inverted range")


class TestUploadFileToBlobPath:
    """_upload_file_to_blob should PUT to the supplied blob path verbatim
    (minus any leading slash)."""

    def test_strips_leading_slash(self):
        from main import _upload_file_to_blob

        async def run():
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value={"url": "https://x/y"})
            mock_client.put = MagicMock()

            async def fake_put(url, **kwargs):
                # Record the URL then return a response-like object
                fake_put.called_with_url = url
                return mock_resp

            mock_client.put = fake_put

            class FakeCtx:
                async def __aenter__(self):
                    return mock_client

                async def __aexit__(self, *args):
                    return False

            with patch.dict(os.environ, {"BLOB_READ_WRITE_TOKEN": "tok"}), \
                 patch("builtins.open", MagicMock()), \
                 patch("httpx.AsyncClient", return_value=FakeCtx()):
                url = await _upload_file_to_blob("/tmp/does-not-matter.mp4", "/baseline-trims/x.mp4")
            assert url == "https://x/y"
            assert fake_put.called_with_url.endswith("/baseline-trims/x.mp4")
            # Make sure we stripped the leading slash (no double-slash in path)
            assert "//baseline-trims" not in fake_put.called_with_url

        import asyncio

        asyncio.run(run())
