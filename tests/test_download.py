"""Local HTTP tests: no GitHub requests, accounts, Git LFS, or model weights needed."""
from contextlib import contextmanager
import copy
import hashlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import ssl
import sys
import threading
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location("downloader", Path(__file__).resolve().parents[1] / "download.py")
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def manifest_for(chunks):
    data = b"".join(chunks)
    manifest = {
        "schema_version": 1,
        "repository": "Davidlequnchen/emotion_recognition_models",
        "release_tag": "models-2026-11-09-v1",
        "models": {"test_model": {"name": "Test model", "source": "local fixture", "files": [{
            "path": "nested/model.safetensors", "size": len(data), "sha256": digest(data),
            "parts": [{"asset": "test_model--weight.part{:03d}".format(i), "size": len(chunk),
                       "sha256": digest(chunk)} for i, chunk in enumerate(chunks)]}]}}
    }
    return manifest


class ServerState:
    def __init__(self, assets):
        self.assets = assets
        self.requests = []
        self.ignore_range = False
        self.corrupt = False
        self.bad_range = False
        self.first_416 = False
        self.truncate_first = False
        self.redirect = False


@contextmanager
def local_server(state):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            state.requests.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
            if state.redirect and self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/public-asset")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = state.assets.get(self.path.lstrip("/"))
            if payload is None:
                self.send_error(404)
                return
            if state.first_416 and len(state.requests) == 1:
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            offset = 0
            raw_range = self.headers.get("Range")
            if raw_range and not state.ignore_range:
                offset = int(raw_range.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                start = offset + 1 if state.bad_range else offset
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, len(payload)-1, len(payload)))
            else:
                self.send_response(200)
            output = payload[offset:]
            if state.corrupt:
                output = bytes(byte ^ 1 for byte in output)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            if state.truncate_first and len(state.requests) == 1:
                self.wfile.write(output[:19])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(output)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:{}".format(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chunks = [b"first-" * 3000, b"second-" * 4000]
        self.manifest = manifest_for(self.chunks)
        self.file = self.manifest["models"]["test_model"]["files"][0]
        self.target = self.root / "test_model" / self.file["path"]
        self.assets = {part["asset"]: data for part, data in zip(self.file["parts"], self.chunks)}
        self.quiet = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        self.no_wait = mock.patch.object(downloader.time, "sleep")
        self.no_wait.start()
        self.addCleanup(self.no_wait.stop)

    @contextmanager
    def session(self, state=None, **options):
        state = state or ServerState(self.assets)
        with local_server(state) as base:
            with mock.patch.object(downloader, "asset_url", side_effect=lambda repo, tag, asset: base + "/" + asset):
                yield downloader.Downloader(self.manifest, self.root, **options), state

    def test_fresh_download_and_cleanup(self):
        with self.session() as (client, state):
            self.assertTrue(client.run(["test_model"]))
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))
            self.assertEqual(len(state.requests), 2)
            self.assertEqual(list(client.cache.glob("*")), [])

    def test_resume_206_preserves_downloaded_prefix(self):
        with self.session(workers=1) as (client, state):
            client.cache.mkdir(parents=True)
            part = self.file["parts"][0]
            (client.cache / (part["asset"] + ".partial")).write_bytes(self.chunks[0][:41])
            self.assertTrue(client.run(["test_model"]))
            requests = [request for request in state.requests if request[0].endswith(part["asset"])]
            self.assertEqual(requests[0][1], "bytes=41-")
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))

    def test_ignored_range_restarts_instead_of_appending(self):
        state = ServerState(self.assets)
        state.ignore_range = True
        with self.session(state) as (client, _):
            client.cache.mkdir(parents=True)
            part = self.file["parts"][0]
            (client.cache / (part["asset"] + ".partial")).write_bytes(b"obsolete prefix")
            client.run(["test_model"])
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))

    def test_checksum_failure_keeps_existing_weight(self):
        state = ServerState(self.assets)
        state.corrupt = True
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"original weight must survive")
        with self.session(state, workers=1) as (client, _):
            with self.assertRaisesRegex(downloader.DownloadError, "SHA-256 mismatch"):
                client.run(["test_model"])
            self.assertEqual(self.target.read_bytes(), b"original weight must survive")
            self.assertFalse(Path(str(self.target) + ".downloading").exists())
            per_asset = [row for row in state.requests if row[0].endswith(self.file["parts"][0]["asset"])]
            self.assertEqual(len(per_asset), 3)

    def test_assembly_checksum_failure_is_atomic(self):
        self.file["sha256"] = digest(b"incorrect full checksum")
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"existing weight")
        with self.session() as (client, _):
            with self.assertRaisesRegex(downloader.DownloadError, "Reassembled weight"):
                client.run(["test_model"])
            self.assertEqual(self.target.read_bytes(), b"existing weight")
            self.assertFalse(Path(str(self.target) + ".downloading").exists())
            self.assertEqual(len(list(client.cache.glob("*.part*"))), 2)

    def test_already_verified_target_uses_no_network(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"".join(self.chunks))
        with self.session() as (client, state):
            client.run(["test_model"])
            self.assertEqual(state.requests, [])
            self.assertFalse(client.cache.exists())

    def test_cached_verified_parts_use_no_network(self):
        with self.session(keep_cache=True) as (client, state):
            client.cache.mkdir(parents=True)
            for asset, data in self.assets.items():
                (client.cache / asset).write_bytes(data)
            client.run(["test_model"])
            self.assertEqual(state.requests, [])
            self.assertEqual(len(list(client.cache.iterdir())), 2)
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))

    def test_invalid_content_range_is_rejected(self):
        state = ServerState(self.assets)
        state.bad_range = True
        with self.session(state, workers=1) as (client, _):
            client.cache.mkdir(parents=True)
            part = self.file["parts"][0]
            (client.cache / (part["asset"] + ".partial")).write_bytes(self.chunks[0][:41])
            with self.assertRaisesRegex(downloader.DownloadError, "Invalid Content-Range"):
                client.run(["test_model"])
            self.assertFalse(self.target.exists())

    def test_416_clears_bad_offset_then_retries(self):
        state = ServerState(self.assets)
        state.first_416 = True
        with self.session(state, workers=1) as (client, _):
            client.cache.mkdir(parents=True)
            part = self.file["parts"][0]
            (client.cache / (part["asset"] + ".partial")).write_bytes(self.chunks[0][:41])
            client.run(["test_model"])
            self.assertEqual(state.requests[0][1], "bytes=41-")
            self.assertEqual(state.requests[1][1], None)
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))

    def test_interrupted_transfer_resumes(self):
        state = ServerState(self.assets)
        state.truncate_first = True
        with self.session(state, workers=1) as (client, _):
            client.run(["test_model"])
            self.assertEqual(state.requests[0][1], None)
            self.assertEqual(state.requests[1][1], "bytes=19-")
            self.assertEqual(self.target.read_bytes(), b"".join(self.chunks))

    def test_verify_only_does_not_download_or_create_cache(self):
        with self.session() as (client, state):
            self.assertFalse(client.run(["test_model"], verify_only=True))
            self.assertEqual(state.requests, [])
            self.assertFalse(client.cache.exists())
            self.target.parent.mkdir(parents=True)
            self.target.write_bytes(b"".join(self.chunks))
            self.assertTrue(client.run(["test_model"], verify_only=True))

    def test_manifest_rejects_path_traversal(self):
        for path in ("../escape", "/absolute", "a/../../escape", "a\\..\\escape", ".", "a//b"):
            with self.subTest(path=path):
                bad = copy.deepcopy(self.manifest)
                bad["models"]["test_model"]["files"][0]["path"] = path
                with self.assertRaises(ValueError):
                    downloader.Downloader(bad, self.root)
        bad = copy.deepcopy(self.manifest)
        bad["models"]["test_model"]["files"][0]["parts"][0]["asset"] = "../escape"
        with self.assertRaises(ValueError):
            downloader.Downloader(bad, self.root)

    def test_symlink_cannot_escape_target_root(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "test_model").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "leaves its download directory"):
                downloader.Downloader(self.manifest, self.root)

    def test_symlink_cannot_escape_cache_root(self):
        client = downloader.Downloader(self.manifest, self.root)
        client.cache.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as outside:
            (client.cache / self.file["parts"][0]["asset"]).symlink_to(Path(outside) / "weight")
            with self.assertRaisesRegex(ValueError, "leaves its download directory"):
                downloader.Downloader(self.manifest, self.root)

    def test_redirect_drops_authorization(self):
        state = ServerState({"public-asset": b"test"})
        state.redirect = True
        with local_server(state) as base:
            opener = urllib.request.build_opener(downloader.StripAuthorizationRedirect())
            request = urllib.request.Request(base + "/redirect", headers={"Authorization": "Bearer secret"})
            with opener.open(request) as response:
                self.assertEqual(response.read(), b"test")
        self.assertEqual(state.requests[0][2], "Bearer secret")
        self.assertIsNone(state.requests[1][2])

    def test_bad_token_does_not_echo_its_value(self):
        token = "secret\nunsafe"
        with self.assertRaises(ValueError) as caught:
            downloader.Downloader(self.manifest, self.root, token=token)
        self.assertNotIn("secret", str(caught.exception))

    def test_certificate_failure_fails_fast_preserving_verified_weight(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'old local weight remains untouched')
        client = downloader.Downloader(self.manifest, self.root)
        error = urllib.error.URLError(ssl.SSLCertVerificationError(1, 'self-signed certificate in certificate chain'))
        with mock.patch.object(client.opener, 'open', side_effect=error) as request:
            with mock.patch.object(downloader.time, 'sleep') as sleep:
                with self.assertRaisesRegex(downloader.DownloadError, 'HTTPS certificate verification failed'):
                    client.download_part(self.file['parts'][0])
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'old local weight remains untouched')

    def test_cli_list_and_verify_exit_status(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        with mock.patch.dict("os.environ", {"GITHUB_TOKEN": ""}):
            self.assertEqual(downloader.main(["--manifest", str(path), "--root", str(self.root), "--list"]), 0)
            self.assertEqual(downloader.main(["--manifest", str(path), "--root", str(self.root), "--verify-only"]), 1)
            self.target.parent.mkdir(parents=True)
            self.target.write_bytes(b"".join(self.chunks))
            self.assertEqual(downloader.main(["--manifest", str(path), "--root", str(self.root), "--verify-only"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
