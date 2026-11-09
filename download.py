#!/usr/bin/env python3
"""Download this repository's versioned model weights from GitHub Releases.

Uses only Python 3.9+ standard-library modules; Git LFS is not used. Run
``python download.py --list``, ``python download.py MODEL_KEY``, or
``python download.py`` for all models. Interrupted downloads resume on rerun.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from tls import create_ssl_context, certificate_error, certificate_error_message


class DownloadError(Exception):
    """A model asset could not be safely restored."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid relative path in manifest")
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or path.as_posix() != value
            or any(part in ("", ".", "..") for part in path.parts)
            or any(ord(char) < 32 for char in value)):
        raise ValueError("Unsafe relative path in manifest: " + repr(value))
    return path


def safe_path(root, value):
    """Reject lexical traversal and existing symlinks that leave the root."""
    path = root.joinpath(*relative_path(value).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path leaves its download directory: " + repr(value))
    return path


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version; expected 1")
    repository = manifest.get("repository", "")
    if (not isinstance(repository, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or any(part in (".", "..") for part in repository.split("/"))):
        raise ValueError("Invalid GitHub repository in manifest")
    tag = manifest.get("release_tag")
    if not isinstance(tag, str) or not tag or any(ord(char) < 32 for char in tag):
        raise ValueError("Invalid release_tag in manifest")
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Manifest must contain models")
    assets = set()
    for key, model in models.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Invalid model key")
        files = model.get("files") if isinstance(model, dict) else None
        if not isinstance(files, list) or not files:
            raise ValueError("Model has no weight files: " + key)
        names = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("Invalid file entry: " + key)
            path = relative_path(item.get("path")).as_posix()
            if path in names:
                raise ValueError("Repeated model file: " + key + "/" + path)
            names.add(path)
            validate_digest_and_size(item)
            parts = item.get("parts")
            if not isinstance(parts, list) or not parts:
                raise ValueError("Weight file has no release assets: " + path)
            total = 0
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("Invalid release asset")
                validate_digest_and_size(part)
                asset = part.get("asset", "")
                if (not isinstance(asset, str)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", asset)):
                    raise ValueError("Unsafe release asset name")
                if asset in assets:
                    raise ValueError("Repeated release asset: " + asset)
                assets.add(asset)
                total += part["size"]
            if total != item["size"]:
                raise ValueError("Release asset sizes do not sum to weight size: " + path)
    return manifest


def validate_digest_and_size(item):
    if (not isinstance(item.get("size"), int) or isinstance(item["size"], bool)
            or item["size"] <= 0):
        raise ValueError("Every file and asset must have a positive integer size")
    if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise ValueError("Every file and asset must have a lowercase SHA-256")


def asset_url(repository, tag, asset):
    return "https://github.com/{}/releases/download/{}/{}".format(
        repository, urllib.parse.quote(tag, safe=""), urllib.parse.quote(asset, safe=""))


class StripAuthorizationRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if redirected is not None:
            # Never forward GitHub credentials to the release CDN or another host.
            redirected.remove_header("Authorization")
        return redirected


class Downloader:
    def __init__(self, manifest, root, cache_dir=None, workers=2, keep_cache=False,
                 token=None, timeout=60, ca_bundle=None):
        self.manifest = validate_manifest(manifest)
        self.root = Path(root).resolve()
        cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else self.root / ".downloads"
        namespace = hashlib.sha256((manifest["repository"] + "\0" + manifest["release_tag"]).encode()).hexdigest()[:16]
        self.cache = safe_path(cache_root, namespace)
        self.workers = max(1, workers)
        self.keep_cache = keep_cache
        if token and any(ord(char) <= 32 or ord(char) >= 127 for char in token):
            raise ValueError("GitHub token must contain printable ASCII without whitespace")
        self.token = token
        self.timeout = timeout
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=create_ssl_context(ca_bundle)),
            StripAuthorizationRedirect())
        self._api_assets = None
        self._api_lock = threading.Lock()
        # Validate every target before making any network request or writing files.
        for key, model in manifest["models"].items():
            for item in model["files"]:
                safe_path(self.root, key + "/" + item["path"])
                safe_path(self.root, key + "/" + item["path"] + ".downloading")
                for part in item["parts"]:
                    safe_path(self.cache, part["asset"])
                    safe_path(self.cache, part["asset"] + ".partial")

    @staticmethod
    def matches(path, item):
        return path.is_file() and path.stat().st_size == item["size"] and sha256(path) == item["sha256"]

    def _request(self, asset, offset):
        headers = {"User-Agent": "emotion-recognition-models-downloader/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
        url = asset_url(self.manifest["repository"], self.manifest["release_tag"], asset)
        if self.token:
            url = self._authenticated_asset(asset)
            headers["Authorization"] = "Bearer " + self.token
            headers["Accept"] = "application/octet-stream"
        return urllib.request.Request(url, headers=headers)

    def _authenticated_asset(self, asset):
        with self._api_lock:
            if self._api_assets is None:
                url = "https://api.github.com/repos/{}/releases/tags/{}".format(
                    self.manifest["repository"], urllib.parse.quote(self.manifest["release_tag"], safe=""))
                request = urllib.request.Request(url, headers={
                    "User-Agent": "emotion-recognition-models-downloader/1",
                    "Accept": "application/vnd.github+json", "Authorization": "Bearer " + self.token})
                with self.opener.open(request, timeout=self.timeout) as response:
                    release = json.load(response)
                self._api_assets = {item["name"]: item["url"] for item in release.get("assets", [])}
            if asset not in self._api_assets:
                raise DownloadError("Release asset not found: " + asset)
            url = self._api_assets[asset]
            expected = "https://api.github.com/repos/{}/releases/assets/".format(self.manifest["repository"])
            if not isinstance(url, str) or not url.startswith(expected) or not url[len(expected):].isdigit():
                raise DownloadError("Unsafe GitHub API asset URL")
            return url

    def download_part(self, part):
        target = safe_path(self.cache, part["asset"])
        partial = safe_path(self.cache, part["asset"] + ".partial")
        if self.matches(target, part):
            return target
        self.cache.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        last_error = None
        for attempt in range(3):
            if attempt:
                time.sleep(min(2 ** (attempt - 1), 4))
            try:
                size = partial.stat().st_size if partial.exists() else 0
                if size > part["size"]:
                    partial.unlink()
                    size = 0
                if size == part["size"]:
                    if self.matches(partial, part):
                        partial.replace(target)
                        return target
                    partial.unlink()
                    size = 0
                print("  fetching {} (resume: {:,} bytes)".format(part["asset"], size), flush=True)
                request = self._request(part["asset"], size)
                with self.opener.open(request, timeout=self.timeout) as response:
                    status = response.getcode()
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if (not match or int(match[1]) != size or int(match[3]) != part["size"]
                                or not size <= int(match[2]) < part["size"]):
                            raise DownloadError("Invalid Content-Range for " + part["asset"])
                    elif status == 200:
                        # A server may ignore Range: discard the old prefix and start over.
                        size = 0
                    else:
                        raise DownloadError("Unexpected HTTP status {} for {}".format(status, part["asset"]))
                    with partial.open("ab" if status == 206 and size else "wb") as stream:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            size += len(chunk)
                            if size > part["size"]:
                                raise DownloadError("Release asset exceeds its recorded size: " + part["asset"])
                            stream.write(chunk)
                if partial.stat().st_size != part["size"]:
                    raise DownloadError("Incomplete release asset: " + part["asset"])
                if sha256(partial) != part["sha256"]:
                    partial.unlink()
                    raise DownloadError("SHA-256 mismatch: " + part["asset"])
                partial.replace(target)
                print("  downloaded and verified " + part["asset"], flush=True)
                return target
            except urllib.error.HTTPError as exc:
                last_error = "HTTP {} for {}".format(exc.code, part["asset"])
                if exc.code == 416:
                    if self.matches(partial, part):
                        partial.replace(target)
                        return target
                    if partial.exists():
                        partial.unlink()
                elif exc.code not in (408, 429) and exc.code < 500:
                    break
            except (OSError, urllib.error.URLError, http.client.HTTPException, DownloadError) as exc:
                if certificate_error(exc):
                    raise DownloadError(certificate_error_message(exc)) from exc
                last_error = str(exc)
                if partial.exists() and partial.stat().st_size > part["size"]:
                    partial.unlink()
        raise DownloadError("Unable to download {} after at most 3 attempts: {}".format(part["asset"], last_error))

    def restore_file(self, key, item):
        name = key + "/" + item["path"]
        target = safe_path(self.root, name)
        if self.matches(target, item):
            print("  already verified: " + name, flush=True)
            return
        print("  downloading {} ({:.3f} GB)".format(name, item["size"] / 1e9), flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # map preserves manifest part order even when downloads finish out of order.
            parts = list(pool.map(self.download_part, item["parts"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe_path(self.root, name + ".downloading")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                for part in parts:
                    with part.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise DownloadError("Reassembled weight failed SHA-256 verification: " + name)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        if not self.keep_cache:
            for part in parts:
                part.unlink()
        print("  restored and verified: " + name, flush=True)

    def run(self, keys, verify_only=False):
        valid = True
        for key in keys:
            print(self.manifest["models"][key].get("name", key), flush=True)
            for item in self.manifest["models"][key]["files"]:
                if verify_only:
                    name = key + "/" + item["path"]
                    matches = self.matches(safe_path(self.root, name), item)
                    print("  {}: {}".format("OK" if matches else "MISSING OR INVALID", name), flush=True)
                    valid = matches and valid
                else:
                    self.restore_file(key, item)
        return valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", metavar="MODEL", help="model keys; default: all")
    parser.add_argument("--list", action="store_true", help="list model keys and weight sizes without downloading")
    parser.add_argument("--verify-only", action="store_true", help="verify local weights without network access")
    parser.add_argument("--workers", type=int, default=2, help="parallel asset downloads per weight file (default: 2)")
    parser.add_argument("--cache-dir", type=Path, help="release part cache; default: .downloads beside this script")
    parser.add_argument("--ca-bundle", type=Path, help="company/custom CA PEM bundle; otherwise SSL_CERT_FILE or system/default roots")
    parser.add_argument("--keep-cache", action="store_true", help="keep verified release parts after restoring weights")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="optional GitHub API token environment variable")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="model root; default: this script's directory")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        manifest = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        keys = list(manifest["models"]) if not args.models or args.models == ["all"] else args.models
        unknown = [key for key in keys if key not in manifest["models"]]
        if unknown:
            parser.error("unknown model keys: " + ", ".join(unknown))
        if len(set(keys)) != len(keys):
            parser.error("repeat model keys are not allowed")
        if args.list:
            for key in keys:
                model = manifest["models"][key]
                size = sum(item["size"] for item in model["files"])
                print("{}\t{:.3f} GB\t{}".format(key, size / 1e9, model.get("name", key)))
            return 0
        downloader = Downloader(manifest, args.root, args.cache_dir, args.workers,
                                args.keep_cache, os.environ.get(args.token_env), ca_bundle=args.ca_bundle)
        return 0 if downloader.run(keys, args.verify_only) else 1
    except (OSError, ValueError, DownloadError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Run the same command again to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
