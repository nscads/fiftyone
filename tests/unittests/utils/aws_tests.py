"""
Unit tests for the S3 media cache utilities in :mod:`fiftyone.utils.aws`.

S3 is mocked via ``moto``, so these tests require no network access or AWS
credentials.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import io
import os
import re
import sys
import time
import uuid
from unittest import mock

import boto3
import botocore
import pytest
from PIL import Image

import fiftyone as fo
import fiftyone.core.cli as focl
import fiftyone.core.config as focc
from fiftyone import ViewField as F

moto = pytest.importorskip("moto")

import fiftyone.utils.aws as fouaws  # pylint: disable=wrong-import-position

BUCKET = "test-bucket"

MB = 1024**2

# key: (width, height)
IMAGES = {
    "dir/a.jpg": (32, 24),
    "dir/b.png": (16, 8),
    "dir/sub/c.jpg": (20, 10),
}

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _uri(key, bucket=BUCKET):
    return "s3://%s/%s" % (bucket, key)


def _image_bytes(width, height, fmt="JPEG", color=(200, 30, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def _put(client, key, body):
    client.put_object(Bucket=BUCKET, Key=key, Body=body)
    return _uri(key)


def _put_multipart(client, key, parts):
    upload_id = client.create_multipart_upload(Bucket=BUCKET, Key=key)[
        "UploadId"
    ]

    completed = []
    for num, data in enumerate(parts, 1):
        resp = client.upload_part(
            Bucket=BUCKET,
            Key=key,
            PartNumber=num,
            UploadId=upload_id,
            Body=data,
        )
        completed.append({"ETag": resp["ETag"], "PartNumber": num})

    client.complete_multipart_upload(
        Bucket=BUCKET,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": completed},
    )

    return _uri(key)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _flip_byte(path, offset=0):
    with open(path, "r+b") as f:
        f.seek(offset)
        value = f.read(1)[0]
        f.seek(offset)
        f.write(bytes([value ^ 0xFF]))


def _cache_path(s3_uri, cache_dir):
    return fouaws.get_s3_cache_path(s3_uri, cache_dir=cache_dir)


def _summary(**kwargs):
    summary = {
        "num_samples": 0,
        "num_backfilled": 0,
        "num_unresolved": 0,
        "num_downloaded": 0,
        "num_failed": 0,
        "num_updated": 0,
    }
    summary.update(kwargs)
    return summary


def _patch_download_file(monkeypatch, client, errors=None, always=None):
    """Patches ``client.download_file`` to record its calls and to raise the
    given errors, in order, before delegating to the real method.
    """
    download_file = client.download_file
    errors = list(errors or [])
    calls = []

    def _download_file(bucket, key, path, *args, **kwargs):
        calls.append(key)
        if always is not None:
            raise always

        if errors:
            raise errors.pop(0)

        return download_file(bucket, key, path, *args, **kwargs)

    monkeypatch.setattr(client, "download_file", _download_file)

    return calls


def _client_error(status, code="Error", operation="GetObject"):
    return botocore.exceptions.ClientError(
        {
            "Error": {"Code": code, "Message": "mock error"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


class _FakeTime(object):
    """Stands in for the ``time`` module of :mod:`fiftyone.utils.aws` and
    records sleeps instead of sleeping.
    """

    def __init__(self):
        self.sleeps = []

    def sleep(self, secs):
        self.sleeps.append(secs)

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture(name="aws_env")
def fixture_aws_env(monkeypatch, tmp_path):
    """Provides fake AWS credentials and prevents any real AWS settings from
    leaking into the tests.
    """
    for var in list(os.environ):
        if var.startswith("AWS_"):
            monkeypatch.delenv(var)

    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws" / "config"))
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws" / "credentials")
    )


@pytest.fixture(name="s3")
def fixture_s3(aws_env):
    """Yields an S3 client for a mocked S3 containing an empty bucket."""
    with moto.mock_aws():
        client = fouaws.get_s3_client()
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture(name="images")
def fixture_images(s3):
    """Uploads :data:`IMAGES` and returns a dict mapping their URIs to their
    ``(width, height)``.
    """
    uris = {}
    for key, (width, height) in IMAGES.items():
        fmt = "PNG" if key.endswith(".png") else "JPEG"
        uris[_put(s3, key, _image_bytes(width, height, fmt=fmt))] = (
            width,
            height,
        )

    return uris


@pytest.fixture(name="api_calls")
def fixture_api_calls(s3):
    """Returns a list that records the names of all API operations made by
    the ``s3`` client.
    """
    calls = []

    def _record(model, **kwargs):
        calls.append(model.name)

    s3.meta.events.register("before-call.s3", _record)
    yield calls
    s3.meta.events.unregister("before-call.s3", _record)


@pytest.fixture(name="fake_time")
def fixture_fake_time(monkeypatch):
    fake_time = _FakeTime()
    monkeypatch.setattr(fouaws, "time", fake_time)
    return fake_time


@pytest.fixture(name="cache_dir")
def fixture_cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture(name="dataset_name")
def fixture_dataset_name():
    name = "test-s3-cli-" + uuid.uuid4().hex
    yield name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


class TestS3Uris:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("s3://bucket/key.jpg", True),
            ("s3://bucket", True),
            ("S3://bucket/key.jpg", False),
            ("gs://bucket/key.jpg", False),
            ("/cwd/s3:/bucket/key.jpg", False),
            ("bucket/key.jpg", False),
            ("", False),
            (None, False),
            (123, False),
        ],
    )
    def test_is_s3_uri(self, path, expected):
        assert fouaws.is_s3_uri(path) is expected

    @pytest.mark.parametrize(
        "key",
        [
            "image.jpg",
            "path/to/image.jpg",
            "query?x=1&y=2.jpg",
            "frag#ment.jpg",
            "both?a=b#c.jpg",
            "with space/and more spaces.jpg",
            "unicodé/猫の写真.jpg",
            "percent%20encoded.jpg",
            "colon:in:key.jpg",
            "double//slash.jpg",
            "trailing/",
        ],
    )
    def test_parse_s3_uri(self, key):
        assert fouaws.parse_s3_uri("s3://my-bucket/" + key) == (
            "my-bucket",
            key,
        )

    @pytest.mark.parametrize(
        "s3_uri",
        [
            "s3://",
            "s3:///key.jpg",
            "s3://bucket",
            "s3://bucket/",
            "S3://bucket/key.jpg",
            "gs://bucket/key.jpg",
            "http://bucket/key.jpg",
            "bucket/key.jpg",
            "/cwd/s3:/bucket/key.jpg",
            "",
            None,
        ],
    )
    def test_parse_s3_uri_invalid(self, s3_uri):
        with pytest.raises(ValueError):
            fouaws.parse_s3_uri(s3_uri)


class TestGetS3CachePath:
    def test_mirrors_bucket_and_key(self, tmp_path):
        cache_dir = str(tmp_path)

        path = fouaws.get_s3_cache_path(
            "s3://bucket/path/to/image.jpg", cache_dir=cache_dir
        )

        assert path == os.path.join(
            cache_dir, "bucket", "path", "to", "image.jpg"
        )

    @pytest.mark.parametrize(
        "key",
        [
            "query?x=1.jpg",
            "frag#ment.jpg",
            "with space/a b.jpg",
            "unicodé/猫.jpg",
            ".hidden/..double-dot-prefix.jpg",
            ".../three-dots.jpg",
        ],
    )
    def test_preserves_key_characters(self, tmp_path, key):
        cache_dir = str(tmp_path)

        path = fouaws.get_s3_cache_path(
            "s3://bucket/" + key, cache_dir=cache_dir
        )

        assert path == os.path.join(cache_dir, "bucket", *key.split("/"))
        assert os.path.commonpath([cache_dir, path]) == cache_dir

    @pytest.mark.parametrize(
        "s3_uri",
        [
            "s3://bucket/a/../b.jpg",
            "s3://bucket/../b.jpg",
            "s3://bucket/a/..",
            "s3://bucket/..",
            "s3://bucket/./b.jpg",
            "s3://bucket/a/./b.jpg",
            "s3://bucket/a/.",
            "s3://bucket/a//b.jpg",
            "s3://bucket//b.jpg",
            "s3://bucket/a/",
            "s3://../b.jpg",
            "s3://./b.jpg",
        ],
    )
    def test_rejects_unmappable_uris(self, tmp_path, s3_uri):
        with pytest.raises(ValueError, match="cannot be mapped"):
            fouaws.get_s3_cache_path(s3_uri, cache_dir=str(tmp_path))

    @pytest.mark.parametrize(
        "s3_uri", ["s3://bucket", "s3://bucket/", "bucket/key.jpg", None]
    )
    def test_rejects_invalid_uris(self, tmp_path, s3_uri):
        with pytest.raises(ValueError):
            fouaws.get_s3_cache_path(s3_uri, cache_dir=str(tmp_path))

    def test_relative_cache_dir_is_made_absolute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        path = fouaws.get_s3_cache_path("s3://bucket/a.jpg", cache_dir="rel")

        assert path == os.path.join(str(tmp_path), "rel", "bucket", "a.jpg")

    def test_default_cache_dir(self, tmp_path):
        cache_dir = str(tmp_path / "default-cache")

        with mock.patch.object(fo.config, "s3_cache_dir", cache_dir):
            path = fouaws.get_s3_cache_path("s3://bucket/dir/a.jpg")

        assert path == os.path.join(cache_dir, "bucket", "dir", "a.jpg")


class TestS3CacheDirConfig:
    def test_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIFTYONE_S3_CACHE_DIR", raising=False)
        monkeypatch.setenv("FIFTYONE_DEFAULT_DATASET_DIR", str(tmp_path))

        config = focc.FiftyOneConfig()

        assert config.s3_cache_dir == os.path.join(
            str(tmp_path), "__s3_cache__"
        )

    def test_env_var(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FIFTYONE_S3_CACHE_DIR", "my-cache")

        config = focc.FiftyOneConfig()

        assert config.s3_cache_dir == os.path.join(str(tmp_path), "my-cache")

    def test_config_dict(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIFTYONE_S3_CACHE_DIR", raising=False)
        cache_dir = str(tmp_path / "from-dict")

        config = focc.FiftyOneConfig({"s3_cache_dir": cache_dir})

        assert config.s3_cache_dir == cache_dir


class TestGetS3Client:
    def test_settings(self, aws_env):
        client = fouaws.get_s3_client(
            region="eu-west-1",
            endpoint_url="http://localhost:9000",
            max_pool_connections=32,
        )

        assert client.meta.region_name == "eu-west-1"
        assert client.meta.endpoint_url == "http://localhost:9000"
        assert client.meta.config.max_pool_connections == 32
        assert client.meta.config.retries["mode"] == "standard"

    def test_defaults(self, aws_env):
        client = fouaws.get_s3_client()

        assert client.meta.region_name == "us-east-1"
        assert client.meta.config.max_pool_connections == 10

        client = fouaws.get_s3_client(max_pool_connections=2)
        assert client.meta.config.max_pool_connections == 10

    def test_endpoint_env_var(self, aws_env, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://minio.local:9000")

        client = fouaws.get_s3_client()

        assert client.meta.endpoint_url == "http://minio.local:9000"

    def test_profile(self, aws_env, monkeypatch, tmp_path):
        # The environment takes precedence over the profile
        monkeypatch.delenv("AWS_DEFAULT_REGION")
        _write(
            str(tmp_path / "aws" / "config"),
            b"[profile custom]\nregion = ap-south-1\n",
        )

        client = fouaws.get_s3_client(profile="custom")
        assert client.meta.region_name == "ap-south-1"

        with pytest.raises(botocore.exceptions.ProfileNotFound):
            fouaws.get_s3_client(profile="missing")


class TestListS3Uris:
    @pytest.fixture(name="objects")
    def fixture_objects(self, s3):
        keys = [
            "dir/a.jpg",
            "dir/B.PNG",
            "dir/notes.txt",
            "dir/clip.mp4",
            "dir/sub/",
            "dir/sub/c.jpeg",
            "dir/sub/d.TXT",
            "dir/folder.jpg/",
            "dir/with space ?#ü.jpg",
            "dir-other/e.jpg",
            "other/f.png",
        ]
        for key in keys:
            body = b"" if key.endswith("/") else b"data"
            s3.put_object(Bucket=BUCKET, Key=key, Body=body)

        return keys

    def test_default_lists_images_under_prefix(self, s3, objects):
        uris = fouaws.list_s3_uris(_uri("dir/"), client=s3)

        assert uris == [
            _uri("dir/B.PNG"),
            _uri("dir/a.jpg"),
            _uri("dir/sub/c.jpeg"),
            _uri("dir/with space ?#ü.jpg"),
        ]

    def test_prefix_is_a_key_prefix(self, s3, objects):
        uris = fouaws.list_s3_uris(_uri("dir"), client=s3)
        assert _uri("dir-other/e.jpg") in uris
        assert _uri("other/f.png") not in uris

        uris = fouaws.list_s3_uris(_uri("dir/sub"), client=s3)
        assert uris == [_uri("dir/sub/c.jpeg")]

        uris = fouaws.list_s3_uris(_uri("dir/a"), client=s3)
        assert uris == [_uri("dir/a.jpg")]

    @pytest.mark.parametrize("prefix", ["s3://%s" % BUCKET, _uri("")])
    def test_whole_bucket(self, s3, objects, prefix):
        uris = fouaws.list_s3_uris(prefix, client=s3)

        assert uris == [
            _uri("dir-other/e.jpg"),
            _uri("dir/B.PNG"),
            _uri("dir/a.jpg"),
            _uri("dir/sub/c.jpeg"),
            _uri("dir/with space ?#ü.jpg"),
            _uri("other/f.png"),
        ]

    @pytest.mark.parametrize("extensions", [".txt", "txt", ".TXT", ["TxT"]])
    def test_extensions(self, s3, objects, extensions):
        uris = fouaws.list_s3_uris(
            _uri("dir/"), extensions=extensions, client=s3
        )

        assert uris == [_uri("dir/notes.txt"), _uri("dir/sub/d.TXT")]

    def test_multiple_extensions(self, s3, objects):
        uris = fouaws.list_s3_uris(
            _uri("dir/"), extensions=("mp4", ".png"), client=s3
        )

        assert uris == [_uri("dir/B.PNG"), _uri("dir/clip.mp4")]

    def test_directory_markers_are_skipped(self, s3, objects):
        everything = (".jpg", ".jpeg", ".png", ".mp4", ".txt", "/", "")
        for extensions in (None, everything):
            uris = fouaws.list_s3_uris(
                _uri("dir/"), extensions=extensions, client=s3
            )
            assert uris
            assert not any(uri.endswith("/") for uri in uris)

    def test_no_matches(self, s3, objects):
        assert fouaws.list_s3_uris(_uri("missing/"), client=s3) == []

    def test_multiple_pages(self, s3, objects, monkeypatch):
        get_paginator = s3.get_paginator
        page_sizes = []

        def _get_paginator(name):
            paginator = get_paginator(name)
            paginate = paginator.paginate

            def _paginate(**kwargs):
                kwargs["PaginationConfig"] = {"PageSize": 2}
                for page in paginate(**kwargs):
                    page_sizes.append(page.get("KeyCount"))
                    yield page

            paginator.paginate = _paginate
            return paginator

        monkeypatch.setattr(s3, "get_paginator", _get_paginator)

        uris = fouaws.list_s3_uris(_uri("dir/"), client=s3)

        assert len(page_sizes) > 1
        assert uris == [
            _uri("dir/B.PNG"),
            _uri("dir/a.jpg"),
            _uri("dir/sub/c.jpeg"),
            _uri("dir/with space ?#ü.jpg"),
        ]

    def test_default_client(self, s3, objects):
        uris = fouaws.list_s3_uris(_uri("dir/sub/"))
        assert uris == [_uri("dir/sub/c.jpeg")]

    @pytest.mark.parametrize("prefix", ["dir/", "s3://", "gs://bucket/"])
    def test_invalid_prefix(self, s3, prefix):
        with pytest.raises(ValueError):
            fouaws.list_s3_uris(prefix, client=s3)


class TestCacheS3Files:
    def test_downloads_to_mirrored_paths(self, s3, images, cache_dir):
        uris = list(images)

        paths = fouaws.cache_s3_files(
            uris, cache_dir=cache_dir, num_workers=4, client=s3
        )

        assert paths == [_cache_path(uri, cache_dir) for uri in uris]
        for uri, path in zip(uris, paths):
            bucket, key = fouaws.parse_s3_uri(uri)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            assert _read(path) == body

        # No temporary files are left behind
        num_files = sum(len(f) for _, _, f in os.walk(cache_dir))
        assert num_files == len(uris)

    def test_special_characters(self, s3, cache_dir):
        key = "dir/with space ?#ü.jpg"
        uri = _put(s3, key, b"special")

        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)

        assert path == os.path.join(cache_dir, BUCKET, "dir", key[4:])
        assert _read(path) == b"special"

    def test_single_worker(self, s3, images, cache_dir):
        uris = list(images)

        paths = fouaws.cache_s3_files(
            uris, cache_dir=cache_dir, num_workers=1, client=s3
        )

        assert paths == [_cache_path(uri, cache_dir) for uri in uris]
        assert all(os.path.isfile(p) for p in paths)

    def test_empty(self, s3, cache_dir, api_calls):
        assert fouaws.cache_s3_files([], cache_dir=cache_dir, client=s3) == []
        assert api_calls == []

    def test_accepts_generators(self, s3, images, cache_dir):
        uris = list(images)

        paths = fouaws.cache_s3_files(
            (uri for uri in uris), cache_dir=cache_dir, client=s3
        )

        assert paths == [_cache_path(uri, cache_dir) for uri in uris]

    def test_cached_files_are_trusted_by_default(
        self, s3, images, cache_dir, api_calls
    ):
        uris = list(images)
        fouaws.cache_s3_files(uris, cache_dir=cache_dir, client=s3)
        path = _cache_path(uris[0], cache_dir)
        _write(path, b"tampered")

        # An object that does not even exist in S3 is trusted if cached
        missing_uri = _uri("dir/not-in-s3.jpg")
        missing_path = _cache_path(missing_uri, cache_dir)
        _write(missing_path, b"local only")

        del api_calls[:]

        paths = fouaws.cache_s3_files(
            uris + [missing_uri], cache_dir=cache_dir, client=s3
        )

        assert api_calls == []
        assert paths[-1] == missing_path
        assert _read(path) == b"tampered"
        assert _read(missing_path) == b"local only"

    def test_validate_size(self, s3, images, cache_dir, api_calls):
        uri = list(images)[0]
        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        expected = _read(path)

        # Same size: validation passes without downloading
        _flip_byte(path)
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="size", client=s3
        )
        assert api_calls == ["HeadObject"]
        assert _read(path) != expected

        # Different size: the file is re-downloaded
        _write(path, expected + b"extra")
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="size", client=s3
        )
        assert _read(path) == expected

    def test_validate_etag(self, s3, images, cache_dir, api_calls):
        uri = list(images)[0]
        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        expected = _read(path)

        # Unchanged file: validation passes without downloading
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert api_calls == ["HeadObject"]

        # Same size but different content: the file is re-downloaded
        _flip_byte(path, offset=len(expected) // 2)
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert "GetObject" in api_calls
        assert _read(path) == expected

        # Different size: the file is re-downloaded
        _write(path, b"x")
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert _read(path) == expected

    def test_validate_etag_multipart(self, s3, cache_dir, api_calls):
        parts = [os.urandom(5 * MB), os.urandom(1000)]
        uri = _put_multipart(s3, "dir/multipart.bin", parts)
        etag = s3.head_object(Bucket=BUCKET, Key="dir/multipart.bin")["ETag"]
        assert etag.strip('"').endswith("-2")

        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        assert _read(path) == b"".join(parts)

        # Unchanged file: validation passes without downloading
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert "GetObject" not in api_calls
        assert api_calls.count("HeadObject") == 2

        # Content of the second part changed: the file is re-downloaded
        _flip_byte(path, offset=5 * MB + 10)
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert "GetObject" in api_calls
        assert _read(path) == b"".join(parts)

        # Content of the first part changed: the file is re-downloaded
        _flip_byte(path, offset=10)
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert _read(path) == b"".join(parts)

        # Size validation cannot detect same-size changes
        _flip_byte(path, offset=10)
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="size", client=s3
        )
        assert _read(path) != b"".join(parts)

    def test_validate_etag_multipart_non_uniform_parts(
        self, s3, cache_dir, api_calls
    ):
        # S3 allows parts of different sizes; only the last part may be
        # smaller than 5MB
        parts = [os.urandom(6 * MB), os.urandom(5 * MB), os.urandom(1000)]
        uri = _put_multipart(s3, "dir/non-uniform.bin", parts)

        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        assert _read(path) == b"".join(parts)

        # An unchanged file must not be re-downloaded
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert "GetObject" not in api_calls

    def test_validate_etag_kms_falls_back_to_size(
        self, s3, cache_dir, api_calls
    ):
        s3.put_object(
            Bucket=BUCKET,
            Key="dir/kms.jpg",
            Body=_image_bytes(8, 8),
            ServerSideEncryption="aws:kms",
        )
        uri = _uri("dir/kms.jpg")
        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        expected = _read(path)

        # KMS ETags are not MD5 digests, so only the size is validated
        _flip_byte(path, offset=len(expected) // 2)
        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert api_calls == ["HeadObject"]
        assert _read(path) != expected

        _write(path, b"x")
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="etag", client=s3
        )
        assert _read(path) == expected

    def test_invalid_validate(self, s3, images, cache_dir):
        with pytest.raises(ValueError, match="validate"):
            fouaws.cache_s3_files(
                list(images), cache_dir=cache_dir, validate="md5", client=s3
            )

        assert not os.path.exists(cache_dir)

    def test_overwrite(self, s3, images, cache_dir, api_calls):
        uri = list(images)[0]
        (path,) = fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)
        expected = _read(path)
        _write(path, b"tampered")

        del api_calls[:]
        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, overwrite=True, client=s3
        )

        assert "GetObject" in api_calls
        assert "HeadObject" in api_calls
        assert _read(path) == expected

    def test_skip_failures_missing_object(self, s3, images, cache_dir):
        uri = list(images)[0]
        missing_uri = _uri("dir/missing.jpg")

        paths = fouaws.cache_s3_files(
            [uri, missing_uri, uri],
            cache_dir=cache_dir,
            skip_failures=True,
            client=s3,
        )

        path = _cache_path(uri, cache_dir)
        assert paths == [path, None, path]
        assert os.path.isfile(path)
        assert not os.path.exists(_cache_path(missing_uri, cache_dir))

    def test_failure_raises_with_uri(self, s3, images, cache_dir):
        missing_uri = _uri("dir/missing.jpg")

        with pytest.raises(RuntimeError) as exc_info:
            fouaws.cache_s3_files(
                list(images) + [missing_uri],
                cache_dir=cache_dir,
                num_workers=1,
                client=s3,
            )

        assert missing_uri in str(exc_info.value)
        assert isinstance(
            exc_info.value.__cause__, botocore.exceptions.ClientError
        )
        assert not os.path.exists(_cache_path(missing_uri, cache_dir))

    def test_skip_failures_unmappable_uri(self, s3, images, cache_dir):
        uri = list(images)[0]
        bad_uri = _uri("dir/../a.jpg")

        paths = fouaws.cache_s3_files(
            [bad_uri, uri, "not-a-uri"],
            cache_dir=cache_dir,
            skip_failures=True,
            client=s3,
        )

        assert paths == [None, _cache_path(uri, cache_dir), None]

        with pytest.raises(ValueError, match="cannot be mapped"):
            fouaws.cache_s3_files([bad_uri], cache_dir=cache_dir, client=s3)

    def test_duplicate_uris_are_downloaded_once(
        self, monkeypatch, s3, images, cache_dir
    ):
        uris = list(images)
        calls = _patch_download_file(monkeypatch, s3)

        paths = fouaws.cache_s3_files(
            uris + uris[::-1] + [uris[0]],
            cache_dir=cache_dir,
            num_workers=4,
            client=s3,
        )

        expected = [_cache_path(uri, cache_dir) for uri in uris]
        assert paths == expected + expected[::-1] + [expected[0]]
        assert sorted(calls) == sorted(IMAGES)

    @pytest.mark.parametrize(
        "error",
        [
            botocore.exceptions.ConnectionError(error="connection reset"),
            botocore.exceptions.EndpointConnectionError(
                endpoint_url="https://s3.amazonaws.com"
            ),
            botocore.exceptions.ReadTimeoutError(
                endpoint_url="https://s3.amazonaws.com"
            ),
            _client_error(503, code="SlowDown"),
            _client_error(500, code="InternalError"),
            # How boto3 reports download streams that failed repeatedly
            boto3.exceptions.RetriesExceededError(
                botocore.exceptions.IncompleteReadError(
                    actual_bytes=1, expected_bytes=2
                )
            ),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_retries_transient_errors(
        self, monkeypatch, s3, images, cache_dir, fake_time, error
    ):
        uri = list(images)[0]
        calls = _patch_download_file(monkeypatch, s3, errors=[error])

        (path,) = fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, num_workers=1, client=s3
        )

        assert calls == ["dir/a.jpg", "dir/a.jpg"]
        assert fake_time.sleeps == [1]
        assert os.path.isfile(path)

    def test_gives_up_after_max_attempts(
        self, monkeypatch, s3, images, cache_dir, fake_time
    ):
        uri = list(images)[0]
        error = botocore.exceptions.EndpointConnectionError(
            endpoint_url="https://s3.amazonaws.com"
        )
        calls = _patch_download_file(monkeypatch, s3, always=error)

        with pytest.raises(RuntimeError) as exc_info:
            fouaws.cache_s3_files(
                [uri], cache_dir=cache_dir, max_attempts=3, client=s3
            )

        assert uri in str(exc_info.value)
        assert exc_info.value.__cause__ is error
        assert len(calls) == 3
        assert fake_time.sleeps == [1, 2]

        # With skip_failures, the object is reported as failed
        del calls[:]
        paths = fouaws.cache_s3_files(
            [uri],
            cache_dir=cache_dir,
            max_attempts=2,
            skip_failures=True,
            client=s3,
        )
        assert paths == [None]
        assert len(calls) == 2

    @pytest.mark.parametrize("status", [403, 404])
    def test_does_not_retry_client_errors(
        self, monkeypatch, s3, images, cache_dir, fake_time, status
    ):
        uri = list(images)[0]
        error = _client_error(status, operation="HeadObject")
        calls = _patch_download_file(monkeypatch, s3, always=error)

        with pytest.raises(RuntimeError, match="dir/a.jpg"):
            fouaws.cache_s3_files(
                [uri], cache_dir=cache_dir, max_attempts=5, client=s3
            )

        assert len(calls) == 1
        assert fake_time.sleeps == []

    def test_does_not_retry_missing_objects(
        self, monkeypatch, s3, cache_dir, fake_time
    ):
        missing_uri = _uri("dir/missing.jpg")
        calls = _patch_download_file(monkeypatch, s3)

        with pytest.raises(RuntimeError, match="dir/missing.jpg") as exc_info:
            fouaws.cache_s3_files(
                [missing_uri], cache_dir=cache_dir, max_attempts=5, client=s3
            )

        cause = exc_info.value.__cause__
        assert isinstance(cause, botocore.exceptions.ClientError)
        assert cause.response["ResponseMetadata"]["HTTPStatusCode"] == 404
        assert calls == ["dir/missing.jpg"]
        assert fake_time.sleeps == []

    def test_retries_validation_requests(
        self, monkeypatch, s3, images, cache_dir, fake_time
    ):
        uri = list(images)[0]
        fouaws.cache_s3_files([uri], cache_dir=cache_dir, client=s3)

        head_object = s3.head_object
        calls = []

        def _head_object(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise botocore.exceptions.ConnectionError(error="reset")

            return head_object(**kwargs)

        monkeypatch.setattr(s3, "head_object", _head_object)
        download_calls = _patch_download_file(monkeypatch, s3)

        fouaws.cache_s3_files(
            [uri], cache_dir=cache_dir, validate="size", client=s3
        )

        assert len(calls) == 2
        assert fake_time.sleeps == [1]
        assert download_calls == []

    def test_default_cache_dir_and_client(self, s3, images, tmp_path):
        cache_dir = str(tmp_path / "default-cache")
        uris = list(images)

        with mock.patch.object(fo.config, "s3_cache_dir", cache_dir):
            paths = fouaws.cache_s3_files(uris)

        assert paths == [_cache_path(uri, cache_dir) for uri in uris]
        assert all(os.path.isfile(p) for p in paths)


class TestCacheCollisions:
    def test_case_insensitive_collisions(self, monkeypatch, s3, cache_dir):
        lower = _put(s3, "dir/img.jpg", _image_bytes(8, 8))
        upper = _put(s3, "dir/IMG.jpg", _image_bytes(16, 16))
        monkeypatch.setattr(fouaws, "_is_case_insensitive", lambda d: True)

        with pytest.raises(ValueError, match="same file"):
            fouaws.cache_s3_files(
                [lower, upper], cache_dir=cache_dir, client=s3
            )

        paths = fouaws.cache_s3_files(
            [lower, upper], cache_dir=cache_dir, skip_failures=True, client=s3
        )
        assert paths == [_cache_path(lower, cache_dir), None]

    def test_case_sensitive_filesystem(self, monkeypatch, s3, cache_dir):
        lower = _put(s3, "dir/img.jpg", _image_bytes(8, 8))
        upper = _put(s3, "dir/IMG.jpg", _image_bytes(16, 16))
        monkeypatch.setattr(fouaws, "_is_case_insensitive", lambda d: False)

        paths = fouaws.cache_s3_files(
            [lower, upper], cache_dir=cache_dir, client=s3
        )
        assert paths == [
            _cache_path(lower, cache_dir),
            _cache_path(upper, cache_dir),
        ]

    def test_is_case_insensitive_probe(self, tmp_path):
        result = fouaws._is_case_insensitive(str(tmp_path / "cache"))
        assert isinstance(result, bool)
        assert os.listdir(tmp_path / "cache") == []

    def test_read_only_cache(self, monkeypatch, s3, images, cache_dir):
        uris = list(images)
        paths = fouaws.cache_s3_files(uris, cache_dir=cache_dir, client=s3)

        def _mkstemp(*args, **kwargs):
            raise PermissionError("Read-only file system")

        # A fully cached, read-only cache is still usable
        monkeypatch.setattr(fouaws.tempfile, "mkstemp", _mkstemp)
        assert fouaws.cache_s3_files(uris, cache_dir=cache_dir) == paths


class TestAddS3Samples:
    def test_add_from_prefix(self, fixture_dataset, s3, images, cache_dir):
        s3.put_object(Bucket=BUCKET, Key="dir/notes.txt", Body=b"text")
        s3.put_object(Bucket=BUCKET, Key="other/x.jpg", Body=b"data")
        dataset = fixture_dataset()

        sample_ids = fouaws.add_s3_samples(
            dataset,
            _uri("dir/"),
            cache_dir=cache_dir,
            progress=False,
            client=s3,
        )

        assert len(sample_ids) == len(images)
        assert len(dataset) == len(images)

        field = dataset.get_field("s3_uri")
        assert isinstance(field, fo.StringField)
        assert field.description == "The S3 URI of the sample's source media"

        for sample in dataset.select(sample_ids):
            assert sample.s3_uri in images
            assert sample.filepath == _cache_path(sample.s3_uri, cache_dir)
            assert os.path.isfile(sample.filepath)
            assert isinstance(sample.metadata, fo.ImageMetadata)
            width, height = images[sample.s3_uri]
            assert sample.metadata.width == width
            assert sample.metadata.height == height

        # Adding the same objects again is a no-op
        assert (
            fouaws.add_s3_samples(
                dataset,
                _uri("dir/"),
                cache_dir=cache_dir,
                progress=False,
                client=s3,
            )
            == []
        )
        assert len(dataset) == len(images)

    def test_add_from_list(self, fixture_dataset, s3, images, cache_dir):
        dataset = fixture_dataset()
        uris = list(images)

        sample_ids = fouaws.add_s3_samples(
            dataset,
            uris + [uris[0]],
            uri_field="source_uri",
            cache_dir=cache_dir,
            progress=False,
            client=s3,
        )

        assert len(sample_ids) == len(uris)
        assert not dataset.has_sample_field("s3_uri")
        assert dataset.values("source_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]

        # Only new objects are added
        new_uri = _put(s3, "dir/new.jpg", _image_bytes(4, 6))
        sample_ids = fouaws.add_s3_samples(
            dataset,
            uris + [new_uri],
            uri_field="source_uri",
            cache_dir=cache_dir,
            progress=False,
            client=s3,
        )

        assert len(sample_ids) == 1
        assert dataset[sample_ids[0]].source_uri == new_uri
        assert dataset[sample_ids[0]].metadata.width == 4
        assert len(dataset) == len(uris) + 1

    def test_add_from_prefix_with_extensions(
        self, fixture_dataset, s3, images, cache_dir
    ):
        dataset = fixture_dataset()

        fouaws.add_s3_samples(
            dataset,
            _uri("dir/"),
            extensions="png",
            cache_dir=cache_dir,
            progress=False,
            client=s3,
        )

        assert dataset.values("s3_uri") == [_uri("dir/b.png")]

    def test_skip_failures(self, fixture_dataset, s3, images, cache_dir):
        dataset = fixture_dataset()
        uris = list(images)
        missing_uri = _uri("dir/missing.jpg")

        with pytest.raises(RuntimeError, match="dir/missing.jpg"):
            fouaws.add_s3_samples(
                dataset,
                uris + [missing_uri],
                cache_dir=cache_dir,
                progress=False,
                client=s3,
            )

        assert len(dataset) == 0

        sample_ids = fouaws.add_s3_samples(
            dataset,
            uris + [missing_uri],
            cache_dir=cache_dir,
            skip_failures=True,
            progress=False,
            client=s3,
        )

        assert len(sample_ids) == len(uris)
        assert dataset.values("s3_uri") == uris

    def test_no_metadata(self, fixture_dataset, s3, images, cache_dir):
        dataset = fixture_dataset()

        fouaws.add_s3_samples(
            dataset,
            list(images),
            cache_dir=cache_dir,
            compute_metadata=False,
            progress=False,
            client=s3,
        )

        assert dataset.values("metadata") == [None] * len(images)

    def test_default_cache_dir_and_client(
        self, fixture_dataset, s3, images, tmp_path
    ):
        cache_dir = str(tmp_path / "default-cache")
        dataset = fixture_dataset()

        with mock.patch.object(fo.config, "s3_cache_dir", cache_dir):
            fouaws.add_s3_samples(dataset, _uri("dir/"), progress=False)

        for filepath, uri in zip(*dataset.values(["filepath", "s3_uri"])):
            assert filepath == _cache_path(uri, cache_dir)
            assert os.path.isfile(filepath)


class TestSyncS3Samples:
    def _make_dataset(self, fixture_dataset, filepaths):
        """Creates a dataset whose filepaths are exactly the given values,
        which may be S3 URIs.
        """
        dataset = fixture_dataset(
            filepaths=[
                "/placeholder/%d.jpg" % i for i in range(len(filepaths))
            ]
        )
        dataset.set_values("filepath", filepaths)
        return dataset

    def test_backfill_from_s3_filepaths(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        assert dataset.values("filepath") == uris

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=3, num_backfilled=3, num_downloaded=3, num_updated=3
        )

        field = dataset.get_field("s3_uri")
        assert isinstance(field, fo.StringField)
        assert field.description == "The S3 URI of the sample's source media"

        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]
        assert dataset.values("metadata.height") == [
            images[uri][1] for uri in uris
        ]

    def test_backfill_from_mangled_filepaths(
        self, fixture_dataset, s3, images, cache_dir
    ):
        special_uri = _put(s3, "dir/with space ?#ü.jpg", _image_bytes(7, 5))
        uris = list(images) + [special_uri]

        # This is what happens when S3 URIs are used as sample filepaths
        dataset = fixture_dataset(filepaths=uris)
        filepaths = dataset.values("filepath")
        assert filepaths == [
            os.path.join(os.getcwd(), "s3:", uri[len("s3://") :])
            for uri in uris
        ]

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=4, num_backfilled=4, num_downloaded=4, num_updated=4
        )
        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [32, 16, 20, 7]

    @pytest.mark.parametrize(
        "s3_root,strip",
        [
            ("s3://%s" % BUCKET, ""),
            ("s3://%s/" % BUCKET, ""),
            ("s3://%s/dir" % BUCKET, "dir/"),
            ("s3://%s/dir/" % BUCKET, "dir/"),
        ],
    )
    def test_backfill_from_local_root(
        self, fixture_dataset, s3, images, cache_dir, tmp_path, s3_root, strip
    ):
        local_root = tmp_path / "local"
        uris = list(images)
        filepaths = []
        for uri in uris:
            _, key = fouaws.parse_s3_uri(uri)
            filepaths.append(
                str(local_root.joinpath(*key[len(strip) :].split("/")))
            )

        dataset = fixture_dataset(filepaths=filepaths)

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            s3_root=s3_root,
            local_root=str(local_root),
            progress=False,
            client=s3,
        )

        assert summary == _summary(
            num_samples=3, num_backfilled=3, num_downloaded=3, num_updated=3
        )
        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]

    def test_backfill_from_relative_local_root(
        self, fixture_dataset, monkeypatch, s3, images, cache_dir, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        uri = list(images)[0]
        dataset = fixture_dataset(
            filepaths=[str(tmp_path / "local" / "dir" / "a.jpg")]
        )

        fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            s3_root="s3://%s" % BUCKET,
            local_root="local",
            progress=False,
            client=s3,
        )

        assert dataset.values("s3_uri") == [uri]

    def test_backfill_from_cache_dir(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        filepaths = [_cache_path(uri, cache_dir) for uri in uris]

        # For example, a dataset that was imported on another machine
        dataset = fixture_dataset(filepaths=filepaths)
        assert not os.path.exists(cache_dir)

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=3, num_backfilled=3, num_downloaded=3
        )
        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == filepaths
        assert all(os.path.isfile(p) for p in filepaths)
        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]

    def test_unresolved_samples_are_untouched(
        self, fixture_dataset, s3, images, cache_dir, tmp_path
    ):
        uri = list(images)[0]
        local_path = str(tmp_path / "elsewhere" / "x.jpg")
        _write(local_path, _image_bytes(3, 3))
        bucket_dir = os.path.join(cache_dir, BUCKET)
        dataset = self._make_dataset(
            fixture_dataset,
            [uri, local_path, "s3://%s" % BUCKET, bucket_dir],
        )

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            compute_metadata=False,
            progress=False,
            client=s3,
        )

        assert summary == _summary(
            num_samples=4,
            num_backfilled=1,
            num_unresolved=3,
            num_downloaded=1,
            num_updated=1,
        )
        assert dataset.values("s3_uri") == [uri, None, None, None]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir),
            local_path,
            "s3://%s" % BUCKET,
            bucket_dir,
        ]

    def test_existing_uris_are_never_modified(
        self, fixture_dataset, s3, images, cache_dir, tmp_path
    ):
        uri_a, uri_b, uri_c = list(images)
        local_root = tmp_path / "local"
        filepaths = [
            # The filepath suggests a different object
            uri_b,
            # The filepath maps to a different object via local_root
            str(local_root / "dir" / "sub" / "c.jpg"),
            # The filepath is the cache path of a different object
            _cache_path(uri_a, cache_dir),
            # The URI field is empty, so it is backfilled
            uri_c,
        ]
        dataset = self._make_dataset(fixture_dataset, filepaths)
        dataset.set_values("s3_uri", [uri_a, uri_b, uri_c, None])

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            s3_root="s3://%s" % BUCKET,
            local_root=str(local_root),
            progress=False,
            client=s3,
        )

        assert summary == _summary(
            num_samples=4, num_backfilled=1, num_downloaded=3, num_updated=4
        )
        assert dataset.values("s3_uri") == [uri_a, uri_b, uri_c, uri_c]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in (uri_a, uri_b, uri_c, uri_c)
        ]
        assert dataset.values("metadata.width") == [32, 16, 20, 20]

    def test_dry_run(self, fixture_dataset, s3, images, cache_dir, api_calls):
        uri_a, uri_b, uri_c = list(images)
        filepaths = [
            uri_a,
            fo.Sample(filepath=uri_b).filepath,
            _cache_path(uri_c, cache_dir),
            "/elsewhere/x.jpg",
        ]
        dataset = self._make_dataset(fixture_dataset, filepaths)

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, dry_run=True, client=s3
        )

        assert summary == _summary(
            num_samples=4, num_backfilled=3, num_unresolved=1, num_updated=2
        )
        assert not dataset.has_sample_field("s3_uri")
        assert dataset.values("filepath") == filepaths
        assert dataset.values("metadata") == [None] * 4
        assert not os.path.exists(cache_dir)
        assert api_calls == []

    def test_dry_run_existing_uris(
        self, fixture_dataset, s3, images, cache_dir, tmp_path, api_calls
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )
        filepaths = dataset.values("filepath")
        del api_calls[:]

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, dry_run=True, client=s3
        )
        assert summary == _summary(num_samples=3)

        # Moving the cache would update every filepath
        other_cache_dir = str(tmp_path / "other-cache")
        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=other_cache_dir, dry_run=True, client=s3
        )
        assert summary == _summary(num_samples=3, num_updated=3)

        # Unmappable URIs would fail
        dataset.set_values("s3_uri", [_uri("dir/../a.jpg"), None, None])
        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, dry_run=True, client=s3
        )
        assert summary == _summary(
            num_samples=3, num_backfilled=2, num_failed=1
        )

        assert dataset.values("filepath") == filepaths
        assert not os.path.exists(other_cache_dir)
        assert api_calls == []

    def test_second_run_is_a_no_op(
        self, fixture_dataset, s3, images, cache_dir, api_calls
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )
        filepaths = dataset.values("filepath")
        mtimes = [os.path.getmtime(p) for p in filepaths]
        del api_calls[:]

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(num_samples=3)
        assert api_calls == []
        assert dataset.values("filepath") == filepaths
        assert dataset.values("s3_uri") == uris
        assert [os.path.getmtime(p) for p in filepaths] == mtimes

        # With validation, only metadata requests are made
        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            validate="etag",
            progress=False,
            client=s3,
        )

        assert summary == _summary(num_samples=3)
        assert api_calls == ["HeadObject"] * 3

    def test_skip_failures(self, fixture_dataset, s3, images, cache_dir):
        uri = list(images)[0]
        missing_uri = _uri("dir/missing.jpg")
        dataset = self._make_dataset(fixture_dataset, [uri, missing_uri])

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            skip_failures=True,
            progress=False,
            client=s3,
        )

        assert summary == _summary(
            num_samples=2,
            num_backfilled=1,
            num_downloaded=1,
            num_failed=1,
            num_updated=1,
        )

        # The URI of the missing object is not recorded, but it can still be
        # inferred from the unchanged filepath
        assert dataset.values("s3_uri") == [uri, None]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir),
            missing_uri,
        ]
        assert dataset.values("metadata.width") == [32, None]

    def test_failures_can_be_resumed(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uri = list(images)[0]
        missing_uri = _uri("dir/missing.jpg")
        mangled = fo.Sample(filepath=missing_uri).filepath
        dataset = self._make_dataset(fixture_dataset, [uri, mangled])

        with pytest.raises(RuntimeError, match="dir/missing.jpg"):
            fouaws.sync_s3_samples(
                dataset, cache_dir=cache_dir, progress=False, client=s3
            )

        # Unverified URIs are not recorded, and the filepaths from which they
        # are inferred are unchanged
        assert dataset.values("s3_uri") == [None, None]
        assert dataset.values("filepath") == [uri, mangled]

        # So the sync can be resumed once the object is available
        _put(s3, "dir/missing.jpg", _image_bytes(9, 9))
        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary["num_failed"] == 0
        assert summary["num_updated"] == 2
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir),
            _cache_path(missing_uri, cache_dir),
        ]
        assert dataset.values("metadata.width") == [32, 9]

    def test_view_only_updates_its_samples(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        view = dataset.skip(1)

        summary = fouaws.sync_s3_samples(
            view, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=2, num_backfilled=2, num_downloaded=2, num_updated=2
        )
        assert dataset.values("s3_uri") == [None] + uris[1:]
        assert dataset.values("filepath") == [uris[0]] + [
            _cache_path(uri, cache_dir) for uri in uris[1:]
        ]
        assert dataset.values("metadata.width") == [None, 16, 20]
        assert not os.path.exists(_cache_path(uris[0], cache_dir))

    @pytest.mark.parametrize("view_type", ["uri_field", "filepath"])
    def test_view_filtered_by_synced_fields(
        self, fixture_dataset, s3, images, cache_dir, view_type
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        dataset.add_sample_field("s3_uri", fo.StringField)

        if view_type == "uri_field":
            view = dataset.exists("s3_uri", False)
        else:
            view = dataset.match(F("filepath").starts_with("s3://"))

        summary = fouaws.sync_s3_samples(
            view, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=3, num_backfilled=3, num_downloaded=3, num_updated=3
        )
        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [32, 16, 20]

    def test_metadata_recomputed_when_filepath_changes(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        dataset.set_values(
            "metadata", [fo.ImageMetadata(width=1, height=1)] * 3
        )

        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert dataset.values("metadata.width") == [32, 16, 20]
        assert dataset.values("metadata.height") == [24, 8, 10]
        assert all(s > 0 for s in dataset.values("metadata.size_bytes"))

    def test_metadata_recomputed_when_redownloaded(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        # The cached file of the first sample is replaced by another image
        # whose metadata is then computed
        path = _cache_path(uris[0], cache_dir)
        stale = _image_bytes(5, 5, fmt="PNG")
        assert len(stale) != os.path.getsize(path)
        _write(path, stale)
        dataset.compute_metadata(overwrite=True, progress=False)
        assert dataset.values("metadata.width") == [5, 16, 20]

        # Without validation, the stale file is trusted
        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )
        assert summary == _summary(num_samples=3)
        assert dataset.values("metadata.width") == [5, 16, 20]

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            validate="size",
            progress=False,
            client=s3,
        )

        assert summary == _summary(num_samples=3, num_downloaded=1)
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris
        ]
        assert dataset.values("metadata.width") == [32, 16, 20]

    def test_missing_metadata_is_computed(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            compute_metadata=False,
            progress=False,
            client=s3,
        )
        assert dataset.values("metadata") == [None] * 3

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary(num_samples=3)
        assert dataset.values("metadata.width") == [32, 16, 20]

    def test_cache_dir_change(self, fixture_dataset, s3, images, tmp_path):
        uris = list(images)
        cache_dir1 = str(tmp_path / "cache1")
        cache_dir2 = str(tmp_path / "cache2")
        dataset = self._make_dataset(fixture_dataset, uris)
        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir1, progress=False, client=s3
        )

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir2, progress=False, client=s3
        )

        assert summary == _summary(
            num_samples=3, num_downloaded=3, num_updated=3
        )
        assert dataset.values("s3_uri") == uris
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir2) for uri in uris
        ]

    def test_custom_uri_field(self, fixture_dataset, s3, images, cache_dir):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)

        fouaws.sync_s3_samples(
            dataset,
            uri_field="source",
            cache_dir=cache_dir,
            progress=False,
            client=s3,
        )

        assert not dataset.has_sample_field("s3_uri")
        assert dataset.values("source") == uris

    def test_default_cache_dir_and_client(
        self, fixture_dataset, s3, images, tmp_path
    ):
        cache_dir = str(tmp_path / "default-cache")
        uri_a, uri_b, uri_c = list(images)
        dataset = self._make_dataset(
            fixture_dataset,
            [uri_a, _cache_path(uri_b, cache_dir), uri_c],
        )

        with mock.patch.object(fo.config, "s3_cache_dir", cache_dir):
            summary = fouaws.sync_s3_samples(dataset, progress=False)

        assert summary == _summary(
            num_samples=3, num_backfilled=3, num_downloaded=3, num_updated=2
        )
        assert dataset.values("s3_uri") == [uri_a, uri_b, uri_c]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in (uri_a, uri_b, uri_c)
        ]

    def test_empty_collection(self, fixture_dataset, s3, cache_dir):
        dataset = fixture_dataset()

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary == _summary()
        assert dataset.has_sample_field("s3_uri")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"s3_root": "s3://%s/dir" % BUCKET},
            {"local_root": "/local"},
            {"s3_root": "%s/dir" % BUCKET, "local_root": "/local"},
            {"s3_root": "s3://", "local_root": "/local"},
        ],
    )
    def test_invalid_roots(self, fixture_dataset, s3, cache_dir, kwargs):
        dataset = self._make_dataset(fixture_dataset, [_uri("dir/a.jpg")])

        with pytest.raises(ValueError):
            fouaws.sync_s3_samples(
                dataset, cache_dir=cache_dir, client=s3, **kwargs
            )

        assert not dataset.has_sample_field("s3_uri")

    def test_invalid_validate(self, fixture_dataset, s3, images, cache_dir):
        dataset = self._make_dataset(fixture_dataset, list(images))

        with pytest.raises(ValueError, match="validate"):
            fouaws.sync_s3_samples(
                dataset, cache_dir=cache_dir, validate="md5", client=s3
            )

        assert dataset.values("filepath") == list(images)


class TestSyncS3SamplesEdgeCases:
    def _make_dataset(self, fixture_dataset, filepaths):
        dataset = fixture_dataset(
            filepaths=[
                "/placeholder/%d.jpg" % i for i in range(len(filepaths))
            ]
        )
        dataset.set_values("filepath", filepaths)
        return dataset

    def test_empty_uri_values(self, fixture_dataset, s3, images, cache_dir):
        uri = list(images)[0]
        dataset = self._make_dataset(fixture_dataset, [uri, "/local/x.jpg"])
        dataset.add_sample_field("s3_uri", fo.StringField)
        dataset.set_values("s3_uri", ["", ""])

        summary = fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert summary["num_backfilled"] == 1
        assert summary["num_unresolved"] == 1
        assert dataset.values("s3_uri") == [uri, ""]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir),
            "/local/x.jpg",
        ]

    def test_grouped_dataset(self, s3, images, cache_dir):
        uris = list(images)
        dataset = fo.Dataset()
        try:
            group = fo.Group()
            dataset.add_samples(
                [
                    fo.Sample(
                        filepath="/placeholder/left.jpg",
                        group=group.element("left"),
                    ),
                    fo.Sample(
                        filepath="/placeholder/right.jpg",
                        group=group.element("right"),
                    ),
                ]
            )
            flat = dataset.select_group_slices(_allow_mixed=True)
            flat.set_values("filepath", uris[:2])

            summary = fouaws.sync_s3_samples(
                dataset, cache_dir=cache_dir, progress=False, client=s3
            )

            assert summary["num_samples"] == 2
            assert summary["num_updated"] == 2
            flat = dataset.select_group_slices(_allow_mixed=True)
            assert sorted(flat.values("s3_uri")) == sorted(uris[:2])
            assert sorted(flat.values("filepath")) == sorted(
                _cache_path(uri, cache_dir) for uri in uris[:2]
            )
            assert None not in flat.values("metadata.width")

            with pytest.raises(ValueError, match="grouped"):
                fouaws.add_s3_samples(
                    dataset, uris[2:], cache_dir=cache_dir, client=s3
                )
        finally:
            dataset.delete()

    def test_group_slice_view(self, s3, images, cache_dir):
        uris = list(images)
        dataset = fo.Dataset()
        try:
            group = fo.Group()
            dataset.add_samples(
                [
                    fo.Sample(
                        filepath="/placeholder/left.jpg",
                        group=group.element("left"),
                    ),
                    fo.Sample(
                        filepath="/placeholder/right.jpg",
                        group=group.element("right"),
                    ),
                ]
            )
            dataset.select_group_slices(_allow_mixed=True).set_values(
                "filepath", uris[:2]
            )
            left = dataset.select_group_slices("left")
            right = dataset.select_group_slices("right")

            summary = fouaws.sync_s3_samples(
                left, cache_dir=cache_dir, progress=False, client=s3
            )

            assert summary["num_updated"] == 1
            assert left.values("filepath") == [_cache_path(uris[0], cache_dir)]
            assert left.values("s3_uri") == [uris[0]]
            assert right.values("filepath") == [uris[1]]
            assert right.values("s3_uri") == [None]
        finally:
            dataset.delete()

    def test_generated_views_raise(
        self, fixture_dataset, s3, images, cache_dir
    ):
        dataset = self._make_dataset(fixture_dataset, list(images))
        dataset.set_values(
            "ground_truth",
            [
                fo.Detections(detections=[fo.Detection(label="a")])
                for _ in range(len(dataset))
            ],
        )

        with pytest.raises(ValueError, match="source collection"):
            fouaws.sync_s3_samples(
                dataset.to_patches("ground_truth"),
                cache_dir=cache_dir,
                client=s3,
            )

    def test_metadata_computed_in_batches(
        self, monkeypatch, fixture_dataset, s3, images, cache_dir
    ):
        monkeypatch.setattr(fouaws, "_SAMPLE_BATCH_SIZE", 2)
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)

        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )

        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]

    def test_most_specific_root_wins(
        self, fixture_dataset, s3, images, cache_dir, tmp_path
    ):
        # The cache directory is inside `local_root`
        uri = list(images)[0]
        local_root = os.path.dirname(cache_dir)
        dataset = self._make_dataset(
            fixture_dataset, [_cache_path(uri, cache_dir)]
        )

        fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            local_root=local_root,
            s3_root="s3://other-bucket/prefix",
            progress=False,
            client=s3,
        )

        assert dataset.values("s3_uri") == [uri]

    def test_stale_metadata_cleared_without_computing(
        self, fixture_dataset, s3, images, cache_dir
    ):
        uris = list(images)
        dataset = self._make_dataset(fixture_dataset, uris)
        dataset.set_values(
            "metadata", [fo.ImageMetadata(width=1, height=1)] * len(uris)
        )

        fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            compute_metadata=False,
            progress=False,
            client=s3,
        )

        # Metadata that described other media is not kept
        assert dataset.values("metadata") == [None] * len(uris)

        fouaws.sync_s3_samples(
            dataset, cache_dir=cache_dir, progress=False, client=s3
        )
        assert dataset.values("metadata.width") == [
            images[uri][0] for uri in uris
        ]

    def test_wrong_inference_is_not_recorded(
        self, fixture_dataset, s3, images, cache_dir, tmp_path
    ):
        local_root = str(tmp_path / "local")
        dataset = self._make_dataset(
            fixture_dataset, [os.path.join(local_root, "dir", "a.jpg")]
        )

        summary = fouaws.sync_s3_samples(
            dataset,
            cache_dir=cache_dir,
            local_root=local_root,
            s3_root="s3://%s/wrong" % BUCKET,
            skip_failures=True,
            progress=False,
            client=s3,
        )

        assert summary["num_failed"] == 1
        assert summary["num_backfilled"] == 0
        assert dataset.values("s3_uri") == [None]


def _num_added(out):
    match = re.search(r"^num_added\s+(\d+)\s*$", out, flags=re.MULTILINE)
    assert match is not None, out
    return int(match.group(1))


def _run_cli(monkeypatch, *args):
    argv = ["fiftyone", "utils", "sync-s3-media"] + [str(a) for a in args]
    monkeypatch.setattr(sys, "argv", argv)
    focl.main()


class TestSyncS3MediaCommandValidation:
    def test_roots_validated_before_changes(
        self, monkeypatch, s3, images, cache_dir, dataset_name
    ):
        with pytest.raises(SystemExit):
            _run_cli(
                monkeypatch,
                dataset_name,
                "--prefix",
                "s3://%s/dir/" % BUCKET,
                "--s3-root",
                "s3://%s/dir" % BUCKET,
                "--cache-dir",
                cache_dir,
            )

        assert not fo.dataset_exists(dataset_name)
        assert not os.path.exists(os.path.join(cache_dir, BUCKET))

    def test_invalid_s3_root_validated_before_changes(
        self, monkeypatch, s3, images, cache_dir, dataset_name, tmp_path
    ):
        with pytest.raises(SystemExit):
            _run_cli(
                monkeypatch,
                dataset_name,
                "--prefix",
                "s3://%s/dir/" % BUCKET,
                "--s3-root",
                "s3://",
                "--local-root",
                tmp_path,
                "--cache-dir",
                cache_dir,
            )

        assert not fo.dataset_exists(dataset_name)


class TestSyncS3MediaCommand:
    def test_create_and_extend_dataset_from_prefix(
        self, monkeypatch, capsys, s3, images, cache_dir, dataset_name
    ):
        prefix = _uri("dir/")

        _run_cli(
            monkeypatch,
            dataset_name,
            "--prefix",
            prefix,
            "--cache-dir",
            cache_dir,
        )

        out = capsys.readouterr().out
        assert _num_added(out) == 3
        assert "num_samples" in out

        assert fo.dataset_exists(dataset_name)
        dataset = fo.load_dataset(dataset_name)
        assert dataset.persistent
        assert sorted(dataset.values("s3_uri")) == sorted(images)
        for sample in dataset:
            assert sample.filepath == _cache_path(sample.s3_uri, cache_dir)
            assert sample.metadata.width == images[sample.s3_uri][0]

        # A dry run reports the new objects without making changes
        new_uri = _put(s3, "dir/new.jpg", _image_bytes(4, 4))
        _run_cli(
            monkeypatch,
            dataset_name,
            "--prefix",
            prefix,
            "--cache-dir",
            cache_dir,
            "--dry-run",
        )

        out = capsys.readouterr().out
        assert _num_added(out) == 1
        assert len(dataset) == 3
        assert not os.path.exists(_cache_path(new_uri, cache_dir))

        # Only the new object is added
        _run_cli(
            monkeypatch,
            dataset_name,
            "--prefix",
            prefix,
            "--cache-dir",
            cache_dir,
        )

        out = capsys.readouterr().out
        assert _num_added(out) == 1
        assert len(dataset) == 4
        assert dataset.count_values("s3_uri")[new_uri] == 1
        assert os.path.isfile(_cache_path(new_uri, cache_dir))

    def test_dry_run_new_dataset(
        self, monkeypatch, capsys, s3, images, cache_dir, dataset_name
    ):
        _run_cli(
            monkeypatch,
            dataset_name,
            "--prefix",
            _uri("dir/"),
            "--cache-dir",
            cache_dir,
            "--dry-run",
        )

        out = capsys.readouterr().out
        assert (
            "Dataset '%s' would be created with 3 sample(s)" % dataset_name
            in out
        )
        assert not fo.dataset_exists(dataset_name)
        assert not os.path.exists(cache_dir)

    def test_missing_dataset_without_uris(
        self, monkeypatch, s3, cache_dir, dataset_name
    ):
        with pytest.raises(ValueError, match="not found"):
            _run_cli(monkeypatch, dataset_name, "--cache-dir", cache_dir)

        assert not fo.dataset_exists(dataset_name)

    def test_uris_and_uris_file(
        self,
        monkeypatch,
        capsys,
        s3,
        images,
        cache_dir,
        dataset_name,
        tmp_path,
    ):
        uri_a, uri_b, uri_c = list(images)
        uris_path = str(tmp_path / "uris.txt")
        with open(uris_path, "w", encoding="utf-8") as f:
            f.write("%s\n\n  %s  \n%s\n" % (uri_b, uri_c, uri_b))

        _run_cli(
            monkeypatch,
            dataset_name,
            "--uris",
            uri_a,
            uri_b,
            "--uris-file",
            uris_path,
            "--cache-dir",
            cache_dir,
        )

        assert _num_added(capsys.readouterr().out) == 3
        dataset = fo.load_dataset(dataset_name)
        assert dataset.values("s3_uri") == [uri_a, uri_b, uri_c]

    def test_skip_failures_and_no_metadata(
        self, monkeypatch, capsys, s3, images, cache_dir, dataset_name
    ):
        uri = list(images)[0]
        missing_uri = _uri("dir/missing.jpg")

        _run_cli(
            monkeypatch,
            dataset_name,
            "--uris",
            uri,
            missing_uri,
            "--cache-dir",
            cache_dir,
            "--skip-failures",
            "--no-metadata",
            "--max-attempts",
            "1",
            "-n",
            "1",
        )

        assert _num_added(capsys.readouterr().out) == 1
        dataset = fo.load_dataset(dataset_name)
        assert dataset.values("s3_uri") == [uri]
        assert dataset.values("metadata") == [None]

    def test_extensions_and_uri_field(
        self, monkeypatch, s3, images, cache_dir, dataset_name
    ):
        _run_cli(
            monkeypatch,
            dataset_name,
            "--prefix",
            _uri("dir/"),
            "--extensions",
            "png",
            "--uri-field",
            "source",
            "--cache-dir",
            cache_dir,
        )

        dataset = fo.load_dataset(dataset_name)
        assert not dataset.has_sample_field("s3_uri")
        assert dataset.values("source") == [_uri("dir/b.png")]

    def test_migrate_with_local_root(
        self,
        monkeypatch,
        capsys,
        fixture_dataset,
        s3,
        images,
        cache_dir,
        tmp_path,
    ):
        uris = list(images)
        local_root = tmp_path / "local"
        filepaths = [str(local_root / "a.jpg"), str(local_root / "b.png")]
        dataset = fixture_dataset(filepaths=filepaths)
        args = [
            dataset.name,
            "--local-root",
            str(local_root),
            "--s3-root",
            _uri("dir"),
            "--cache-dir",
            cache_dir,
        ]

        _run_cli(monkeypatch, *(args + ["--dry-run"]))

        out = capsys.readouterr().out
        assert "num_backfilled" in out
        assert not dataset.has_sample_field("s3_uri")
        assert dataset.values("filepath") == filepaths

        _run_cli(monkeypatch, *(args + ["--validate", "etag"]))

        assert dataset.values("s3_uri") == uris[:2]
        assert dataset.values("filepath") == [
            _cache_path(uri, cache_dir) for uri in uris[:2]
        ]
        assert dataset.values("metadata.width") == [32, 16]

    def test_migrate_legacy_dataset_with_prefix(
        self, monkeypatch, capsys, fixture_dataset, s3, images, cache_dir
    ):
        # A legacy dataset whose filepaths are S3 URIs is extended with all
        # objects under the prefix. Objects that are already in the dataset
        # must not be added again
        uri_a, uri_b, uri_c = list(images)
        dataset = fixture_dataset(filepaths=["/placeholder/a.jpg"] * 2)
        dataset.set_values("filepath", [uri_a, uri_b])

        _run_cli(
            monkeypatch,
            dataset.name,
            "--prefix",
            _uri("dir/"),
            "--cache-dir",
            cache_dir,
            "--dry-run",
        )

        assert _num_added(capsys.readouterr().out) == 1

        _run_cli(
            monkeypatch,
            dataset.name,
            "--prefix",
            _uri("dir/"),
            "--cache-dir",
            cache_dir,
        )

        assert len(dataset) == 3
        assert sorted(dataset.values("s3_uri")) == [uri_a, uri_b, uri_c]

    def test_default_cache_dir(
        self, monkeypatch, s3, images, tmp_path, dataset_name
    ):
        cache_dir = str(tmp_path / "default-cache")

        with mock.patch.object(fo.config, "s3_cache_dir", cache_dir):
            _run_cli(monkeypatch, dataset_name, "--prefix", _uri("dir/"))

        dataset = fo.load_dataset(dataset_name)
        for filepath, uri in zip(*dataset.values(["filepath", "s3_uri"])):
            assert filepath == _cache_path(uri, cache_dir)

    def test_client_options(
        self, monkeypatch, s3, images, cache_dir, dataset_name
    ):
        # The CLI imports fiftyone.utils.aws lazily, so its own reference to
        # the module must be patched
        with mock.patch.object(
            focl.fouaws, "get_s3_client", return_value=s3
        ) as get_s3_client:
            _run_cli(
                monkeypatch,
                dataset_name,
                "--prefix",
                _uri("dir/"),
                "--cache-dir",
                cache_dir,
                "--profile",
                "my-profile",
                "--region",
                "eu-central-1",
                "--endpoint-url",
                "https://minio.local:9000",
                "-n",
                "2",
            )

        get_s3_client.assert_called_once()
        kwargs = get_s3_client.call_args.kwargs
        assert kwargs["profile"] == "my-profile"
        assert kwargs["region"] == "eu-central-1"
        assert kwargs["endpoint_url"] == "https://minio.local:9000"
        assert len(fo.load_dataset(dataset_name)) == 3
