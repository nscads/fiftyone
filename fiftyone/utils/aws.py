"""
Utilities for working with `Amazon Web Services <https://aws.amazon.com>`.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import hashlib
import logging
import multiprocessing.dummy
import os
import re
import time
from urllib.parse import urlparse

import boto3
import botocore
import s3transfer.exceptions

import eta.core.utils as etau

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.core.media as fom
import fiftyone.core.storage as fos
import fiftyone.core.utils as fou

logger = logging.getLogger(__name__)


_S3_PREFIX = "s3://"

# The form that ``s3://bucket/key`` takes after it was passed through
# ``os.path.abspath()``, which is what happens when an S3 URI is provided as
# the ``filepath`` of a new sample: ``/current/dir/s3:/bucket/key``
_MANGLED_S3_URI_PATT = re.compile(r"(?:^|[\\/])s3:[\\/]+(.+)$")

# HTTP status codes of S3 errors that are worth retrying
_RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)

_RETRYABLE_ERRORS = (
    botocore.exceptions.ConnectionError,
    botocore.exceptions.HTTPClientError,
    botocore.exceptions.IncompleteReadError,
    s3transfer.exceptions.RetriesExceededError,
)

# ETags of objects encrypted with these methods are not MD5 digests
_NON_MD5_ETAG_ENCRYPTIONS = ("aws:kms", "aws:kms:dsse")

_HASH_CHUNK_SIZE = 8 * 1024**2


def download_public_s3_files(
    urls, download_dir=None, num_workers=None, overwrite=True
):
    """Download files from a public AWS S3 bucket using unsigned URLs.

    The `url` argument either accepts:

        -   A list of paths to objects in the s3 bucket::

                urls = ["s3://bucket_name/dir1/file1.ext", ...]

            When `urls` is a list, then the `download_dir` argument is required
            and all objects will be downloaded into that directory

        -   A dictionary mapping the paths of objects to files on disk to store
            each object::

                urls = {
                    "s3://bucket_name/dir1/file1.ext": "/path/to/local/file1.ext",
                    ...
                }

    Args:
        urls: either a list of URLs to objects in an s3 bucket, or a dict
            mapping these URLs to locations on disk. If `urls` is a list, then
            the `download_dir` argument is required
        download_dir (None): the directory to store all downloaded objects.
            This is only used if `urls` is a list
        num_workers (None): a suggested number of threads to use when
            downloading files
        overwrite (True): whether to overwrite existing files
    """
    if not isinstance(urls, dict):
        if download_dir is None:
            raise ValueError(
                "`download_dir` is required when `urls` is a list"
            )

        urls = {url: None for url in urls}

    if download_dir:
        etau.ensure_dir(download_dir)

    num_workers = fou.recommend_thread_pool_workers(num_workers)

    s3_client = boto3.client(
        "s3",
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
            max_pool_connections=max(10, num_workers),
        ),
    )

    inputs = _build_inputs(
        urls, s3_client, download_dir=download_dir, overwrite=overwrite
    )

    if not inputs:
        return

    if num_workers <= 1:
        _single_thread_download(inputs)
    else:
        _multi_thread_download(inputs, num_workers)


def get_s3_client(
    profile=None, region=None, endpoint_url=None, max_pool_connections=None
):
    """Returns an authenticated boto3 S3 client.

    Any settings that are not provided are resolved by boto3's standard
    configuration chain: environment variables such as ``AWS_PROFILE``,
    ``AWS_REGION`` and ``AWS_ENDPOINT_URL_S3``, then ``~/.aws/config`` and
    ``~/.aws/credentials``, then any instance or container role.

    Args:
        profile (None): an optional AWS profile name
        region (None): an optional AWS region name
        endpoint_url (None): an optional S3 endpoint URL, for example to use
            an S3-compatible service such as MinIO
        max_pool_connections (None): an optional maximum number of concurrent
            connections to allow

    Returns:
        a boto3 S3 client
    """
    session = boto3.Session(profile_name=profile, region_name=region)
    config = botocore.config.Config(
        retries={"mode": "standard"},
        max_pool_connections=max(10, max_pool_connections or 0),
    )
    return session.client("s3", endpoint_url=endpoint_url, config=config)


def is_s3_uri(path):
    """Determines whether the given path is an S3 URI like
    ``s3://bucket/key``.

    Args:
        path: a path

    Returns:
        True/False
    """
    return etau.is_str(path) and path.startswith(_S3_PREFIX)


def parse_s3_uri(s3_uri):
    """Parses the bucket and key from the given S3 URI.

    Unlike URL parsing, characters such as ``?`` and ``#`` that are valid in
    S3 keys are kept as part of the key.

    Example usage::

        import fiftyone.utils.aws as fouaws

        fouaws.parse_s3_uri("s3://bucket/path/to/image.jpg")
        # ('bucket', 'path/to/image.jpg')

    Args:
        s3_uri: an S3 URI like ``s3://bucket/path/to/image.jpg``

    Returns:
        a ``(bucket, key)`` tuple

    Raises:
        ValueError: if the URI is not a valid S3 object URI
    """
    bucket, key = _split_s3_uri(s3_uri)
    if not key:
        raise ValueError("S3 URI '%s' does not contain an object key" % s3_uri)

    return bucket, key


def get_s3_cache_path(s3_uri, cache_dir=None):
    """Returns the local cache path of the given S3 object.

    The path mirrors the object's bucket and key, so
    ``s3://bucket/path/to/image.jpg`` is cached at
    ``<cache_dir>/bucket/path/to/image.jpg``.

    Args:
        s3_uri: an S3 URI like ``s3://bucket/path/to/image.jpg``
        cache_dir (None): the cache directory. By default,
            ``fiftyone.config.s3_cache_dir`` is used

    Returns:
        the local path

    Raises:
        ValueError: if the URI is invalid, or if its key contains empty,
            ``.`` or ``..`` path segments, which cannot be mapped to a unique
            path inside the cache directory
    """
    cache_dir = _parse_cache_dir(cache_dir)
    bucket, key = parse_s3_uri(s3_uri)

    parts = [bucket] + key.split("/")
    for part in parts:
        if part in ("", ".", "..") or (
            os.sep != "/" and any(c in part for c in "\\:")
        ):
            raise ValueError(
                "S3 URI '%s' cannot be mapped to a unique local path" % s3_uri
            )

    return os.path.join(cache_dir, *parts)


def list_s3_uris(prefix, extensions=None, client=None):
    """Lists the URIs of the S3 objects under the given prefix.

    Args:
        prefix: an S3 prefix like ``s3://bucket/path/to/dir/``. All objects
            whose keys start with the prefix's key are included, so include a
            trailing slash to only list the contents of a directory
        extensions (None): an optional file extension or iterable of
            extensions to include, like ``".jpg"``. By default, only images
            are included
        client (None): a boto3 S3 client to use. By default, one is created
            via :func:`get_s3_client`

    Returns:
        a list of S3 URIs
    """
    bucket, key_prefix = _split_s3_uri(prefix)

    if extensions is not None:
        if etau.is_str(extensions):
            extensions = [extensions]

        extensions = tuple(
            (e if e.startswith(".") else "." + e).lower() for e in extensions
        )

    if client is None:
        client = get_s3_client()

    s3_uris = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            if extensions is not None:
                if not key.lower().endswith(extensions):
                    continue
            elif fom.get_media_type(key) != fom.IMAGE:
                continue

            s3_uris.append(_S3_PREFIX + bucket + "/" + key)

    return s3_uris


def cache_s3_files(
    s3_uris,
    cache_dir=None,
    validate=None,
    overwrite=False,
    num_workers=None,
    max_attempts=5,
    skip_failures=False,
    progress=None,
    client=None,
):
    """Downloads the given S3 objects to the local cache.

    Each object is downloaded to the path returned by
    :func:`get_s3_cache_path`, which mirrors its bucket and key. Objects that
    are already cached are skipped unless they fail the ``validate`` check or
    ``overwrite`` is True.

    Args:
        s3_uris: an iterable of S3 URIs like ``s3://bucket/path/to/image.jpg``
        cache_dir (None): the cache directory. By default,
            ``fiftyone.config.s3_cache_dir`` is used
        validate (None): how to validate files that are already cached.
            Supported values are:

            -   ``None``: trust existing files
            -   ``"size"``: re-download files whose size differs from the
                object's
            -   ``"etag"``: re-download files whose size or MD5-based ETag
                differs from the object's. Objects whose ETag is not an MD5
                digest, such as those encrypted via SSE-KMS or SSE-C, are
                validated by size

        overwrite (False): whether to re-download objects that are already
            cached
        num_workers (None): a suggested number of threads to use
        max_attempts (5): the maximum number of attempts to make per object
            when transient errors occur
        skip_failures (False): whether to gracefully continue without raising
            an error if an object cannot be downloaded
        progress (None): whether to render a progress bar (True/False), use
            the default value ``fiftyone.config.show_progress_bars`` (None),
            or a progress callback function to invoke instead
        client (None): a boto3 S3 client to use. By default, one is created
            via :func:`get_s3_client`

    Returns:
        a list of local paths, one per URI. When ``skip_failures`` is True,
        the paths of objects that could not be downloaded are ``None``
    """
    paths, _ = _cache_s3_files(
        s3_uris,
        cache_dir=cache_dir,
        validate=validate,
        overwrite=overwrite,
        num_workers=num_workers,
        max_attempts=max_attempts,
        skip_failures=skip_failures,
        progress=progress,
        client=client,
    )

    return paths


def add_s3_samples(
    dataset,
    s3_uris,
    uri_field="s3_uri",
    extensions=None,
    cache_dir=None,
    validate=None,
    compute_metadata=True,
    num_workers=None,
    max_attempts=5,
    skip_failures=False,
    progress=None,
    client=None,
):
    """Adds samples for the given S3 objects to the dataset.

    Each object is downloaded to the local cache via :func:`cache_s3_files`.
    The ``filepath`` of its sample is the cached path, and ``uri_field``
    stores the object's S3 URI, which remains the source of truth for the
    media.

    Objects whose URI is already stored in ``uri_field`` of a sample in the
    dataset are skipped, so this method can be called repeatedly to add new
    objects.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        s3_uris: an iterable of S3 URIs, or an S3 prefix like
            ``s3://bucket/path/to/dir/`` whose objects to add. See
            :func:`list_s3_uris`
        uri_field ("s3_uri"): the name of the sample field in which to store
            the S3 URIs
        extensions (None): when ``s3_uris`` is a prefix, an optional file
            extension or iterable of extensions to include. By default, only
            images are included
        cache_dir (None): the cache directory. By default,
            ``fiftyone.config.s3_cache_dir`` is used
        validate (None): how to validate files that are already cached. See
            :func:`cache_s3_files` for the supported values
        compute_metadata (True): whether to compute the metadata of the added
            samples
        num_workers (None): a suggested number of threads to use
        max_attempts (5): the maximum number of attempts to make per object
            when transient errors occur
        skip_failures (False): whether to gracefully continue without raising
            an error if an object cannot be downloaded. Objects that could not
            be downloaded are not added
        progress (None): whether to render a progress bar (True/False), use
            the default value ``fiftyone.config.show_progress_bars`` (None),
            or a progress callback function to invoke instead
        client (None): a boto3 S3 client to use. By default, one is created
            via :func:`get_s3_client`

    Returns:
        a list of IDs of the samples that were added
    """
    if client is None:
        client = get_s3_client(
            max_pool_connections=fou.recommend_thread_pool_workers(num_workers)
        )

    if etau.is_str(s3_uris):
        s3_uris = list_s3_uris(s3_uris, extensions=extensions, client=client)

    _ensure_uri_field(dataset, uri_field)

    seen = set(dataset.distinct(uri_field))
    new_uris = []
    for s3_uri in s3_uris:
        if s3_uri not in seen:
            seen.add(s3_uri)
            new_uris.append(s3_uri)

    if not new_uris:
        logger.info("No new S3 objects to add")
        return []

    paths, _ = _cache_s3_files(
        new_uris,
        cache_dir=cache_dir,
        validate=validate,
        num_workers=num_workers,
        max_attempts=max_attempts,
        skip_failures=skip_failures,
        progress=progress,
        client=client,
    )

    samples = []
    for s3_uri, path in zip(new_uris, paths):
        if path is not None:
            sample = fo.Sample(filepath=path)
            sample[uri_field] = s3_uri
            samples.append(sample)

    sample_ids = dataset.add_samples(samples, progress=progress)

    if compute_metadata and sample_ids:
        dataset.select(sample_ids).compute_metadata(
            num_workers=num_workers, progress=progress
        )

    return sample_ids


def sync_s3_samples(
    sample_collection,
    uri_field="s3_uri",
    cache_dir=None,
    s3_root=None,
    local_root=None,
    validate=None,
    compute_metadata=True,
    dry_run=False,
    num_workers=None,
    max_attempts=5,
    skip_failures=False,
    progress=None,
    client=None,
):
    """Syncs the given samples with their S3 source media.

    After this method runs, every sample whose S3 URI is known has the URI
    stored in ``uri_field``, its media downloaded to the local cache, and its
    ``filepath`` set to the cached path. The method is idempotent, so it can
    be used to migrate existing datasets, and to repopulate the cache of a
    dataset on another machine or after changing the cache directory.

    Samples whose ``uri_field`` is empty are backfilled with a URI inferred
    from their current ``filepath``, which may be:

    -   an S3 URI like ``s3://bucket/key``
    -   an S3 URI that was turned into a local path like
        ``/current/dir/s3:/bucket/key``, which happens when an S3 URI is
        passed as the ``filepath`` of a new
        :class:`fiftyone.core.sample.Sample`
    -   a path under ``local_root``, which is mapped to the same relative path
        under ``s3_root``
    -   a path in the cache directory

    Existing ``uri_field`` values are never modified. Samples whose URI cannot
    be determined are left unchanged and reported.

    Args:
        sample_collection: a
            :class:`fiftyone.core.collections.SampleCollection`
        uri_field ("s3_uri"): the name of the sample field that stores the S3
            URIs
        cache_dir (None): the cache directory. By default,
            ``fiftyone.config.s3_cache_dir`` is used
        s3_root (None): an S3 prefix like ``s3://bucket/path/to/dir`` that
            corresponds to ``local_root``
        local_root (None): a local directory whose contents mirror
            ``s3_root``, used to infer the S3 URIs of samples whose media are
            local copies of the S3 objects
        validate (None): how to validate files that are already cached. See
            :func:`cache_s3_files` for the supported values
        compute_metadata (True): whether to compute the metadata of samples
            whose media changed or that have no metadata
        dry_run (False): whether to only report what would be done, without
            downloading media or modifying any samples
        num_workers (None): a suggested number of threads to use
        max_attempts (5): the maximum number of attempts to make per object
            when transient errors occur
        skip_failures (False): whether to gracefully continue without raising
            an error if an object cannot be downloaded. The ``filepath`` of
            such samples is left unchanged
        progress (None): whether to render a progress bar (True/False), use
            the default value ``fiftyone.config.show_progress_bars`` (None),
            or a progress callback function to invoke instead
        client (None): a boto3 S3 client to use. By default, one is created
            via :func:`get_s3_client`

    Returns:
        a dict containing the following counts:

        -   ``num_samples``: the number of samples in the collection
        -   ``num_backfilled``: the number of samples whose URI was inferred
            from their filepath
        -   ``num_unresolved``: the number of samples whose URI could not be
            determined
        -   ``num_downloaded``: the number of objects that were downloaded
        -   ``num_failed``: the number of samples whose media could not be
            downloaded
        -   ``num_updated``: the number of samples whose ``filepath`` was
            (or, when ``dry_run`` is True, would be) updated
    """
    if (s3_root is None) != (local_root is None):
        raise ValueError(
            "`s3_root` and `local_root` must be provided together"
        )

    if s3_root is not None:
        _split_s3_uri(s3_root)
        local_root = fos.normalize_path(local_root)

    cache_dir = _parse_cache_dir(cache_dir)
    dataset = sample_collection._dataset

    if not dry_run:
        _ensure_uri_field(dataset, uri_field)

    if sample_collection.has_field(uri_field):
        sample_ids, filepaths, s3_uris = sample_collection.values(
            ["id", "filepath", uri_field]
        )
    else:
        sample_ids, filepaths = sample_collection.values(["id", "filepath"])
        s3_uris = [None] * len(sample_ids)

    backfill = {}
    unresolved = []
    for idx, (sample_id, filepath, s3_uri) in enumerate(
        zip(sample_ids, filepaths, s3_uris)
    ):
        if s3_uri:
            continue

        s3_uri = _infer_s3_uri(filepath, cache_dir, local_root, s3_root)
        if s3_uri is None:
            unresolved.append(filepath)
        else:
            backfill[sample_id] = s3_uri
            s3_uris[idx] = s3_uri

    if unresolved:
        logger.warning(
            "Could not determine the S3 URIs of %d sample(s), whose filepaths "
            "will not be changed. For example: '%s'. If these are local "
            "copies of S3 objects, provide `local_root` and `s3_root`",
            len(unresolved),
            unresolved[0],
        )

    sync = [
        (_id, f, u)
        for _id, f, u in zip(sample_ids, filepaths, s3_uris)
        if u is not None
    ]

    summary = {
        "num_samples": len(sample_ids),
        "num_backfilled": len(backfill),
        "num_unresolved": len(unresolved),
        "num_downloaded": 0,
        "num_failed": 0,
        "num_updated": 0,
    }

    if dry_run:
        for _, filepath, s3_uri in sync:
            try:
                if get_s3_cache_path(s3_uri, cache_dir=cache_dir) != filepath:
                    summary["num_updated"] += 1
            except ValueError:
                summary["num_failed"] += 1

        return summary

    if backfill:
        # The source URIs are recorded first so that they are never lost,
        # even if downloading fails
        sample_collection.set_values(uri_field, backfill, key_field="id")

    if not sync:
        return summary

    sync_ids, sync_filepaths, sync_uris = zip(*sync)

    paths, downloaded = _cache_s3_files(
        sync_uris,
        cache_dir=cache_dir,
        validate=validate,
        num_workers=num_workers,
        max_attempts=max_attempts,
        skip_failures=skip_failures,
        progress=progress,
        client=client,
    )

    new_filepaths = {}
    synced_ids = []
    refresh_ids = []
    for sample_id, filepath, path in zip(sync_ids, sync_filepaths, paths):
        if path is None:
            continue

        synced_ids.append(sample_id)

        if path != filepath:
            new_filepaths[sample_id] = path

        if path != filepath or path in downloaded:
            refresh_ids.append(sample_id)

    if new_filepaths:
        sample_collection.set_values("filepath", new_filepaths, key_field="id")

    if compute_metadata:
        if refresh_ids:
            dataset.select(refresh_ids).compute_metadata(
                overwrite=True, num_workers=num_workers, progress=progress
            )

        if synced_ids:
            dataset.select(synced_ids).compute_metadata(
                num_workers=num_workers, progress=progress
            )

    summary["num_downloaded"] = len(downloaded)
    summary["num_failed"] = len(sync_ids) - len(synced_ids)
    summary["num_updated"] = len(new_filepaths)

    return summary


def _build_inputs(urls, s3_client, download_dir=None, overwrite=True):
    inputs = []
    for url, filepath in urls.items():
        bucket_name, object_path = _parse_url(url)
        if filepath is None:
            filepath = os.path.join(download_dir, object_path)

        if not os.path.isfile(filepath):
            inputs.append((bucket_name, object_path, filepath, s3_client))
        else:
            if overwrite:
                os.remove(filepath)
            else:
                logger.warning(
                    "File '%s' already exists, skipping...", filepath
                )

    return inputs


def _parse_url(url):
    result = urlparse(url, allow_fragments=False)
    return result.netloc, result.path.lstrip("/")


def _single_thread_download(inputs):
    with fou.ProgressBar(total=len(inputs), iters_str="files") as pb:
        for bucket_name, obj, filepath, s3_client in pb(inputs):
            s3_client.download_file(bucket_name, obj, filepath)


def _multi_thread_download(inputs, num_workers):
    with fou.ProgressBar(total=len(inputs), iters_str="files") as pb:
        with multiprocessing.dummy.Pool(num_workers) as pool:
            for _ in pool.imap_unordered(_do_s3_download, inputs):
                pb.update()


def _do_s3_download(args):
    bucket_name, obj, filepath, s3_client = args
    s3_client.download_file(bucket_name, obj, filepath)


def _split_s3_uri(s3_uri):
    if not is_s3_uri(s3_uri):
        raise ValueError(
            "Invalid S3 URI '%s'; expected 's3://bucket/key'" % s3_uri
        )

    bucket, _, key = s3_uri[len(_S3_PREFIX) :].partition("/")
    if not bucket:
        raise ValueError("S3 URI '%s' does not contain a bucket" % s3_uri)

    return bucket, key


def _parse_cache_dir(cache_dir):
    if cache_dir is None:
        cache_dir = fo.config.s3_cache_dir

    return fos.normalize_path(cache_dir)


def _ensure_uri_field(dataset, uri_field):
    if not dataset.has_sample_field(uri_field):
        dataset.add_sample_field(
            uri_field,
            fof.StringField,
            description="The S3 URI of the sample's source media",
        )


def _infer_s3_uri(filepath, cache_dir, local_root, s3_root):
    if not filepath:
        return None

    s3_uri = None

    if is_s3_uri(filepath):
        s3_uri = filepath
    else:
        match = _MANGLED_S3_URI_PATT.search(filepath)
        if match:
            s3_uri = _S3_PREFIX + match.group(1).replace(os.sep, "/")

    if s3_uri is None and local_root is not None:
        relpath = _get_relpath(filepath, local_root)
        if relpath is not None:
            s3_uri = s3_root.rstrip("/") + "/" + relpath

    if s3_uri is None:
        relpath = _get_relpath(filepath, cache_dir)
        if relpath is not None:
            s3_uri = _S3_PREFIX + relpath

    if s3_uri is None:
        return None

    try:
        parse_s3_uri(s3_uri)
    except ValueError:
        return None

    return s3_uri


def _get_relpath(path, root):
    path = os.path.abspath(path)

    try:
        if os.path.commonpath([root, path]) != root:
            return None
    except ValueError:
        # Paths are on different drives
        return None

    relpath = os.path.relpath(path, root)
    if relpath == os.curdir:
        return None

    return relpath.replace(os.sep, "/")


def _cache_s3_files(
    s3_uris,
    cache_dir=None,
    validate=None,
    overwrite=False,
    num_workers=None,
    max_attempts=5,
    skip_failures=False,
    progress=None,
    client=None,
):
    if validate not in (None, "size", "etag"):
        raise ValueError(
            "Unsupported validate=%s. Supported values are None, 'size', and "
            "'etag'" % validate
        )

    s3_uris = list(s3_uris)
    cache_dir = _parse_cache_dir(cache_dir)

    paths = []
    tasks = {}
    for s3_uri in s3_uris:
        try:
            path = get_s3_cache_path(s3_uri, cache_dir=cache_dir)
        except ValueError as e:
            if not skip_failures:
                raise

            if skip_failures != "ignore":
                logger.warning(e)

            path = None

        paths.append(path)
        if path is not None:
            tasks[path] = s3_uri

    num_workers = fou.recommend_thread_pool_workers(num_workers)

    if client is None:
        client = get_s3_client(max_pool_connections=num_workers)

    inputs = [
        (client, s3_uri, path, validate, overwrite, max_attempts)
        for path, s3_uri in tasks.items()
    ]

    downloaded = set()
    failed = set()

    def _process(results, pb):
        for s3_uri, path, was_downloaded, error in pb(results):
            if error is not None:
                if not skip_failures:
                    raise RuntimeError(
                        "Failed to download '%s' to '%s': %s"
                        % (s3_uri, path, error)
                    ) from error

                if skip_failures != "ignore":
                    logger.warning(
                        "Failed to download '%s': %s", s3_uri, error
                    )

                failed.add(path)
            elif was_downloaded:
                downloaded.add(path)

    with fou.ProgressBar(
        total=len(inputs), iters_str="files", progress=progress
    ) as pb:
        if num_workers <= 1 or len(inputs) <= 1:
            _process(map(_do_cache_s3_file, inputs), pb)
        else:
            with multiprocessing.dummy.Pool(processes=num_workers) as pool:
                _process(pool.imap_unordered(_do_cache_s3_file, inputs), pb)

    paths = [p if p not in failed else None for p in paths]

    return paths, downloaded


def _do_cache_s3_file(arg):
    client, s3_uri, path, validate, overwrite, max_attempts = arg

    try:
        was_downloaded = _cache_s3_file(
            client, s3_uri, path, validate, overwrite, max_attempts
        )
        return s3_uri, path, was_downloaded, None
    except Exception as e:
        return s3_uri, path, False, e


def _cache_s3_file(client, s3_uri, path, validate, overwrite, max_attempts):
    bucket, key = parse_s3_uri(s3_uri)

    if not overwrite and os.path.isfile(path):
        if validate is None or _is_cached_file_valid(
            client, bucket, key, path, validate, max_attempts
        ):
            return False

        logger.debug("Cached file '%s' is stale; re-downloading", path)

    fos.ensure_basedir(path)

    # boto3 downloads to a temporary file that is renamed when complete, so
    # an interrupted download is never mistaken for a cached file
    _retry(lambda: client.download_file(bucket, key, path), max_attempts)

    return True


def _is_cached_file_valid(client, bucket, key, path, validate, max_attempts):
    head = _retry(
        lambda: client.head_object(Bucket=bucket, Key=key), max_attempts
    )

    if os.path.getsize(path) != head["ContentLength"]:
        return False

    if validate == "size":
        return True

    etag = head.get("ETag", "").strip('"').lower()

    if (
        not etag
        or head.get("ServerSideEncryption") in _NON_MD5_ETAG_ENCRYPTIONS
        or head.get("SSECustomerAlgorithm")
    ):
        return True

    if "-" not in etag:
        return _compute_etag(path) == etag

    # Multipart ETags are the MD5 of the concatenated MD5s of each part
    num_parts = etag.rsplit("-", 1)[1]
    if not num_parts.isdigit():
        return True

    part = _retry(
        lambda: client.head_object(Bucket=bucket, Key=key, PartNumber=1),
        max_attempts,
    )

    return _compute_etag(path, part_size=part["ContentLength"]) == etag


def _compute_etag(path, part_size=None):
    if part_size is None:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
                md5.update(chunk)

        return md5.hexdigest()

    digests = []
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(part_size), b""):
            digests.append(hashlib.md5(part).digest())

    md5 = hashlib.md5(b"".join(digests))
    return "%s-%d" % (md5.hexdigest(), len(digests))


def _retry(fcn, max_attempts):
    attempt = 1
    while True:
        try:
            return fcn()
        except Exception as e:
            if attempt >= max_attempts or not _is_retryable(e):
                raise

            delay = min(2 ** (attempt - 1), 30)
            logger.debug("Retrying in %ds after error: %s", delay, e)
            time.sleep(delay)
            attempt += 1


def _is_retryable(error):
    if isinstance(error, botocore.exceptions.ClientError):
        status = error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        )
        return status in _RETRYABLE_STATUS_CODES

    return isinstance(error, _RETRYABLE_ERRORS)
