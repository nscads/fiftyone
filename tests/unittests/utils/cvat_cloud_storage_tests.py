"""
Unit tests for annotating media that CVAT reads from a cloud storage.

The tests run the real FiftyOne CVAT backend against an in-process fake CVAT
server, so no network access is required.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from copy import deepcopy
import json
import os
import re
import shutil
import tempfile
import types
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from PIL import Image
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

import fiftyone as fo
import fiftyone.utils.cvat as fouc

_URL = "http://fake-cvat"
_BUCKET = "bucket"
_CLOUD_STORAGE_ID = 7
_MAX_SLEEPS = 25

# (object key, (width, height), [(label, (x, y, w, h) in pixels), ...])
# The dataset order intentionally differs from the lexicographic order of the
# keys, and every image has a different size, so mapping a label to the wrong
# frame changes its relative bounding box
_SPECS = [
    ("images/delta.jpg", (64, 48), [("cat", (8, 4, 32, 24))]),
    (
        "images/alpha.jpg",
        (80, 60),
        [("dog", (10, 10, 40, 30)), ("cat", (50, 20, 20, 30))],
    ),
    ("images/echo.jpg", (100, 50), None),
    ("images/charlie.jpg", (40, 120), [("bird", (4, 12, 20, 60))]),
    ("images/bravo.jpg", (120, 90), [("dog", (30, 15, 60, 45))]),
]
_SIZES = {key: size for key, size, _ in _SPECS}
_WEATHER = {
    "images/delta.jpg": "sunny",
    "images/alpha.jpg": "rainy",
    "images/charlie.jpg": "cloudy",
    "images/bravo.jpg": "sunny",
}
_SORTED_KEYS = sorted(_SIZES.keys())


def _rel_box(px_box, size):
    x, y, w, h = px_box
    width, height = size
    return [x / width, y / height, w / width, h / height]


def _key(uri):
    return uri.split("://", 1)[1].split("/", 1)[1]


class _Call(object):
    def __init__(self, method, path, query, body, content_type):
        self.method = method
        self.path = path
        self.query = query
        self.body = body
        self.content_type = content_type

    def __repr__(self):
        return "%s %s" % (self.method, self.path)


def _parse_multipart(content_type, body):
    boundary = content_type.split("boundary=", 1)[1].strip('"').encode()
    fields = {}
    files = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, content = part.partition(b"\r\n\r\n")
        if content.endswith(b"\r\n"):
            content = content[:-2]

        head = head.decode()
        name = re.search(r'\bname="([^"]*)"', head).group(1)
        filename = re.search(r'\bfilename="([^"]*)"', head)
        if filename is not None:
            files[name] = (filename.group(1), content)
        else:
            fields[name] = content.decode()

    return {"fields": fields, "files": files}


def _parse_body(request):
    body = request.body
    if body is None:
        return None

    if isinstance(body, str):
        body = body.encode()

    content_type = request.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        return json.loads(body)

    if content_type.startswith("multipart/form-data"):
        return _parse_multipart(content_type, body)

    return body


def _make_response(request, status, payload=None, cookies=None):
    response = requests.Response()
    response.status_code = status
    response.reason = requests.status_codes._codes.get(status, ("",))[0]
    response.url = request.url
    response.request = request
    response.encoding = "utf-8"
    response.headers = CaseInsensitiveDict(
        {"Content-Type": "application/json"}
    )
    if payload is None:
        response._content = b""
    else:
        response._content = json.dumps(payload).encode()

    for name, value in (cookies or {}).items():
        response.cookies.set(name, value)

    return response


class FakeCVATServer(object):
    """An in-process fake of the CVAT >= 2.4.6 REST API (API v2).

    It emulates the endpoints that the FiftyOne CVAT backend uses to create
    tasks, upload media or reference media in a cloud storage, upload and
    download annotations, and delete tasks.

    Like real CVAT servers, tasks that read from a cloud storage name their
    frames by object key. Unless ``sort_frames`` is False, their frames are
    sorted lexicographically by name regardless of the requested
    ``sorting_method``, like servers that predate the ``predefined`` sorting
    method, which exercises the mapping of frames to samples by name.
    """

    def __init__(self, version="2.20.0", sort_frames=True):
        self.version = version
        self.sort_frames = sort_frames

        self.cloud_storages = {
            _CLOUD_STORAGE_ID: {
                "id": _CLOUD_STORAGE_ID,
                "provider_type": "AWS_S3_BUCKET",
                "resource": _BUCKET,
                "display_name": "company-s3",
                "credentials_type": "KEY_SECRET_KEY_PAIR",
                "manifests": [],
            }
        }

        # frame name -> (width, height)
        self.frame_sizes = {}

        # object key -> the frame name that CVAT reports for it
        self.frame_names = {}

        # object keys that CVAT silently drops from tasks
        self.missing_keys = set()

        # an error message with which creating task data fails
        self.data_error = None

        # whether the server predates the /api/requests endpoint and only
        # reports task data creation via the legacy /status endpoint
        self.legacy_status = False

        # the number of job listings of a task that are empty before its jobs
        # are created
        self.job_delay = 0

        # the format of the IDs of the requests that create task data
        self.rq_id_template = "create:task-%d"

        self.tasks = {}
        self.labels = {}
        self.attributes = {}
        self.jobs = {}
        self.rq = {}
        self.calls = []
        self.unhandled = []

        self._ids = {}

        self._routes = [
            ("POST", r"/api/auth/login", self._login),
            ("GET", r"/api/server/about", self._about),
            ("GET", r"/api/cloudstorages/(\d+)", self._get_cloud_storage),
            ("POST", r"/api/tasks", self._create_task),
            ("GET", r"/api/tasks/(\d+)", self._get_task),
            ("DELETE", r"/api/tasks/(\d+)", self._delete_task),
            ("POST", r"/api/tasks/(\d+)/data", self._upload_data),
            ("GET", r"/api/tasks/(\d+)/data/meta", self._get_data_meta),
            ("GET", r"/api/tasks/(\d+)/status", self._get_task_status),
            ("PUT", r"/api/tasks/(\d+)/annotations", self._put_annotations),
            ("GET", r"/api/labels", self._list_labels),
            ("GET", r"/api/jobs", self._list_jobs),
            ("GET", r"/api/jobs/(\d+)", self._get_job),
            ("GET", r"/api/jobs/(\d+)/annotations", self._get_job_annos),
            ("GET", r"/api/requests/([^/]+)", self._get_request),
        ]

    #
    # Transport
    #

    def handle(self, request):
        parts = urlsplit(request.url)
        path = parts.path.rstrip("/")
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        body = _parse_body(request)
        content_type = request.headers.get("Content-Type", None)
        self.calls.append(
            _Call(request.method, path, query, body, content_type)
        )

        for method, pattern, handler in self._routes:
            if method != request.method:
                continue

            match = re.fullmatch(pattern, path)
            if match is None:
                continue

            result = handler(*match.groups(), query=query, body=body)
            return _make_response(request, *result)

        self.unhandled.append("%s %s" % (request.method, request.url))
        return _make_response(request, 404, {"detail": "Not found."})

    #
    # Helpers for tests
    #

    def get_calls(self, method, pattern):
        return [
            c
            for c in self.calls
            if c.method == method and re.fullmatch(pattern, c.path)
        ]

    def data_request(self, task_id):
        return self.tasks[task_id]["data_request"]

    def frame_names_of(self, task_id):
        return [f["name"] for f in self.tasks[task_id]["frames"]]

    def label_name(self, label_id):
        return self.labels[label_id]["name"]

    def label_id(self, task_id, name):
        for label_id in self.tasks[task_id]["labels"]:
            if self.labels[label_id]["name"] == name:
                return label_id

        raise KeyError(name)

    def fo_label_id(self, shape):
        for attr in shape["attributes"]:
            if self.attributes[attr["spec_id"]]["name"] == "label_id":
                return attr["value"]

        return None

    def shapes_on(self, task_id, name):
        frame = self.frame_names_of(task_id).index(name)
        return [
            s for s in self.tasks[task_id]["shapes"] if s["frame"] == frame
        ]

    def tags_on(self, task_id, name):
        frame = self.frame_names_of(task_id).index(name)
        return [t for t in self.tasks[task_id]["tags"] if t["frame"] == frame]

    def edit_shape(self, task_id, shape_id, label=None, points=None):
        """Edits a shape or tag like a CVAT user would."""
        task = self.tasks[task_id]
        shape = next(
            a for a in task["shapes"] + task["tags"] if a["id"] == shape_id
        )
        if label is not None:
            # CVAT keeps the values of the attributes that the new label
            # shares by name with the old label
            new_label_id = self.label_id(task_id, label)
            new_attrs = {
                self.attributes[a]["name"]: a
                for a in self.labels[new_label_id]["attributes"]
            }
            shape["attributes"] = [
                {
                    "spec_id": new_attrs[
                        self.attributes[a["spec_id"]]["name"]
                    ],
                    "value": a["value"],
                }
                for a in shape["attributes"]
                if self.attributes[a["spec_id"]]["name"] in new_attrs
            ]
            shape["label_id"] = new_label_id

        if points is not None:
            shape["points"] = list(points)

        task["version"] += 1

    def delete_shape(self, task_id, shape_id):
        """Deletes a shape like a CVAT user would."""
        task = self.tasks[task_id]
        task["shapes"] = [s for s in task["shapes"] if s["id"] != shape_id]
        task["version"] += 1

    def permute_frames(self, task_id, order):
        """Reorders the frames of a task, keeping each annotation on its
        frame, so that new frame ``i`` is old frame ``order[i]``.
        """
        task = self.tasks[task_id]
        old_to_new = {old: new for new, old in enumerate(order)}
        task["frames"] = [task["frames"][old] for old in order]
        for anno in task["shapes"] + task["tags"]:
            anno["frame"] = old_to_new[anno["frame"]]

    #
    # Endpoints
    #

    def _next_id(self, kind):
        self._ids[kind] = self._ids.get(kind, 0) + 1
        return self._ids[kind]

    def _task_json(self, task):
        return {
            "id": task["id"],
            "name": task["name"],
            "url": "%s/api/tasks/%d" % (_URL, task["id"]),
            "labels": {
                "url": "%s/api/labels?task_id=%d" % (_URL, task["id"]),
                "count": len(task["labels"]),
            },
            "jobs": {
                "url": "%s/api/jobs?task_id=%d" % (_URL, task["id"]),
                "count": len(task["jobs"]),
            },
            "size": len(task["frames"] or []),
            "mode": "annotation",
            "status": "annotation",
            "assignee": None,
            "project_id": None,
            "segment_size": task["segment_size"],
            "image_quality": task["image_quality"],
            "updated_date": "2026-10-18T00:00:00Z",
        }

    def _label_json(self, label_id):
        label = self.labels[label_id]
        return {
            "id": label_id,
            "name": label["name"],
            "color": "#ff0000",
            "type": "any",
            "task_id": label["task_id"],
            "parent_id": None,
            "has_parent": False,
            "sublabels": [],
            "attributes": [
                deepcopy(self.attributes[attr_id])
                for attr_id in label["attributes"]
            ],
        }

    def _job_json(self, job_id):
        job = self.jobs[job_id]
        return {
            "id": job_id,
            "task_id": job["task_id"],
            "url": "%s/api/jobs/%d" % (_URL, job_id),
            "start_frame": job["start_frame"],
            "stop_frame": job["stop_frame"],
            "status": "annotation",
            "stage": "annotation",
            "state": "new",
            "type": "annotation",
            "assignee": None,
        }

    def _get_live_task(self, task_id):
        return self.tasks.get(int(task_id), None)

    def _login(self, query=None, body=None):
        return 200, {"key": "fake-token"}, {"csrftoken": "fake-csrf"}

    def _about(self, query=None, body=None):
        return 200, {
            "name": "Computer Vision Annotation Tool",
            "version": self.version,
        }

    def _get_cloud_storage(self, storage_id, query=None, body=None):
        storage = self.cloud_storages.get(int(storage_id), None)
        if storage is None:
            return 404, {"detail": "Not found."}

        return 200, deepcopy(storage)

    def _create_task(self, query=None, body=None):
        task_id = self._next_id("task")
        label_ids = []
        for label in body.get("labels", []):
            label_id = self._next_id("label")
            attr_ids = []
            for attr in label.get("attributes", []):
                attr_id = self._next_id("attribute")
                self.attributes[attr_id] = {
                    "id": attr_id,
                    "name": attr["name"],
                    "mutable": attr.get("mutable", False),
                    "input_type": attr["input_type"],
                    "default_value": attr.get("default_value", ""),
                    "values": attr.get("values", []),
                }
                attr_ids.append(attr_id)

            self.labels[label_id] = {
                "name": label["name"],
                "task_id": task_id,
                "attributes": attr_ids,
            }
            label_ids.append(label_id)

        self.tasks[task_id] = {
            "id": task_id,
            "name": body["name"],
            "image_quality": body.get("image_quality", 75),
            "segment_size": body.get("segment_size", None),
            "labels": label_ids,
            "jobs": [],
            "frames": None,
            "pending_job_polls": 0,
            "data_request": None,
            "put_requests": [],
            "shapes": [],
            "tags": [],
            "tracks": [],
            "version": 0,
        }

        return 201, self._task_json(self.tasks[task_id])

    def _get_task(self, task_id, query=None, body=None):
        task = self._get_live_task(task_id)
        if task is None:
            return 404, {"detail": "Not found."}

        return 200, self._task_json(task)

    def _delete_task(self, task_id, query=None, body=None):
        task = self.tasks.pop(int(task_id), None)
        if task is None:
            return 404, {"detail": "Not found."}

        for job_id in task["jobs"]:
            self.jobs.pop(job_id, None)

        return 204, None

    def _upload_data(self, task_id, query=None, body=None):
        task = self._get_live_task(task_id)
        if task is None:
            return 404, {"detail": "Not found."}

        content_type = self.calls[-1].content_type
        task["data_request"] = {"content_type": content_type, "body": body}

        if "files" in body:
            # Uploaded files, which CVAT stores in the predefined order
            names = [
                body["files"]["client_files[%d]" % idx][0]
                for idx in range(len(body["files"]))
            ]
            if body["fields"].get("sorting_method", None) != "predefined":
                names = sorted(names)

            frames = [(name, name) for name in names]
        else:
            # Objects in a cloud storage, which CVAT names by object key. They
            # are sorted by name, regardless of the requested sorting method
            frames = [
                (self.frame_names.get(key, key), key)
                for key in body["server_files"]
                if key not in self.missing_keys
            ]
            if self.sort_frames:
                frames = sorted(frames)

        rq_id = self.rq_id_template % task["id"]
        if self.data_error is not None:
            task["frames"] = []
            self.rq[rq_id] = {"status": "failed", "message": self.data_error}
        else:
            task["frames"] = [
                {
                    "name": name,
                    "width": self.frame_sizes[key][0],
                    "height": self.frame_sizes[key][1],
                    "related_files": 0,
                    "has_related_context": False,
                }
                for name, key in frames
            ]
            task["pending_job_polls"] = self.job_delay
            self.rq[rq_id] = {"status": "started", "message": ""}

        task["rq_id"] = rq_id

        if self.legacy_status:
            return 202, None

        return 202, {"rq_id": rq_id}

    def _maybe_create_jobs(self, task):
        if task["jobs"] or task["frames"] is None or self.data_error:
            return

        if task["pending_job_polls"] > 0:
            task["pending_job_polls"] -= 1
            return

        num_frames = len(task["frames"])
        segment_size = task["segment_size"] or num_frames
        for start in range(0, num_frames, segment_size):
            job_id = self._next_id("job")
            self.jobs[job_id] = {
                "task_id": task["id"],
                "start_frame": start,
                "stop_frame": min(start + segment_size, num_frames) - 1,
            }
            task["jobs"].append(job_id)

        self.rq[task["rq_id"]] = {"status": "finished", "message": ""}

    def _get_data_meta(self, task_id, query=None, body=None):
        task = self._get_live_task(task_id)
        if task is None:
            return 404, {"detail": "Not found."}

        frames = task["frames"] or []
        return 200, {
            "chunk_size": 72,
            "size": len(frames),
            "image_quality": task["image_quality"],
            "start_frame": 0,
            "stop_frame": max(len(frames) - 1, 0),
            "frame_filter": "",
            "frames": deepcopy(frames),
            "deleted_frames": [],
            "included_frames": None,
        }

    def _get_task_status(self, task_id, query=None, body=None):
        task = self._get_live_task(task_id)
        if task is None or not self.legacy_status:
            return 404, {"detail": "Not found."}

        rq = self.rq.get(task.get("rq_id", None), None)
        if rq is None:
            return 200, {"state": "Queued", "message": ""}

        return 200, {
            "state": rq["status"].capitalize(),
            "message": rq["message"],
        }

    def _get_request(self, rq_id, query=None, body=None):
        if self.legacy_status or rq_id not in self.rq:
            return 404, {"detail": "Not found."}

        rq = self.rq[rq_id]
        return 200, {
            "id": rq_id,
            "status": rq["status"],
            "message": rq["message"],
            "operation": {"type": "create:task"},
        }

    def _put_annotations(self, task_id, query=None, body=None):
        task = self._get_live_task(task_id)
        if task is None:
            return 404, {"detail": "Not found."}

        task["put_requests"].append(deepcopy(body))

        num_frames = len(task["frames"] or [])
        annos = {}
        for kind in ("shapes", "tags", "tracks"):
            annos[kind] = []
            for anno in body.get(kind, []):
                frame = anno.get("frame", None)
                if frame is None or not 0 <= frame < num_frames:
                    return 400, {"detail": "Invalid frame %s" % frame}

                label_id = anno["label_id"]
                if label_id not in task["labels"]:
                    return 400, {"detail": "Invalid label %s" % label_id}

                attr_ids = self.labels[label_id]["attributes"]
                for attr in anno.get("attributes", []):
                    if attr["spec_id"] not in attr_ids:
                        return 400, {"detail": "Invalid attribute"}

                anno = deepcopy(anno)
                anno["id"] = self._next_id("annotation")
                annos[kind].append(anno)

        task.update(annos)
        task["version"] += 1

        return 200, dict(version=task["version"], **deepcopy(annos))

    def _list_labels(self, query=None, body=None):
        task = self._get_live_task(query["task_id"])
        if task is None:
            return 404, {"detail": "Not found."}

        results = [self._label_json(label_id) for label_id in task["labels"]]
        return 200, {
            "count": len(results),
            "next": None,
            "previous": None,
            "results": results,
        }

    def _list_jobs(self, query=None, body=None):
        task = self._get_live_task(query["task_id"])
        if task is None:
            results = []
        else:
            self._maybe_create_jobs(task)
            results = [self._job_json(job_id) for job_id in task["jobs"]]

        return 200, {
            "count": len(results),
            "next": None,
            "previous": None,
            "results": results,
        }

    def _get_job(self, job_id, query=None, body=None):
        if int(job_id) not in self.jobs:
            return 404, {"detail": "Not found."}

        return 200, self._job_json(int(job_id))

    def _get_job_annos(self, job_id, query=None, body=None):
        job = self.jobs.get(int(job_id), None)
        if job is None:
            return 404, {"detail": "Not found."}

        task = self.tasks[job["task_id"]]

        def _in_job(anno):
            return job["start_frame"] <= anno["frame"] <= job["stop_frame"]

        return 200, {
            "version": task["version"],
            "shapes": deepcopy([s for s in task["shapes"] if _in_job(s)]),
            "tags": deepcopy([t for t in task["tags"] if _in_job(t)]),
            "tracks": [],
        }


class _FakeCVATAdapter(BaseAdapter):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def send(self, request, **kwargs):
        return self._server.handle(request)

    def close(self):
        pass


class _NoNetworkAdapter(BaseAdapter):
    def send(self, request, **kwargs):
        raise AssertionError("Unexpected request to %s" % request.url)

    def close(self):
        pass


class _RequestsProxy(object):
    """A stand-in for the ``requests`` module in :mod:`fiftyone.utils.cvat`
    whose sessions send requests to a fake CVAT server.
    """

    def __init__(self, server):
        self._server = server

    def Session(self):
        session = requests.Session()
        session.mount("http://", _NoNetworkAdapter())
        session.mount("https://", _NoNetworkAdapter())
        session.mount(_URL, _FakeCVATAdapter(self._server))
        return session

    def __getattr__(self, name):
        return getattr(requests, name)


class CVATCloudStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._paths = {}
        for idx, (key, size, _) in enumerate(_SPECS):
            # Mirror the layout of the S3 cache, like `sync_s3_samples()`
            path = os.path.join(cls._tmp_dir, _BUCKET, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.new("RGB", size, (40 * idx, 255 - 40 * idx, 128)).save(
                path, format="JPEG"
            )
            cls._paths[key] = path

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def setUp(self):
        self.server = FakeCVATServer()
        for key, size in _SIZES.items():
            self.server.frame_sizes[key] = size
            self.server.frame_sizes[os.path.basename(key)] = size

        self.num_sleeps = 0

        patchers = [
            mock.patch.object(fouc, "requests", _RequestsProxy(self.server)),
            mock.patch.object(
                fouc, "time", types.SimpleNamespace(sleep=self._sleep)
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        self._datasets = []

    def tearDown(self):
        for dataset in self._datasets:
            if not dataset.deleted:
                dataset.delete()

    def _sleep(self, seconds):
        self.num_sleeps += 1
        if self.num_sleeps > _MAX_SLEEPS:
            raise AssertionError("The CVAT client polled forever")

    #
    # Helpers
    #

    def _make_dataset(self, uris=None):
        dataset = fo.Dataset()
        self._datasets.append(dataset)

        samples = []
        for idx, (key, size, objects) in enumerate(_SPECS):
            if uris is not None:
                uri = uris[idx]
            else:
                uri = "s3://%s/%s" % (_BUCKET, key)

            sample = fo.Sample(filepath=self._paths[key], s3_uri=uri)
            if key in _WEATHER:
                sample["weather"] = fo.Classification(label=_WEATHER[key])

            if objects is not None:
                sample["ground_truth"] = fo.Detections(
                    detections=[
                        fo.Detection(
                            label=label, bounding_box=_rel_box(box, size)
                        )
                        for label, box in objects
                    ]
                )

            samples.append(sample)

        dataset.add_samples(samples)
        return dataset

    def _annotate(self, dataset, anno_key="cloud", **kwargs):
        kwargs.setdefault("media_field", "s3_uri")
        kwargs.setdefault("cloud_storage_id", _CLOUD_STORAGE_ID)
        kwargs.setdefault("label_field", "ground_truth")
        return dataset.annotate(
            anno_key,
            backend="cvat",
            url=_URL,
            username="u",
            password="p",
            **kwargs,
        )

    def _load(self, dataset, anno_key="cloud"):
        dataset.load_annotations(
            anno_key, url=_URL, username="u", password="p"
        )

    def _samples_by_key(self, dataset):
        dataset.reload()
        return {_key(s["s3_uri"]): s for s in dataset}

    def _snapshot(self, dataset):
        """Returns ``{key: [(id, label, bbox), ...]}`` for the dataset."""
        snapshot = {}
        for key, sample in self._samples_by_key(dataset).items():
            dets = sample["ground_truth"]
            snapshot[key] = [
                (d.id, d.label, list(d.bounding_box))
                for d in (dets.detections if dets is not None else [])
            ]

        return snapshot

    def _assert_boxes_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=6)

    def _assert_snapshots_equal(self, actual, expected, skip=()):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in expected:
            if key in skip:
                continue

            self.assertEqual(
                [(i, l) for i, l, _ in actual[key]],
                [(i, l) for i, l, _ in expected[key]],
                msg="labels of '%s' changed" % key,
            )
            for (_, _, a), (_, _, e) in zip(actual[key], expected[key]):
                self._assert_boxes_equal(a, e)

    def _assert_uploaded_labels_on_correct_frames(self, dataset, task_ids):
        # Maps every FiftyOne label ID to the key of its sample and its box
        expected = {}
        for key, sample in self._samples_by_key(dataset).items():
            if sample["ground_truth"] is None:
                continue

            for det in sample["ground_truth"].detections:
                expected[det.id] = (key, det.label, det.bounding_box)

        found = set()
        for task_id in task_ids:
            names = self.server.frame_names_of(task_id)
            frames = self.server.tasks[task_id]["frames"]
            put_requests = self.server.tasks[task_id]["put_requests"]
            self.assertGreaterEqual(len(put_requests), 1)

            for shape in put_requests[-1]["shapes"]:
                label_id = self.server.fo_label_id(shape)
                key, label, bbox = expected[label_id]
                frame = frames[shape["frame"]]

                self.assertEqual(names[shape["frame"]], key)
                self.assertEqual(
                    self.server.label_name(shape["label_id"]), label
                )

                width, height = _SIZES[key]
                self.assertEqual(
                    (frame["width"], frame["height"]), (width, height)
                )
                xtl, ytl, xbr, ybr = shape["points"]
                self._assert_boxes_equal(
                    [
                        xtl / width,
                        ytl / height,
                        (xbr - xtl) / width,
                        (ybr - ytl) / height,
                    ],
                    bbox,
                )
                found.add(label_id)

        self.assertEqual(found, set(expected.keys()))

    #
    # Tests
    #

    def test_round_trip(self):
        self.server.job_delay = 2
        dataset = self._make_dataset()
        sample_ids = {_key(s["s3_uri"]): s.id for s in dataset}
        before = self._snapshot(dataset)

        results = self._annotate(dataset)

        self.assertEqual(self.server.unhandled, [])
        self.assertEqual(len(results.task_ids), 1)
        task_id = results.task_ids[0]

        # The media was referenced in the cloud storage, not uploaded
        request = self.server.data_request(task_id)
        self.assertEqual(request["content_type"], "application/json")
        body = request["body"]
        self.assertNotIn("client_files", body)
        self.assertNotIn("files", body)
        self.assertEqual(body["cloud_storage_id"], _CLOUD_STORAGE_ID)
        self.assertIs(body["use_cache"], True)
        self.assertEqual(body["sorting_method"], "predefined")
        self.assertEqual(body["image_quality"], 75)
        self.assertEqual(body["server_files"], [key for key, _, _ in _SPECS])
        self.assertEqual(len(self.server.get_calls("POST", r"/api/tasks")), 1)

        # Polling for jobs checked the status of the data request
        self.assertGreater(self.num_sleeps, 0)
        self.assertGreater(
            len(self.server.get_calls("GET", r"/api/requests/.+")), 0
        )

        # CVAT sorted the frames by name, unlike the samples
        self.assertEqual(self.server.frame_names_of(task_id), _SORTED_KEYS)
        self.assertNotEqual(_SORTED_KEYS, [key for key, _, _ in _SPECS])

        # Frames were mapped to samples by key
        expected_map = {
            idx: {"sample_id": sample_ids[key], "name": key}
            for idx, key in enumerate(_SORTED_KEYS)
        }
        self.assertEqual(results.frame_id_map, {task_id: expected_map})

        stored = dataset.load_annotation_results("cloud", cache=False)
        self.assertEqual(stored.frame_id_map, {task_id: expected_map})

        # Existing labels were uploaded to the frames of their samples' keys
        self._assert_uploaded_labels_on_correct_frames(dataset, [task_id])
        shape = self.server.shapes_on(task_id, "images/delta.jpg")[0]
        self.assertEqual(
            shape["frame"], _SORTED_KEYS.index("images/delta.jpg")
        )

        # Edit one shape and delete another in CVAT
        bird = self.server.shapes_on(task_id, "images/charlie.jpg")
        self.assertEqual(len(bird), 1)
        bird = bird[0]
        bird_id = self.server.fo_label_id(bird)
        self.server.edit_shape(
            task_id, bird["id"], label="dog", points=[2.0, 30.0, 30.0, 90.0]
        )

        alpha_cat = [
            s
            for s in self.server.shapes_on(task_id, "images/alpha.jpg")
            if self.server.label_name(s["label_id"]) == "cat"
        ]
        self.assertEqual(len(alpha_cat), 1)
        alpha_cat_id = self.server.fo_label_id(alpha_cat[0])
        self.server.delete_shape(task_id, alpha_cat[0]["id"])

        self._load(dataset)

        self.assertEqual(self.server.unhandled, [])
        after = self._snapshot(dataset)

        # The edit landed on the sample with the edited key, and only there
        charlie = after["images/charlie.jpg"]
        self.assertEqual(len(charlie), 1)
        self.assertEqual(charlie[0][:2], (bird_id, "dog"))
        self._assert_boxes_equal(charlie[0][2], [0.05, 0.25, 0.7, 0.5])

        for key, dets in after.items():
            if key != "images/charlie.jpg":
                self.assertNotIn(bird_id, [d[0] for d in dets])
                for _, _, bbox in dets:
                    self.assertFalse(
                        all(
                            abs(a - e) < 1e-6
                            for a, e in zip(bbox, [0.05, 0.25, 0.7, 0.5])
                        )
                    )

        # The deleted label was removed from its sample only
        alpha = after["images/alpha.jpg"]
        self.assertEqual([d[1] for d in alpha], ["dog"])
        self.assertNotIn(alpha_cat_id, [d[0] for d in alpha])
        self.assertEqual(alpha[0][0], before["images/alpha.jpg"][0][0])
        self._assert_boxes_equal(alpha[0][2], before["images/alpha.jpg"][0][2])

        # The other samples are unchanged
        self._assert_snapshots_equal(
            after,
            before,
            skip=("images/charlie.jpg", "images/alpha.jpg"),
        )

        # Cleanup deletes the task
        results.cleanup()
        self.assertNotIn(task_id, self.server.tasks)
        self.assertEqual(
            len(self.server.get_calls("DELETE", r"/api/tasks/%d" % task_id)),
            1,
        )

    def test_multiple_tasks(self):
        dataset = self._make_dataset()
        sample_ids = {_key(s["s3_uri"]): s.id for s in dataset}
        before = self._snapshot(dataset)

        results = self._annotate(dataset, task_size=2)

        self.assertEqual(len(results.task_ids), 3)
        batches = [
            [key for key, _, _ in _SPECS[i : i + 2]] for i in range(0, 5, 2)
        ]
        for task_id, keys in zip(results.task_ids, batches):
            body = self.server.data_request(task_id)["body"]
            self.assertEqual(body["server_files"], keys)
            self.assertEqual(self.server.frame_names_of(task_id), sorted(keys))
            self.assertEqual(
                results.frame_id_map[task_id],
                {
                    idx: {"sample_id": sample_ids[key], "name": key}
                    for idx, key in enumerate(sorted(keys))
                },
            )

        self._assert_uploaded_labels_on_correct_frames(
            dataset, results.task_ids
        )

        # Edit a label in the second task
        task_id = results.task_ids[1]
        shape = self.server.shapes_on(task_id, "images/charlie.jpg")[0]
        self.server.edit_shape(task_id, shape["id"], points=[0, 0, 20, 60])

        self._load(dataset)

        after = self._snapshot(dataset)
        self._assert_boxes_equal(
            after["images/charlie.jpg"][0][2], [0.0, 0.0, 0.5, 0.5]
        )
        self._assert_snapshots_equal(
            after, before, skip=("images/charlie.jpg",)
        )

    def test_objects_outside_storage_prefix_raise(self):
        # CVAT >= 2.9.0 ignores objects outside of the storage's prefix
        self.server.cloud_storages[_CLOUD_STORAGE_ID][
            "specific_attributes"
        ] = "prefix=other/&region=us-east-1"
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("other/", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_objects_inside_storage_prefix(self):
        self.server.cloud_storages[_CLOUD_STORAGE_ID][
            "specific_attributes"
        ] = "prefix=images/"
        dataset = self._make_dataset()
        before = self._snapshot(dataset)

        self._annotate(dataset)
        self._load(dataset)

        self._assert_snapshots_equal(self._snapshot(dataset), before)

    def test_download_remaps_permuted_frames(self):
        dataset = self._make_dataset()
        before = self._snapshot(dataset)

        results = self._annotate(dataset)
        task_id = results.task_ids[0]

        # CVAT now reports the frames in a different order than when the
        # task was created
        self.server.permute_frames(task_id, [4, 2, 0, 3, 1])
        self.assertNotEqual(self.server.frame_names_of(task_id), _SORTED_KEYS)

        shape = self.server.shapes_on(task_id, "images/delta.jpg")[0]
        self.server.edit_shape(task_id, shape["id"], points=[0, 0, 32, 24])

        with self.assertLogs("fiftyone.utils.cvat", level="WARNING") as logs:
            self._load(dataset)

        self.assertTrue(
            any("not in the expected order" in msg for msg in logs.output)
        )

        after = self._snapshot(dataset)
        self.assertEqual(
            after["images/delta.jpg"][0][:2], before["images/delta.jpg"][0][:2]
        )
        self._assert_boxes_equal(
            after["images/delta.jpg"][0][2], [0.0, 0.0, 0.5, 0.5]
        )
        self._assert_snapshots_equal(after, before, skip=("images/delta.jpg",))

    def test_predefined_order_honored(self):
        # If CVAT keeps the requested order, frames map to samples in order
        self.server.sort_frames = False
        dataset = self._make_dataset()
        sample_ids = dataset.values("id")
        keys = [key for key, _, _ in _SPECS]
        before = self._snapshot(dataset)

        results = self._annotate(dataset)
        task_id = results.task_ids[0]

        self.assertEqual(self.server.frame_names_of(task_id), keys)
        self.assertEqual(
            results.frame_id_map[task_id],
            {
                idx: {"sample_id": sample_id, "name": key}
                for idx, (sample_id, key) in enumerate(zip(sample_ids, keys))
            },
        )
        self._assert_uploaded_labels_on_correct_frames(dataset, [task_id])

        shape = self.server.shapes_on(task_id, "images/bravo.jpg")[0]
        self.server.edit_shape(task_id, shape["id"], points=[0, 0, 60, 45])

        self._load(dataset)

        after = self._snapshot(dataset)
        self._assert_boxes_equal(
            after["images/bravo.jpg"][0][2], [0.0, 0.0, 0.5, 0.5]
        )
        self._assert_snapshots_equal(after, before, skip=("images/bravo.jpg",))

    def test_server_without_sorting_method(self):
        # CVAT < 2.4.6 does not support `sorting_method`
        self.server.version = "2.4.5"
        dataset = self._make_dataset()
        sample_ids = {_key(s["s3_uri"]): s.id for s in dataset}
        before = self._snapshot(dataset)

        results = self._annotate(dataset)
        task_id = results.task_ids[0]

        body = self.server.data_request(task_id)["body"]
        self.assertNotIn("sorting_method", body)
        self.assertEqual(body["cloud_storage_id"], _CLOUD_STORAGE_ID)
        self.assertEqual(
            results.frame_id_map[task_id],
            {
                idx: {"sample_id": sample_ids[key], "name": key}
                for idx, key in enumerate(_SORTED_KEYS)
            },
        )
        self._assert_uploaded_labels_on_correct_frames(dataset, [task_id])

        self._load(dataset)

        self._assert_snapshots_equal(self._snapshot(dataset), before)

    def test_existing_classifications_remapped(self):
        dataset = self._make_dataset()

        def _weather():
            dataset.reload()
            return {
                _key(s["s3_uri"]): (
                    (s["weather"].id, s["weather"].label)
                    if s["weather"] is not None
                    else None
                )
                for s in dataset
            }

        before = _weather()
        keys_by_label_id = {v[0]: k for k, v in before.items() if v}

        results = self._annotate(
            dataset, anno_key="weather", label_field="weather"
        )
        task_id = results.task_ids[0]
        names = self.server.frame_names_of(task_id)
        self.assertEqual(names, _SORTED_KEYS)

        # Existing tags were uploaded to the frames of their samples' keys
        tags = self.server.tasks[task_id]["put_requests"][-1]["tags"]
        self.assertEqual(len(tags), len(_WEATHER))
        for tag in tags:
            key = keys_by_label_id[self.server.fo_label_id(tag)]
            self.assertEqual(names[tag["frame"]], key)
            self.assertEqual(
                self.server.label_name(tag["label_id"]), _WEATHER[key]
            )

        tag = self.server.tags_on(task_id, "images/charlie.jpg")[0]
        self.server.edit_shape(task_id, tag["id"], label="rainy")

        self._load(dataset, anno_key="weather")

        expected = dict(before)
        expected["images/charlie.jpg"] = (
            before["images/charlie.jpg"][0],
            "rainy",
        )
        self.assertEqual(_weather(), expected)

    def test_local_media_field_raises(self):
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset, media_field="filepath")

        self.assertIn("media_field", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_missing_uri_raises(self):
        uris = ["s3://%s/%s" % (_BUCKET, key) for key, _, _ in _SPECS]
        uris[2] = None
        dataset = self._make_dataset(uris=uris)

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("media_field", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_bucket_mismatch_raises(self):
        uris = ["s3://%s/%s" % (_BUCKET, key) for key, _, _ in _SPECS]
        uris[3] = "s3://other-bucket/images/charlie.jpg"
        dataset = self._make_dataset(uris=uris)

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("other-bucket", str(cm.exception))
        self.assertIn(_BUCKET, str(cm.exception))
        self.assertEqual(
            len(self.server.get_calls("GET", r"/api/cloudstorages/7")), 1
        )
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_unknown_cloud_storage_raises(self):
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset, cloud_storage_id=99)

        self.assertIn("99", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_duplicate_uri_raises(self):
        uris = ["s3://%s/%s" % (_BUCKET, key) for key, _, _ in _SPECS]
        uris[4] = uris[0]
        dataset = self._make_dataset(uris=uris)

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn(uris[0], str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_old_server_raises(self):
        # CVAT < 2.4.5 requires manifests to read from cloud storages
        self.server.version = "2.4.4"
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("2.4.5", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_video_collection_raises(self):
        dataset = fo.Dataset()
        self._datasets.append(dataset)
        dataset.add_sample(
            fo.Sample(
                filepath=os.path.join(self._tmp_dir, "video.mp4"),
                s3_uri="s3://%s/videos/video.mp4" % _BUCKET,
            )
        )

        with self.assertRaises(ValueError) as cm:
            self._annotate(
                dataset,
                label_field="frames.detections",
                label_type="detections",
                classes=["cat"],
            )

        self.assertIn("image collections", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_non_canonical_key_raises(self):
        uris = ["s3://%s/%s" % (_BUCKET, key) for key, _, _ in _SPECS]
        uris[2] = "s3://%s/images//echo.jpg" % _BUCKET
        dataset = self._make_dataset(uris=uris)

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("images//echo.jpg", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_missing_metadata_raises(self):
        dataset = self._make_dataset()
        sample = dataset.first()
        sample.filepath = os.path.join(self._tmp_dir, "not-cached.jpg")
        sample.metadata = None
        sample.save()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("sync-s3-media", str(cm.exception))
        self.assertEqual(self.server.get_calls("POST", r"/api/tasks"), [])

    def test_stale_metadata_deletes_task(self):
        # The S3 object changed since it was cached, so its labels would be
        # rescaled if they were uploaded
        self.server.frame_sizes["images/bravo.jpg"] = (240, 180)
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("images/bravo.jpg", str(cm.exception))
        self.assertIn("stale", str(cm.exception))
        self.assertIn("--validate etag", str(cm.exception))
        self.assertEqual(self.server.tasks, {})
        self.assertEqual(
            self.server.get_calls("PUT", r"/api/tasks/\d+/annotations"), []
        )

    def test_failure_deletes_all_created_tasks(self):
        # The second of three tasks fails, so the first is deleted too
        self.server.missing_keys.add("images/charlie.jpg")
        dataset = self._make_dataset()

        with self.assertRaises(ValueError):
            self._annotate(dataset, task_size=2)

        self.assertEqual(len(self.server.get_calls("POST", r"/api/tasks")), 2)
        self.assertEqual(self.server.tasks, {})

    def test_old_server_file_limit(self):
        # CVAT 2.4.5-2.4.6 accept at most 500 cloud storage files per task
        self.server.version = "2.4.6"
        path, size = (
            self._paths["images/delta.jpg"],
            _SIZES["images/delta.jpg"],
        )

        dataset = fo.Dataset()
        self._datasets.append(dataset)
        samples = []
        for idx in range(501):
            key = "many/%04d.jpg" % idx
            self.server.frame_sizes[key] = size
            samples.append(
                fo.Sample(filepath=path, s3_uri="s3://%s/%s" % (_BUCKET, key))
            )

        dataset.add_samples(samples)

        results = self._annotate(
            dataset,
            label_field="new_field",
            label_type="detections",
            classes=["cat"],
        )

        self.assertEqual(len(results.task_ids), 2)
        num_files = [
            len(self.server.data_request(t)["body"]["server_files"])
            for t in results.task_ids
        ]
        self.assertEqual(num_files, [500, 1])

    def test_string_cloud_storage_id(self):
        dataset = self._make_dataset()

        results = self._annotate(
            dataset, cloud_storage_id=str(_CLOUD_STORAGE_ID)
        )

        task_id = results.task_ids[0]
        body = self.server.data_request(task_id)["body"]
        self.assertEqual(body["cloud_storage_id"], _CLOUD_STORAGE_ID)

    def test_unmatched_frame_name_deletes_task(self):
        self.server.frame_names["images/bravo.jpg"] = "unrelated/zzz.jpg"
        self.server.frame_sizes["unrelated/zzz.jpg"] = _SIZES[
            "images/bravo.jpg"
        ]
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("unrelated/zzz.jpg", str(cm.exception))
        self.assertEqual(
            len(self.server.get_calls("DELETE", r"/api/tasks/1")), 1
        )
        self.assertEqual(self.server.tasks, {})

        # No labels were uploaded to the task
        self.assertEqual(
            self.server.get_calls("PUT", r"/api/tasks/\d+/annotations"), []
        )

    def test_missing_object_deletes_task(self):
        self.server.missing_keys.add("images/echo.jpg")
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn("images/echo.jpg", str(cm.exception))
        self.assertEqual(
            len(self.server.get_calls("DELETE", r"/api/tasks/1")), 1
        )
        self.assertEqual(self.server.tasks, {})

    def test_failed_data_request_raises(self):
        message = "Failed to read images/alpha.jpg: 403 Forbidden"
        self.server.data_error = message
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn(message, str(cm.exception))
        self.assertLessEqual(self.num_sleeps, 1)
        self.assertGreater(
            len(self.server.get_calls("GET", r"/api/requests/.+")), 0
        )

        # The task whose data failed is deleted
        self.assertEqual(self.server.tasks, {})

    def test_failed_data_request_with_query_like_id_raises(self):
        # Newer CVAT servers use request IDs like "action=create&..."
        message = "Failed to read images/alpha.jpg: 403 Forbidden"
        self.server.data_error = message
        self.server.rq_id_template = "action=create&target=task&target_id=%d"
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn(message, str(cm.exception))
        self.assertLessEqual(self.num_sleeps, 1)
        self.assertGreater(
            len(self.server.get_calls("GET", r"/api/requests/action=.+")), 0
        )

    def test_failed_legacy_status_raises(self):
        message = "Failed to read images/alpha.jpg: 403 Forbidden"
        self.server.data_error = message
        self.server.legacy_status = True
        dataset = self._make_dataset()

        with self.assertRaises(ValueError) as cm:
            self._annotate(dataset)

        self.assertIn(message, str(cm.exception))
        self.assertLessEqual(self.num_sleeps, 1)
        self.assertGreater(
            len(self.server.get_calls("GET", r"/api/tasks/\d+/status")), 0
        )

    def test_non_cloud_upload_unchanged(self):
        self.server.job_delay = 2
        dataset = self._make_dataset()
        sample_ids = dataset.values("id")
        before = self._snapshot(dataset)

        results = dataset.annotate(
            "local",
            backend="cvat",
            url=_URL,
            username="u",
            password="p",
            label_field="ground_truth",
        )

        self.assertEqual(self.server.unhandled, [])
        task_id = results.task_ids[0]

        # The media was uploaded as multipart client files in sample order
        request = self.server.data_request(task_id)
        self.assertTrue(
            request["content_type"].startswith("multipart/form-data")
        )
        body = request["body"]
        filenames = [os.path.basename(key) for key, _, _ in _SPECS]
        self.assertEqual(
            sorted(body["files"].keys()),
            sorted("client_files[%d]" % i for i in range(len(_SPECS))),
        )
        self.assertEqual(
            [body["files"]["client_files[%d]" % i][0] for i in range(5)],
            filenames,
        )
        for key, _, _ in _SPECS:
            idx = filenames.index(os.path.basename(key))
            with open(self._paths[key], "rb") as f:
                self.assertEqual(
                    body["files"]["client_files[%d]" % idx][1], f.read()
                )

        self.assertEqual(body["fields"]["sorting_method"], "predefined")
        self.assertEqual(body["fields"]["image_quality"], "75")
        self.assertNotIn("server_files", body["fields"])
        self.assertNotIn("cloud_storage_id", body["fields"])

        # No cloud storage requests were made
        self.assertEqual(
            self.server.get_calls("GET", r"/api/cloudstorages/.+"), []
        )
        self.assertEqual(self.server.get_calls("GET", r"/api/requests/.+"), [])
        self.assertEqual(
            self.server.get_calls("GET", r"/api/tasks/\d+/status"), []
        )

        # Frames are mapped positionally, exactly as before
        self.assertEqual(
            results.frame_id_map,
            {
                task_id: {
                    idx: {"sample_id": sample_id}
                    for idx, sample_id in enumerate(sample_ids)
                }
            },
        )
        self.assertEqual(self.server.frame_names_of(task_id), filenames)

        put = self.server.tasks[task_id]["put_requests"][-1]
        frames_by_label_id = {
            self.server.fo_label_id(s): s["frame"] for s in put["shapes"]
        }
        for idx, sample in enumerate(dataset):
            if sample["ground_truth"] is None:
                continue

            for det in sample["ground_truth"].detections:
                self.assertEqual(frames_by_label_id[det.id], idx)

        # Edit a label and load it back
        shape = self.server.shapes_on(task_id, "charlie.jpg")[0]
        self.server.edit_shape(task_id, shape["id"], points=[0, 0, 20, 60])

        dataset.load_annotations("local", url=_URL, username="u", password="p")

        after = self._snapshot(dataset)
        self._assert_boxes_equal(
            after["images/charlie.jpg"][0][2], [0.0, 0.0, 0.5, 0.5]
        )
        self._assert_snapshots_equal(
            after, before, skip=("images/charlie.jpg",)
        )


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)
