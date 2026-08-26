"""
The defense-oracle / attacker split, guarded at the place it fails silently.

Routing a call to the wrong backend still yields well-formed JSON from a well-trained model -- it
is just no longer the measurement you thought you were making. That is why it gets a test rather
than a comment.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json

import pytest

pytest.importorskip("openai", reason="core dependency; not installed in this environment")

from veillens.reason.oracle import GeoLocationAPI  # noqa: E402
from veillens.vlm.base import VLMError, encode_image_data_url  # noqa: E402
from veillens.vlm.registry import build_client, client_from_config  # noqa: E402

Image = pytest.importorskip("PIL.Image", reason="pillow is a core dependency")


@pytest.fixture
def cityscapes_like(tmp_path):
    """A stand-in for the real dataset: 2048x1024, and a PNG."""
    path = tmp_path / "frame_leftImg8bit.png"
    Image.new("RGB", (2048, 1024), (10, 20, 30)).save(path)
    return path


def test_mime_type_follows_the_file(cityscapes_like, tmp_path):
    assert encode_image_data_url(cityscapes_like).startswith("data:image/png;base64,")

    jpeg = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(jpeg)
    assert encode_image_data_url(jpeg).startswith("data:image/jpeg;base64,")


def test_max_edge_downscales(cityscapes_like):
    url = encode_image_data_url(cityscapes_like, max_edge=1280)
    payload = base64.b64decode(url.split(",", 1)[1])
    assert Image.open(io.BytesIO(payload)).size == (1280, 640)
    assert len(url) < len(encode_image_data_url(cityscapes_like))


def test_registry_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        build_client("gemma_local")


def test_client_from_config_drops_pipeline_only_keys():
    client = client_from_config({
        "backend": "openai",
        "model": "some/model",
        "concurrent_requests": 6,   # the oracle's business, not the client's
    })
    assert client.model == "some/model"


class _FakeClient:
    """Records which backend was asked, and for which schema."""

    def __init__(self, tag):
        self.tag = tag
        self.schemas = []
        self.trace_labels = []

    def encode_image(self, image_path):
        return "data:image/png;base64,AAAA"

    async def infer(self, image_data_url, prompt, schema, schema_name, trace_label=None):
        self.schemas.append(schema_name)
        self.trace_labels.append(trace_label)
        return json.dumps({"who": self.tag})


class _Spec:
    name = "geolocation"

    def get_schema(self, mode):
        return {"type": "object"}

    def get_round1_prompt(self):
        return "round 1"

    def get_eval_prompt(self):
        return "eval"

    def format_round2_prompt(self, predicted_location, region_name):
        return f"round 2 {predicted_location} {region_name}"


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "schema_type,role,expected",
    [
        ("original", "reason", "reason"),
        ("eval", "eval", "eval"),
        # The trap: the compositional spec runs its FINAL eval through the 'original' schema, so
        # a router that inferred the backend from schema_type would bill it to the attacker.
        ("original", "eval", "eval"),
    ],
)
def test_role_routes_independently_of_schema_type(schema_type, role, expected):
    reason, evaluate = _FakeClient("reason"), _FakeClient("eval")
    api = GeoLocationAPI(spec=_Spec(), reason_client=reason, eval_client=evaluate)

    result = _run(api.get_geolocation_prediction("x.png", schema_type=schema_type, role=role))

    assert result["who"] == expected
    assert result["image_path"] == "x.png"


def test_reason_defaults_to_the_eval_backend():
    """A config with no `reasoning:` section must behave exactly as it did before the split."""
    evaluate = _FakeClient("eval")
    api = GeoLocationAPI(spec=_Spec(), eval_client=evaluate)
    assert api.reason_client is evaluate


def test_backend_error_becomes_none_not_an_exception():
    """
    Every call site tests the result for falsiness. A truncated reply is loud in the log but must
    not escape as an exception, or one bad image takes down the run.
    """

    class _Truncating(_FakeClient):
        async def infer(self, **kwargs):
            raise VLMError("hit the token ceiling inside <think>")

    api = GeoLocationAPI(spec=_Spec(), reason_client=_Truncating("boom"),
                         eval_client=_FakeClient("eval"))
    assert _run(api.get_geolocation_prediction("x.png", schema_type="original")) is None


def test_unknown_role_is_rejected():
    api = GeoLocationAPI(spec=_Spec(), eval_client=_FakeClient("eval"))
    with pytest.raises(ValueError, match="Unknown role"):
        _run(api.get_geolocation_prediction("x.png", role="attacker"))
