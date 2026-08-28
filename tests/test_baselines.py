"""The baselines, checked without a GPU, a key, or a dataset.

Three things are worth pinning here, and they are the three that would otherwise fail silently:

  1. **configs/baselines.yaml and the classes agree, both ways.** A typo in a config key does not
     raise -- it lands in `**params` and the method runs with its default, so a reviewer changing
     `blur_ksize` to `blur_size` would get a full run, a full score table, and no indication that
     the knob did nothing.
  2. **The registry is complete and importable** without torch, opencv or an API key, because
     `--dry-run` has to work on a laptop.
  3. **The pure-Python pieces are correct**: the target pairing, the tiling, and the one baseline
     that has no dependencies at all.
"""

import importlib.util
import inspect

import numpy as np
import pytest
import yaml

from veillens import paths
from veillens.baselines import REGISTRY, available, get_baseline_class
from veillens.baselines.base import Baseline
from veillens.baselines.geoshield import _tile_boxes, pair_targets

CONFIG = yaml.safe_load((paths.CONFIGS / "baselines.yaml").read_text())

needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="opencv is not installed (pip install -e '.[baselines]')",
)


# -- registry ---------------------------------------------------------------

def test_every_registered_method_imports_and_is_a_baseline():
    for name in available():
        if name in ("rand-noise", "ocr-blur") and importlib.util.find_spec("cv2") is None:
            continue
        cls = get_baseline_class(name)
        assert issubclass(cls, Baseline), f"{name} is not a Baseline"
        assert cls.name == name, f"{cls.__name__}.name is {cls.name!r}, registered as {name!r}"
        assert cls.summary, f"{name} has no summary; the CLI and the docs table print it"


def test_unknown_method_names_its_alternatives():
    with pytest.raises(ValueError, match="rand-noise"):
        get_baseline_class("random-noise")


# -- config surface ---------------------------------------------------------

def test_config_and_registry_cover_the_same_methods():
    assert sorted(CONFIG) == sorted(REGISTRY), (
        "configs/baselines.yaml and veillens.baselines.REGISTRY disagree about which methods "
        "exist; a config section with no class is dead, a class with no section cannot be run."
    )


@pytest.mark.parametrize("name", sorted(CONFIG))
def test_every_config_key_is_a_real_constructor_parameter(name):
    """A key that lands in **params configures nothing and says nothing about it."""
    if name in ("rand-noise", "ocr-blur") and importlib.util.find_spec("cv2") is None:
        pytest.skip("opencv is not installed")
    accepted = set(inspect.signature(get_baseline_class(name).__init__).parameters) - {"self",
                                                                                       "params"}
    unknown = sorted(set(CONFIG[name]) - accepted)
    assert not unknown, (
        f"configs/baselines.yaml [{name}] sets {unknown}, which {name} does not accept. "
        f"It would be swallowed by **params. Accepted: {sorted(accepted)}"
    )


def test_gpt_image_prompt_resolves_for_every_privacy_class():
    """The one config value that is filled in per run; a missing file would fail mid-run."""
    template = CONFIG["gpt-image"]["prompt"]
    assert "{privacy_class}" in template
    for privacy_class in ("geolocation", "compositional"):
        path = paths.resolve(template.format(privacy_class=privacy_class))
        assert path.is_file(), f"missing image-edit prompt for {privacy_class}: {path}"


def test_the_two_adversarial_budgets_still_match():
    """mm-pgd and geoshield are budget-matched at 8/255 on purpose; unmatching them silently
    invalidates the comparison the paper draws between them."""
    mm_pgd = CONFIG["mm-pgd"]["epsilon"]
    geoshield = CONFIG["geoshield"]["epsilon"] / 255.0
    assert abs(mm_pgd - geoshield) < 1e-3, (
        f"mm-pgd epsilon {mm_pgd} != geoshield epsilon {geoshield} (= "
        f"{CONFIG['geoshield']['epsilon']}/255). See docs/BASELINES.md."
    )


# -- geoshield's pure-python pieces -----------------------------------------

def _items(*ids):
    return [{"image_id": i, "image_path": f"/data/{i}.png"} for i in ids]


def test_pairing_covers_every_image_and_avoids_the_same_city():
    items = _items("aachen_1", "aachen_2", "bonn_1", "cologne_1", "cologne_2")
    pairing = pair_targets(items)
    assert set(pairing) == {item["image_id"] for item in items}
    for item in items:
        source_city = item["image_id"].split("_")[0]
        target_city = pairing[item["image_id"]].split("/")[-1].split("_")[0]
        assert source_city != target_city, f"{item['image_id']} was paired within its own city"


def test_pairing_is_deterministic_and_order_independent():
    items = _items("aachen_1", "bonn_1", "cologne_1")
    assert pair_targets(items) == pair_targets(list(reversed(items)))


def test_pairing_prefers_an_explicit_city_field_over_the_id():
    items = [{"image_id": "one", "image_path": "/a.png", "city": "alpha"},
             {"image_id": "two", "image_path": "/b.png", "city": "beta"}]
    assert pair_targets(items) == {"one": "/b.png", "two": "/a.png"}


@pytest.mark.parametrize("size,expected", [
    ((2048, 1024), [(0, 0, 1024, 1024), (1024, 0, 2048, 1024)]),   # Cityscapes
    ((1024, 1024), [(0, 0, 1024, 1024)]),                          # already square
    ((512, 1536), [(0, 0, 512, 512), (0, 512, 512, 1024), (0, 1024, 512, 1536)]),   # portrait
])
def test_tiling_covers_the_frame_exactly(size, expected):
    assert _tile_boxes(size[0], size[1], 1024) == expected


def test_tiling_leaves_no_gap_for_an_awkward_aspect_ratio():
    width, height = 1900, 1000
    boxes = _tile_boxes(width, height, 1024)
    assert boxes[0][0] == 0 and boxes[-1][2] == width
    for earlier, later in zip(boxes, boxes[1:]):
        assert earlier[2] == later[0], "tiles must abut, or reassembly loses a column"


# -- rand-noise, end to end -------------------------------------------------

@needs_cv2
def test_rand_noise_blurs_the_configured_fraction(tmp_path):
    from PIL import Image

    from veillens.baselines import build

    source = tmp_path / "frame.png"
    rng = np.random.default_rng(0)
    original = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(original).save(source)

    protected, stats = build("rand-noise", CONFIG["rand-noise"]).apply(str(source), "frame")

    assert protected.size == (64, 64)
    assert stats["modified_pixels"] == int(64 * 64 * CONFIG["rand-noise"]["pixel_ratio"])
    # The blur can leave a selected pixel unchanged where the neighbourhood is already flat, so
    # this is an upper bound, not an equality.
    changed = np.any(original != np.asarray(protected), axis=2).sum()
    assert 0 < changed <= stats["modified_pixels"]


@needs_cv2
def test_rand_noise_is_reproducible_per_image(tmp_path):
    from PIL import Image

    from veillens.baselines import build

    source = tmp_path / "frame.png"
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(source)

    method = build("rand-noise", CONFIG["rand-noise"])
    first, _ = method.apply(str(source), "frame")
    again, _ = method.apply(str(source), "frame")
    other_id, _ = method.apply(str(source), "different-frame")

    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other_id)), (
        "the scatter must depend on the image id, or every frame gets the same mask"
    )


# -- prerequisites ----------------------------------------------------------

def test_missing_weights_are_reported_rather_than_discovered_mid_run():
    problems = get_baseline_class("mm-pgd").check_prerequisites(
        {"models": [{"arch": "resnet18", "weight_path": "/nope/resnet18.pth.tar"}]}
    )
    assert problems and "fetch_baseline_weights" in problems[0]


@needs_cv2
def test_ocr_blur_reports_a_missing_east_model():
    problems = get_baseline_class("ocr-blur").check_prerequisites({"east_model": "/nope/east.pb"})
    assert problems and "fetch_baseline_weights" in problems[0]
