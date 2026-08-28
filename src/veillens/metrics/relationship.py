"""
The PISC relationship taxonomy, and how a pair prediction is scored against it.

Two label spaces, and the paper reports both (Sec. 3.2). The six-way space is what PISC annotates;
the three-way space groups it into intimate / non-intimate / no relationship. Every three-way
number is *derived* from the same six-way predictions rather than produced by a separate run, so
there is one place -- :data:`THREE_WAY` -- where the grouping is written down.

Scoring is deliberately three-numbered: ``correct``, ``answered``, ``total``. A pair the model
declined to answer is not correct, but it is also not the same thing as a wrong answer, and
collapsing the two picks a denominator by accident. Both denominators are reported downstream.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional

__all__ = [
    "LABELS",
    "LABEL_TO_INT",
    "THREE_WAY",
    "THREE_WAY_NAMES",
    "to_int",
    "to_three_way",
    "PairScore",
    "pair_accuracy",
]

#: PISC's six-way relationship labels, by their integer code in ``relationship.json``.
LABELS: Dict[int, str] = {
    1: "friends",
    2: "family-members",
    3: "couple",
    4: "professional",
    5: "commercial",
    6: "no-relationship",
}

LABEL_TO_INT: Dict[str, int] = {name: code for code, name in LABELS.items()}

#: The standard PISC coarse grouping. friends/family/couple are intimate; professional/commercial
#: are non-intimate; no-relationship stands alone.
THREE_WAY: Dict[int, int] = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}

THREE_WAY_NAMES: Dict[int, str] = {1: "intimate", 2: "non-intimate", 3: "no-relation"}


def to_int(label: Any) -> Optional[int]:
    """
    Normalise a label to its six-way integer code, or ``None`` if it is not one.

    Accepts the integer itself or the name the schema's enum uses. ``bool`` is rejected explicitly:
    it is an ``int`` subclass, so ``True`` would otherwise silently become ``friends``.
    """
    if isinstance(label, bool):
        return None
    if isinstance(label, int):
        return label if label in LABELS else None
    if isinstance(label, str):
        text = label.strip().lower()
        if text in LABEL_TO_INT:
            return LABEL_TO_INT[text]
        # Tolerate the spellings a model reaches for when it paraphrases the enum.
        text = text.replace("_", "-").replace(" ", "-")
        if text in LABEL_TO_INT:
            return LABEL_TO_INT[text]
        if text.isdigit():
            code = int(text)
            return code if code in LABELS else None
    return None


def to_three_way(label: Any) -> Optional[int]:
    """The coarse code for a six-way label, or ``None`` if the label is unreadable."""
    code = to_int(label)
    return THREE_WAY.get(code) if code is not None else None


class PairScore(NamedTuple):
    """How many of an image's annotated pairs the model got right, and how many it answered."""

    correct: int
    answered: int
    total: int

    @property
    def error_rate(self) -> float:
        """
        Share of annotated pairs *not* correctly recovered, over every annotated pair.

        An unanswered pair counts against the model here, which is the conservative reading when
        this measures a defense: silence is not protection, and reporting it as such is how a
        model that returned nothing scores as a perfect defense. ``answered`` is carried alongside
        so the other denominator stays available.
        """
        if self.total == 0:
            return 0.0
        return (self.total - self.correct) / self.total


def pair_accuracy(ground_truth: Mapping[str, Any],
                  predictions: Mapping[str, Any],
                  *, three_way: bool = False) -> PairScore:
    """
    Score predicted relationships against the PISC labels for one image.

    Args:
        ground_truth: ``{"i j": label}`` for every annotated pair, as PISC stores it.
        predictions: the same shape, keyed identically. Pairs absent from it are unanswered.
        three_way: score in the coarse label space instead of the six-way one.

    Returns:
        A :class:`PairScore`. Only pairs present in *ground_truth* are counted -- a prediction for
        a pair the dataset does not annotate is neither right nor wrong, it is out of scope.
    """
    project = to_three_way if three_way else to_int

    correct = answered = 0
    for pair_key, truth in ground_truth.items():
        expected = project(truth)
        predicted = project(predictions.get(pair_key)) if pair_key in predictions else None
        if predicted is None:
            continue
        answered += 1
        if expected is not None and predicted == expected:
            correct += 1

    return PairScore(correct=correct, answered=answered, total=len(ground_truth))
