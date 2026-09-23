"""Strict public request contract. Model outputs never supply JSON or field names."""

import base64
import binascii
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Number = Annotated[float, Field(allow_inf_nan=False, ge=-1e100, le=1e100)]
MAX_SLOTS = 26  # every candidate, special ones included, is one uppercase answer letter
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 16 * 1024 * 1024


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Policy(Strict):
    allow_abstain: bool = True
    max_unavailable_probability: float = Field(default=0.5, gt=0, le=1)
    min_top_probability: float = Field(default=0, ge=0, le=1)


class BaseQuestion(Strict):
    instructions: Text
    policy: Policy = Field(default_factory=Policy)

    def require_slots(self, entries, reserved=0):
        reserved += self.policy.allow_abstain
        if len(entries) + reserved > MAX_SLOTS:
            raise ValueError(
                f"At most {MAX_SLOTS - reserved} entries fit here: {MAX_SLOTS} answer letters, "
                f"{reserved} reserved for abstention/out-of-range"
            )


class BooleanQuestion(BaseQuestion):
    type: Literal["boolean"]
    true_description: Text = "Yes. The evidence supports an affirmative answer to the question."
    false_description: Text = "No. The evidence supports a negative answer to the question."


class Option(Strict):
    id: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]
    description: Text


class ChoiceQuestion(BaseQuestion):
    type: Literal["choice"]
    options: list[Option] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("Option IDs must be unique")
        self.require_slots(self.options)
        return self


class ScoreQuestion(BaseQuestion):
    type: Literal["score"]
    levels: list[Text] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def distinct_levels(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Levels must have distinct descriptions")
        self.require_slots(self.levels)
        return self


class Anchor(Strict):
    value: Number
    description: Text


class NumericQuestion(BaseQuestion):
    type: Literal["numeric"]
    unit: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    anchors: list[Anchor] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def increasing_anchors(self):
        if any(b.value <= a.value for a, b in zip(self.anchors, self.anchors[1:])):
            raise ValueError("Numeric anchors must be strictly increasing")
        self.require_slots(self.anchors, reserved=2)
        return self


Question = Annotated[
    BooleanQuestion | ChoiceQuestion | ScoreQuestion | NumericQuestion,
    Field(discriminator="type"),
]


class ImageInput(Strict):
    """One image transported inside the JSON request.

    The API deliberately accepts bytes, not URLs or server-side paths: document data never needs
    to leave the caller/server boundary and a request cannot read arbitrary files from the host.
    """

    data_base64: str
    mime_type: Literal["image/jpeg", "image/png", "image/webp", "image/bmp"] | None = None
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=128)] | None = None

    @model_validator(mode="after")
    def valid_image(self):
        # Fast upper bound before allocating the decoded buffer.
        if not self.data_base64 or len(self.data_base64) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 8:
            raise ValueError(f"Image must be non-empty and at most {MAX_IMAGE_BYTES >> 20} MiB")
        try:
            raw = base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Image data_base64 is not valid base64") from error
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image must be non-empty and at most {MAX_IMAGE_BYTES >> 20} MiB")
        return self

    def bytes(self) -> bytes:
        return base64.b64decode(self.data_base64, validate=True)


class Request(Strict):
    state: str | dict[str, JsonValue] | list[JsonValue]
    images: list[ImageInput] = Field(default_factory=list, max_length=MAX_IMAGES)
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    mode: Literal["shared", "direct"] = "shared"

    @model_validator(mode="after")
    def valid_state(self):
        state_empty = not self.state or (isinstance(self.state, str) and not self.state.strip())
        if state_empty and not self.images:
            raise ValueError("State must not be empty unless at least one image is supplied")
        rendered = json.dumps(self.state, ensure_ascii=False, allow_nan=False)
        if len(rendered.encode()) > 256_000:
            raise ValueError("State exceeds 256 KB; no silent truncation")
        if any(not key.strip() or len(key) > 128 for key in self.questions):
            raise ValueError("Question IDs must have 1–128 nonblank characters")
        return self
