"""Unit coverage for the multimodal request/compiler/scoring path; no model or GPU required."""

import base64

import pytest

from rizzo_flow.backend_llama import LlamaBackend
from rizzo_flow.prompts import Compiled, compile_request
from rizzo_flow.schema import Request


class CharTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, **_variables):
        assert not tokenize
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


def request_dict(data="image bytes"):
    return {
        "state": "",
        "images": [
            {
                "data_base64": base64.b64encode(data.encode()).decode(),
                "mime_type": "image/jpeg",
                "label": "Registro pagina 1",
            }
        ],
        "questions": {
            "firma_docente": {
                "type": "boolean",
                "instructions": "Is the teacher signature present?",
            },
            "data": {
                "type": "boolean",
                "instructions": "Is the date field filled in?",
            },
        },
    }


def test_schema_accepts_image_only_evidence_and_rejects_bad_base64():
    request = Request.model_validate(request_dict())
    assert request.state == ""
    assert request.images[0].bytes() == b"image bytes"

    bad = request_dict()
    bad["images"][0]["data_base64"] = "not base64!"
    with pytest.raises(ValueError, match="base64"):
        Request.model_validate(bad)

    without_image = request_dict()
    without_image["images"] = []
    with pytest.raises(ValueError, match="State must not be empty"):
        Request.model_validate(without_image)


def test_compiler_keeps_media_in_one_shared_text_prefix():
    request = Request.model_validate(request_dict())
    prefix, jobs = compile_request(CharTokenizer(), request, 8192, "<__media__>")

    assert isinstance(prefix, str)
    assert prefix.count("<__media__>") == 1
    assert "Registro pagina 1" in prefix
    assert len(jobs) == 2
    assert all(job.suffix_text and "<__media__>" not in job.suffix_text for job in jobs)
    assert "teacher signature" in jobs[0].suffix_text
    assert "date field" in jobs[1].suffix_text


class FakeSession:
    n_batch = 2048
    idle_free = None
    context = object()

    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.append(("clear",))

    def branch(self, source, target):
        self.calls.append(("branch", source, target))

    def drop(self, sequence):
        self.calls.append(("drop", sequence))

    def logits(self, row, slots):
        self.calls.append(("logits", row, tuple(slots)))
        return [float(slot) for slot in slots]

    def synchronize(self):
        self.calls.append(("sync",))

    def free_bytes(self):
        return None


class FakeVision:
    marker = "<__media__>"
    uses_mrope = True

    def __init__(self):
        self.calls = []

    def evaluate(self, text, images, *, start=0, sequence=0, logits_last=False):
        self.calls.append((text, len(images), start, sequence, logits_last))
        if images:
            # Deliberately make position count differ from token count to exercise M-RoPE.
            return start + 17, 64, None
        tokens = len(text)
        return start + tokens, tokens, (tokens - 1) % 2048 if logits_last else None


def make_backend():
    return LlamaBackend(
        FakeSession(),
        CharTokenizer(),
        {"fingerprint": "test"},
        vision=FakeVision(),
        input_ctx=8192,
    )


def jobs():
    return [
        Compiled("a", [1, 2], [65, 66, 67], "a", "\nQuestion A"),
        Compiled("b", [1, 3], [65, 66, 67], "b", "\nQuestion B"),
    ]


def test_shared_vision_prefills_image_once_and_branches_questions():
    backend = make_backend()
    logits, timing = backend.score_vision("PREFIX<__media__>", jobs(), [b"jpeg"], "shared")

    vision = backend.vision.calls
    assert vision[0] == ("PREFIX<__media__>", 1, 0, 0, False)
    assert [call[1] for call in vision] == [1, 0, 0]
    assert [call[2] for call in vision[1:]] == [17, 17]
    assert [call[3] for call in vision[1:]] == [1, 1]
    assert [call[4] for call in vision[1:]] == [True, True]
    assert [call[0] for call in backend.session.calls].count("branch") == 2
    assert logits["a"] == [65.0, 66.0, 67.0]
    assert timing["shared_prefix_tokens"] == 64
    assert timing["generated_tokens"] == 0
    assert timing["image_count"] == 1


def test_direct_vision_recomputes_image_prefix_for_each_question():
    backend = make_backend()
    backend.score_vision("PREFIX<__media__>", jobs(), [b"jpeg"], "direct")

    prefix_calls = [call for call in backend.vision.calls if call[1] == 1]
    suffix_calls = [call for call in backend.vision.calls if call[1] == 0]
    assert len(prefix_calls) == 2
    assert len(suffix_calls) == 2
    assert all(call[2] == 17 and call[4] for call in suffix_calls)
    assert not any(call[0] == "branch" for call in backend.session.calls)
