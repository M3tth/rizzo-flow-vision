"""Minimal ctypes binding for llama.cpp libmtmd, pinned to the same release as libllama.

The binding intentionally exposes only the operations ValoraAI needs: load one vision
projector, turn in-memory images into multimodal chunks, evaluate those chunks into an existing
llama_context, and report the resulting logical position. No sampling or text generation.
"""

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_void_p,
)
from pathlib import Path


from . import llama_release



class MtmdContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("device", c_void_p),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("image_marker", c_char_p),
        ("media_marker", c_char_p),
        ("flash_attn_type", c_int),
        ("warmup", c_bool),
        ("image_min_tokens", c_int),
        ("image_max_tokens", c_int),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("batch_max_tokens", c_int32),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
    ]


class MtmdHelperVideoInitParams(ctypes.Structure):
    _fields_ = [
        ("fps_target", c_float),
        ("ffmpeg_bin_dir", c_char_p),
        ("timestamp_interval_ms", c_int64),
    ]


class MtmdHelperInitOpt(ctypes.Structure):
    _fields_ = [("video_params", MtmdHelperVideoInitParams)]


class MtmdHelperBitmapWrapper(ctypes.Structure):
    _fields_ = [("bitmap", c_void_p), ("video_ctx", c_void_p)]


class MtmdInputText(ctypes.Structure):
    _fields_ = [
        ("text", c_char_p),
        ("text_len", c_size_t),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]


SIGNATURES = {
    "mtmd_context_params_default": (MtmdContextParams, []),
    "mtmd_init_from_file": (c_void_p, [c_char_p, c_void_p, MtmdContextParams]),
    "mtmd_free": (None, [c_void_p]),
    "mtmd_support_vision": (c_bool, [c_void_p]),
    "mtmd_decode_use_mrope": (c_bool, [c_void_p]),
    "mtmd_get_marker": (c_char_p, [c_void_p]),
    "mtmd_helper_init_opt_default": (MtmdHelperInitOpt, []),
    "mtmd_helper_bitmap_init_from_buf": (
        MtmdHelperBitmapWrapper,
        [c_void_p, POINTER(ctypes.c_ubyte), c_size_t, c_bool, MtmdHelperInitOpt],
    ),
    "mtmd_bitmap_free": (None, [c_void_p]),
    "mtmd_input_chunks_init": (c_void_p, []),
    "mtmd_input_chunks_free": (None, [c_void_p]),
    "mtmd_tokenize": (
        c_int32,
        [c_void_p, c_void_p, POINTER(MtmdInputText), POINTER(c_void_p), c_size_t],
    ),
    "mtmd_helper_get_n_tokens": (c_size_t, [c_void_p]),
    "mtmd_helper_get_n_pos": (c_int32, [c_void_p]),
    "mtmd_helper_eval_chunks": (
        c_int32,
        [c_void_p, c_void_p, c_void_p, c_int32, c_int32, c_int32, c_bool, POINTER(c_int32)],
    ),
}


def _shared(directory: Path, stem: str) -> Path | None:
    if sys.platform == "win32":
        names = [f"{stem}.dll"]
    elif sys.platform == "darwin":
        names = [f"lib{stem}.dylib"]
    else:
        names = [f"lib{stem}.so"]
    return next((directory / name for name in names if (directory / name).is_file()), None)


class MtmdLibrary:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        path = _shared(self.directory, "mtmd")
        if path is None:
            raise ValueError(
                f"libmtmd is missing from {self.directory}; install llama.cpp "
                f"{llama_release.RELEASE} with multimodal support"
            )
        if sys.platform == "win32":
            os.add_dll_directory(str(self.directory))
        self.handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        for name, (result, arguments) in SIGNATURES.items():
            function = getattr(self.handle, name, None)
            if function is None:
                raise ValueError(
                    f"{path}: symbol {name} is missing; expected llama.cpp "
                    f"{llama_release.RELEASE} ({llama_release.COMMIT[:7]})"
                )
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)


class VisionContext:
    """Vision projector attached to an already-loaded text model/context."""

    def __init__(self, session, library: MtmdLibrary, context, mmproj: Path):
        self.session = session
        self.library = library
        self.context = context
        self.mmproj = Path(mmproj)
        marker = library.mtmd_get_marker(context)
        self.marker = marker.decode("utf-8") if marker else "<__media__>"
        self.uses_mrope = bool(library.mtmd_decode_use_mrope(context))

    @classmethod
    def load(cls, session, mmproj: Path, threads: int | None = None) -> "VisionContext":
        mmproj = Path(mmproj).resolve()
        if not mmproj.is_file():
            raise ValueError(f"Vision projector not found at {mmproj}")
        library = MtmdLibrary(session.library.directory)
        params = library.mtmd_context_params_default()
        params.use_gpu = session.device is not None
        params.device = session.device.handle if session.device else None
        params.print_timings = False
        params.warmup = False
        if threads:
            params.n_threads = threads
        context = library.mtmd_init_from_file(str(mmproj).encode(), session.model, params)
        if not context:
            raise ValueError(f"libmtmd cannot load vision projector {mmproj}")
        if not library.mtmd_support_vision(context):
            library.mtmd_free(context)
            raise ValueError(f"{mmproj} does not provide vision input")
        return cls(session, library, context, mmproj)

    def close(self):
        if self.context:
            self.library.mtmd_free(self.context)
            self.context = None

    def _chunks(self, text: str, images: list[bytes]):
        raw_text = text.encode("utf-8")
        if text.count(self.marker) != len(images):
            raise ValueError(
                f"Multimodal prompt contains {text.count(self.marker)} media markers "
                f"but {len(images)} images were supplied"
            )
        chunks = self.library.mtmd_input_chunks_init()
        if not chunks:
            raise ValueError("libmtmd could not allocate input chunks")
        bitmaps = []
        backing = []
        try:
            options = self.library.mtmd_helper_init_opt_default()
            for index, raw in enumerate(images):
                buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
                wrapper = self.library.mtmd_helper_bitmap_init_from_buf(
                    self.context, buffer, len(raw), False, options
                )
                if not wrapper.bitmap:
                    raise ValueError(f"libmtmd could not decode image {index + 1}")
                # The request schema accepts still images only, so video_ctx should stay null.
                if wrapper.video_ctx:
                    self.library.mtmd_bitmap_free(wrapper.bitmap)
                    raise ValueError("Video input is not supported by the decisions API")
                bitmaps.append(wrapper.bitmap)
                backing.append(buffer)
            array = (c_void_p * len(bitmaps))(*bitmaps) if bitmaps else None
            input_text = MtmdInputText(raw_text, len(raw_text), False, True)
            status = self.library.mtmd_tokenize(
                self.context,
                chunks,
                ctypes.byref(input_text),
                array,
                len(bitmaps),
            )
            if status != 0:
                raise ValueError(f"mtmd_tokenize returned {status}")
            return chunks, bitmaps, backing
        except Exception:
            self.library.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                self.library.mtmd_bitmap_free(bitmap)
            raise

    def evaluate(
        self,
        text: str,
        images: list[bytes],
        *,
        start: int = 0,
        sequence: int = 0,
        logits_last: bool = False,
    ) -> tuple[int, int, int | None]:
        """Evaluate text/media into the shared llama context.

        Returns (new logical position, multimodal token count, final logits row). For M-RoPE,
        the first value deliberately differs from token count and must be used for the next chunk.
        """

        chunks, bitmaps, _backing = self._chunks(text, images)
        try:
            tokens = int(self.library.mtmd_helper_get_n_tokens(chunks))
            new_position = c_int32(start)
            status = self.library.mtmd_helper_eval_chunks(
                self.context,
                self.session.context,
                chunks,
                start,
                sequence,
                self.session.n_batch,
                logits_last,
                ctypes.byref(new_position),
            )
            if status != 0:
                raise ValueError(f"mtmd_helper_eval_chunks returned {status}")
            # The helper splits text into n_batch-sized llama_decode calls. The logits row is
            # therefore the last row of the final internal batch.
            row = (tokens - 1) % self.session.n_batch if logits_last and tokens else None
            return new_position.value, tokens, row
        finally:
            self.library.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                self.library.mtmd_bitmap_free(bitmap)
