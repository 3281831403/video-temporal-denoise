"""Temporal denoising primitives for grayscale video frames.

A frame is a rectangular ``list[list[float]]`` of grayscale values in the
0..255 range. Functions return new matrices and never mutate their inputs.
All floating point comparisons in the test suite use an absolute tolerance of
1e-6. Invalid shapes or configuration values raise :class:`DenoiseError`.
"""


class DenoiseError(ValueError):
    """Raised when a frame, mask, weight, or configuration is invalid."""


def frame_diff(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Return the per-pixel absolute difference between two frames.

    Both frames must be non-empty rectangular matrices with identical
    dimensions. The result has the same shape and each cell is
    ``abs(a[y][x] - b[y][x])``; negative differences are never clamped to
    zero. Raise :class:`DenoiseError` when either frame is ragged/empty or the
    two dimensions differ. Inputs are not modified.
    """
    raise NotImplementedError


def motion_mask(
    diff: list[list[float]], threshold: float, dilate: int
) -> list[list[bool]]:
    """Mark the moving pixels in a frame-difference matrix.

    ``diff`` is a non-negative rectangular matrix, typically the result of
    :func:`frame_diff`. A pixel starts as motion when
    ``diff[y][x] > threshold`` (threshold is in the same 0..255 units as the
    difference values). ``dilate`` is a non-negative integer radius: every
    initial motion pixel expands over its square Chebyshev neighbourhood
    (radius 1 is the 8-neighbourhood), so object edges keep a margin.
    ``dilate=0`` performs thresholding only. Return a bool matrix with the
    same shape as ``diff``. Raise :class:`DenoiseError` for a negative or
    non-integer radius, negative threshold, or invalid/ragged input.
    """
    raise NotImplementedError


def blend_temporal(
    cur: list[list[float]],
    prev: list[list[float]],
    alpha_map: list[list[float]],
) -> list[list[float]]:
    """Blend a current frame with a previous frame pixel by pixel.

    Every cell is computed as
    ``(1 - w) * cur[y][x] + w * prev[y][x]`` with ``w`` taken from
    ``alpha_map``. Weights lie in 0..1: ``0`` keeps the current pixel and
    ``1`` copies the previous pixel, so a motion pixel with weight 0 has no
    trailing ghost. All three matrices must have the same non-empty
    rectangular shape; a weight outside 0..1 also raises
    :class:`DenoiseError`. Inputs are not modified.
    """
    raise NotImplementedError


def accumulate(
    history: list[list[list[float]]],
    frame: list[list[float]],
    max_history: int,
) -> list[list[list[float]]]:
    """Append a frame to a sliding temporal history window.

    ``history`` is chronological (oldest first). Return a new list with
    ``frame`` appended, keeping only the newest ``max_history`` frames. The
    returned length is therefore ``min(len(history) + 1, max_history)``, so
    memory cannot grow without bound. ``max_history`` must be a positive
    integer; 0 or a negative value raises :class:`DenoiseError`. Every stored
    frame must share the same non-empty rectangular shape as ``frame``;
    ragged or mismatched frames also raise :class:`DenoiseError`. Inputs are
    not mutated.
    """
    raise NotImplementedError


def denoise(
    cur: list[list[float]],
    history: list[list[list[float]]],
    config: dict[str, object],
) -> list[list[float]]:
    """Temporally denoise one frame against a chronological history window.

    ``config`` contains:

    * ``threshold`` (float, 0..255): difference above which a pixel moves;
    * ``dilate`` (int, >= 0): dilation radius passed to :func:`motion_mask`;
    * ``max_history`` (int, >= 1): number of recent history frames used;
    * ``alpha`` (float, 0..1): base temporal weight for still pixels.

    With no history, return a copy of ``cur``. Never synthesize an all-zero
    previous frame, which would darken the first frame. Otherwise:

    1. Build ``mask = motion_mask(frame_diff(cur, history[-1]), threshold,
       dilate)``.
    2. Compute a per-pixel temporal mean from the last ``max_history``
       history frames plus ``cur``.
    3. For still pixels use the brightness-adaptive weight
       ``w = alpha * (0.5 + 0.5 * clamp(cur / 255, 0, 1))`` and output
       ``(1 - w) * cur + w * temporal_mean``. Motion pixels keep ``cur``
       unchanged. Darker (noisier) pixels therefore receive a lower weight,
       limiting over-smoothing in shadows while avoiding motion trails.

    Validate the configuration first with :func:`validate_config` and raise
    :class:`DenoiseError` if it reports a problem (``dilate`` must also be a
    non-negative integer). Return a new frame; inputs are not modified.
    """
    raise NotImplementedError


def ghost_score(
    cur: list[list[float]],
    prev: list[list[float]],
    mask: list[list[bool]],
) -> float:
    """Return a local ghosting score in 0..1; lower is better.

    For every pixel marked ``True`` in ``mask`` compute
    ``abs(cur[y][x] - prev[y][x]) / 255``. The score is the maximum of those
    values, clamped to 0..1; a mask with no ``True`` pixel scores 0.0. Taking
    the maximum over the motion region prevents a small local ghost from
    being diluted by the rest of the frame. Frames and mask must share the
    same non-empty rectangular shape; otherwise raise
    :class:`DenoiseError`.
    """
    raise NotImplementedError


def validate_config(config: dict[str, object]) -> list[str]:
    """Return human-readable configuration problems; ``[]`` means valid.

    A valid configuration has ``threshold`` in 0..255, ``max_history`` as a
    positive integer (0 is invalid), and ``alpha`` in 0..1. Missing,
    non-numeric, or out-of-range values produce one problem string each; the
    string names the offending key (for example ``"alpha"``). This function
    reports problems instead of raising for bad values.
    """
    raise NotImplementedError


def denoise_batch(
    frames: list[list[list[float]]], config: dict[str, object]
) -> dict[str, object]:
    """Denoise a flat chronological batch and return a result mapping.

    The result is ``{"frames": list[list[list[float]]], "count": int}``.
    Every call starts with its own empty history: history is never shared
    between calls or shots, so one clip cannot bleed into the next. Frames
    are processed in input order, and each output comes from :func:`denoise`
    using the history accumulated from earlier frames in this same call.

    Validate the configuration and every frame before producing output. Any
    frame that is not a non-empty rectangular matrix, or that differs in shape
    from the first frame, raises :class:`DenoiseError` with no partial result
    (whole-batch rollback). An empty input returns
    ``{"frames": [], "count": 0}``. Inputs are not mutated.
    """
    raise NotImplementedError
