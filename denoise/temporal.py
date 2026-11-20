"""Temporal denoising primitives for grayscale video frames.

A frame is a rectangular ``list[list[float]]`` of grayscale values in the
0..255 range. Functions return new matrices and never mutate their inputs.
All floating point comparisons in the test suite use an absolute tolerance of
1e-6. Invalid shapes or configuration values raise :class:`DenoiseError`.
"""


class DenoiseError(ValueError):
    """Raised when a frame, mask, weight, or configuration is invalid."""


def _is_number(value: object) -> bool:
    """Return True for real int/float numbers (bool is not a number here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_frame(
    frame: object, name: str = "frame", non_negative: bool = False
) -> tuple[int, int]:
    """Validate a non-empty rectangular numeric matrix; return (h, w)."""
    if not isinstance(frame, list) or len(frame) == 0:
        raise DenoiseError(f"{name} must be a non-empty matrix")
    width: int | None = None
    for row in frame:
        if not isinstance(row, list) or len(row) == 0:
            raise DenoiseError(f"{name} must be a non-empty rectangular matrix")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise DenoiseError(f"{name} must be rectangular (ragged rows)")
        for value in row:
            if not _is_number(value):
                raise DenoiseError(f"{name} contains a non-numeric pixel")
            if non_negative and value < 0:
                raise DenoiseError(f"{name} contains a negative pixel")
    assert width is not None
    return len(frame), width


def _validate_same_shape(
    a: list, b: list, name_a: str, name_b: str
) -> tuple[int, int]:
    shape_a = _validate_frame(a, name_a)
    shape_b = _validate_frame(b, name_b)
    if shape_a != shape_b:
        raise DenoiseError(
            f"{name_a} and {name_b} must have identical dimensions: "
            f"{shape_a} != {shape_b}"
        )
    return shape_a


def _validate_mask(mask: object, height: int, width: int) -> None:
    """Validate a bool matrix matching the given shape."""
    if not isinstance(mask, list) or len(mask) != height:
        raise DenoiseError("mask must match the frame dimensions")
    for row in mask:
        if not isinstance(row, list) or len(row) != width:
            raise DenoiseError("mask must be a rectangular bool matrix")
        for value in row:
            if not isinstance(value, bool):
                raise DenoiseError("mask must contain only bool values")


def frame_diff(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Return the per-pixel absolute difference between two frames.

    Both frames must be non-empty rectangular matrices with identical
    dimensions. The result has the same shape and each cell is
    ``abs(a[y][x] - b[y][x])``; negative differences are never clamped to
    zero. Raise :class:`DenoiseError` when either frame is ragged/empty or the
    two dimensions differ. Inputs are not modified.
    """
    height, width = _validate_same_shape(a, b, "frame a", "frame b")
    return [
        [abs(a[y][x] - b[y][x]) for x in range(width)] for y in range(height)
    ]


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
    height, width = _validate_frame(diff, "diff", non_negative=True)
    if not _is_number(threshold) or threshold < 0:
        raise DenoiseError("threshold must be a non-negative number")
    if not isinstance(dilate, int) or isinstance(dilate, bool) or dilate < 0:
        raise DenoiseError("dilate must be a non-negative integer radius")

    seeds = [
        [diff[y][x] > threshold for x in range(width)] for y in range(height)
    ]
    if dilate == 0:
        return seeds

    mask: list[list[bool]] = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            y0 = max(0, y - dilate)
            y1 = min(height, y + dilate + 1)
            x0 = max(0, x - dilate)
            x1 = min(width, x + dilate + 1)
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    if seeds[ny][nx]:
                        mask[y][x] = True
                        break
                if mask[y][x]:
                    break
    return mask


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
    height, width = _validate_same_shape(cur, prev, "cur", "prev")
    alpha_shape = _validate_frame(alpha_map, "alpha_map")
    if alpha_shape != (height, width):
        raise DenoiseError("alpha_map must match the frame dimensions")
    for row in alpha_map:
        for weight in row:
            if weight < 0 or weight > 1:
                raise DenoiseError("blend weights must lie in 0..1")

    return [
        [
            (1.0 - alpha_map[y][x]) * cur[y][x]
            + alpha_map[y][x] * prev[y][x]
            for x in range(width)
        ]
        for y in range(height)
    ]


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
    if (
        not isinstance(max_history, int)
        or isinstance(max_history, bool)
        or max_history <= 0
    ):
        raise DenoiseError("max_history must be a positive integer")
    if not isinstance(history, list):
        raise DenoiseError("history must be a list of frames")

    shape = _validate_frame(frame, "frame")
    for index, old in enumerate(history):
        if _validate_frame(old, f"history[{index}]") != shape:
            raise DenoiseError(
                "all history frames must share the frame's dimensions"
            )

    window = list(history)
    window.append(frame)
    if len(window) > max_history:
        window = window[-max_history:]
    return window


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
    problems = validate_config(config)
    if problems:
        raise DenoiseError("invalid configuration: " + "; ".join(problems))
    dilate = config["dilate"]
    if not isinstance(dilate, int) or isinstance(dilate, bool) or dilate < 0:
        raise DenoiseError("dilate must be a non-negative integer")
    if not isinstance(history, list):
        raise DenoiseError("history must be a list of frames")

    height, width = _validate_frame(cur, "cur")
    for index, old in enumerate(history):
        if _validate_frame(old, f"history[{index}]") != (height, width):
            raise DenoiseError(
                "all history frames must share the current frame's dimensions"
            )

    # First frame: no temporal information, never blend against a zero frame.
    if not history:
        return [row[:] for row in cur]

    threshold = config["threshold"]
    alpha = config["alpha"]
    max_history = config["max_history"]

    diff = frame_diff(cur, history[-1])
    mask = motion_mask(diff, threshold, dilate)

    window = history[-max_history:] + [cur]
    span = len(window)
    temporal_mean = [
        [
            sum(frame[y][x] for frame in window) / span
            for x in range(width)
        ]
        for y in range(height)
    ]

    result: list[list[float]] = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if mask[y][x]:
                # Motion region: zero temporal weight, no ghosting trail.
                result[y][x] = cur[y][x]
            else:
                brightness = _clamp(cur[y][x] / 255.0, 0.0, 1.0)
                weight = alpha * (0.5 + 0.5 * brightness)
                result[y][x] = (
                    (1.0 - weight) * cur[y][x]
                    + weight * temporal_mean[y][x]
                )
    return result


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
    height, width = _validate_same_shape(cur, prev, "cur", "prev")
    _validate_mask(mask, height, width)

    worst = 0.0
    for y in range(height):
        for x in range(width):
            if mask[y][x]:
                local = abs(cur[y][x] - prev[y][x]) / 255.0
                if local > worst:
                    worst = local
    return _clamp(worst, 0.0, 1.0)


def validate_config(config: dict[str, object]) -> list[str]:
    """Return human-readable configuration problems; ``[]`` means valid.

    A valid configuration has ``threshold`` in 0..255, ``max_history`` as a
    positive integer (0 is invalid), and ``alpha`` in 0..1. Missing,
    non-numeric, or out-of-range values produce one problem string each; the
    string names the offending key (for example ``"alpha"``). This function
    reports problems instead of raising for bad values.
    """
    if not isinstance(config, dict):
        return ["config must be a mapping"]

    problems: list[str] = []

    threshold = config.get("threshold")
    if not _is_number(threshold) or not (0 <= threshold <= 255):
        problems.append("threshold must be a number in 0..255")

    max_history = config.get("max_history")
    if (
        not isinstance(max_history, int)
        or isinstance(max_history, bool)
        or max_history <= 0
    ):
        problems.append("max_history must be a positive integer")

    alpha = config.get("alpha")
    if not _is_number(alpha) or not (0 <= alpha <= 1):
        problems.append("alpha must be a number in 0..1")

    return problems


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
    problems = validate_config(config)
    if problems:
        raise DenoiseError("invalid configuration: " + "; ".join(problems))

    if not isinstance(frames, list):
        raise DenoiseError("frames must be a list of frames")
    if len(frames) == 0:
        return {"frames": [], "count": 0}

    # Validate the whole batch up front so a failure rolls back everything.
    batch_shape: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        shape = _validate_frame(frame, f"frames[{index}]")
        if batch_shape is None:
            batch_shape = shape
        elif shape != batch_shape:
            raise DenoiseError(
                "all frames in a batch must share the same dimensions"
            )

    max_history = config["max_history"]
    history: list[list[list[float]]] = []
    outputs: list[list[list[float]]] = []
    for frame in frames:
        outputs.append(denoise(frame, history, config))
        history = accumulate(history, frame, max_history)

    return {"frames": outputs, "count": len(outputs)}
