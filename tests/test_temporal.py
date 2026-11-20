"""video-temporal-denoise 的红色基线测试。

10 个测试分别覆盖题面列出的 10 个缺陷，另有 5 个组合/边界测试。
初始状态下所有测试都调用尚未实现的 denoise.temporal 桩，因此应恰好
15 failed，且没有 collection/import/syntax error。
"""

import pytest

from denoise.temporal import (
    DenoiseError,
    accumulate,
    blend_temporal,
    denoise,
    denoise_batch,
    frame_diff,
    ghost_score,
    motion_mask,
    validate_config,
)


def _assert_frame_close(actual, expected, tolerance=1e-6):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, abs=tolerance)


def test_frame_diff_uses_absolute_difference():
    """缺陷 1：帧差用有符号相减，负值被当成 0，运动漏检。"""
    a = [[10.0, 80.0], [200.0, 5.0]]
    b = [[40.0, 60.0], [100.0, 50.0]]

    result = frame_diff(a, b)

    _assert_frame_close(result, [[30.0, 20.0], [100.0, 45.0]])


def test_motion_mask_dilates_object_edges():
    """缺陷 2：掩码不做膨胀，物体边缘残留噪点。"""
    diff = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 80.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    mask = motion_mask(diff, threshold=20.0, dilate=1)

    assert mask[2][2] is True
    for y, x in (
        (1, 2),
        (3, 2),
        (2, 1),
        (2, 3),
        (1, 1),
        (1, 3),
        (3, 1),
        (3, 3),
    ):
        assert mask[y][x] is True, f"dilation missed ({y}, {x})"
    assert mask[0][0] is False


def test_motion_pixels_keep_current_frame():
    """缺陷 3：运动区也参与平均，走动人出现拖影。"""
    previous = [[0.0] * 4 for _ in range(4)]
    current = [[0.0] * 4 for _ in range(4)]
    current[1][1] = 120.0
    config = {
        "threshold": 10.0,
        "max_history": 2,
        "alpha": 1.0,
        "dilate": 0,
    }

    result = denoise(current, [previous], config)

    assert result[1][1] == pytest.approx(120.0, abs=1e-6)
    assert result[0][0] == pytest.approx(0.0, abs=1e-6)


def test_accumulate_slides_within_max_history():
    """缺陷 4：累积不设上限，内存线性增长。"""
    history = [[[0.0]], [[1.0]], [[2.0]], [[3.0]]]

    result = accumulate(history, [[4.0]], max_history=3)

    assert len(result) == 3
    assert [frame[0][0] for frame in result] == pytest.approx([2.0, 3.0, 4.0])
    assert len(history) == 4


def test_denoise_uses_adaptive_dark_weights():
    """缺陷 5：权重不按噪声水平自适应，暗部过平滑。"""
    previous = [[0.0, 0.0]]
    current = [[20.0, 220.0]]
    config = {
        "threshold": 255.0,
        "max_history": 1,
        "alpha": 1.0,
        "dilate": 0,
    }

    result = denoise(current, [previous], config)

    dark_retention = result[0][0] / 20.0
    bright_retention = result[0][1] / 220.0
    assert dark_retention > 0.5
    assert dark_retention > bright_retention + 0.1


def test_ghost_score_focuses_on_masked_region():
    """缺陷 6：残影评分按全帧平均，局部残影被稀释。"""
    current = [[0.0] * 8 for _ in range(8)]
    previous = [[0.0] * 8 for _ in range(8)]
    previous[3][3] = 255.0
    mask = [[False] * 8 for _ in range(8)]
    mask[3][3] = True

    score = ghost_score(current, previous, mask)

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-6)


def test_validate_config_rejects_out_of_range_alpha():
    """缺陷 7：配置不校验 alpha，超过 1 后画面发灰。"""
    problems = validate_config(
        {"threshold": 20.0, "max_history": 4, "alpha": 1.2}
    )

    assert isinstance(problems, list)
    assert any("alpha" in problem.lower() for problem in problems)
    assert validate_config(
        {"threshold": 20.0, "max_history": 4, "alpha": 0.5}
    ) == []


def test_first_frame_without_history_is_not_darkened():
    """缺陷 8：首帧没有历史，直接用 0 帧混合，整体变暗。"""
    frame = [[8.0, 96.0], [160.0, 240.0]]
    config = {
        "threshold": 16.0,
        "max_history": 3,
        "alpha": 0.9,
        "dilate": 1,
    }

    result = denoise(frame, [], config)

    _assert_frame_close(result, frame)


def test_batch_does_not_share_history_across_calls():
    """缺陷 9：批量共用历史队列，镜头之间串。"""
    config = {
        "threshold": 16.0,
        "max_history": 3,
        "alpha": 0.85,
        "dilate": 0,
    }
    denoise_batch([[[0.0]], [[0.0]]], config)

    second = denoise_batch([[[200.0]]], config)

    assert second["count"] == 1
    _assert_frame_close(second["frames"][0], [[200.0]])


def test_size_mismatch_and_batch_validation_raise():
    """缺陷 10：尺寸不一致不报错，逐像素越界。"""
    with pytest.raises(DenoiseError):
        frame_diff([[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0]])

    config = {
        "threshold": 16.0,
        "max_history": 2,
        "alpha": 0.5,
        "dilate": 0,
    }
    with pytest.raises(DenoiseError):
        denoise_batch([[[0.0, 0.0]], [[0.0]]], config)


def test_alpha_zero_keeps_current_frame():
    """组合：alpha=0 时不混合，直接保留当前帧。"""
    current = [[10.0, 20.0], [30.0, 40.0]]
    previous = [[200.0, 200.0], [200.0, 200.0]]
    alpha_map = [[0.0, 0.0], [0.0, 0.0]]

    result = blend_temporal(current, previous, alpha_map)

    _assert_frame_close(result, current)


def test_empty_batch_returns_empty_result():
    """组合：空帧列表返回空结果。"""
    config = {
        "threshold": 16.0,
        "max_history": 2,
        "alpha": 0.5,
        "dilate": 0,
    }

    result = denoise_batch([], config)

    assert result == {"frames": [], "count": 0}


def test_accumulate_rejects_zero_max_history():
    """组合：max_history=0 必须抛 DenoiseError。"""
    with pytest.raises(DenoiseError):
        accumulate([], [[0.0]], max_history=0)


def test_batch_preserves_input_order():
    """组合：批量全成功时输出顺序与输入一致。"""
    frames = [[[10.0]], [[20.0]], [[30.0]]]
    config = {
        "threshold": 16.0,
        "max_history": 3,
        "alpha": 0.0,
        "dilate": 0,
    }

    result = denoise_batch(frames, config)

    assert result["count"] == 3
    assert len(result["frames"]) == 3
    for actual, expected in zip(result["frames"], frames):
        _assert_frame_close(actual, expected)


def test_single_frame_without_history_returns_original():
    """组合：单帧无历史时返回原帧，误差不超过 1e-6。"""
    frame = [[5.0, 7.0], [11.0, 13.0]]
    config = {
        "threshold": 16.0,
        "max_history": 3,
        "alpha": 0.75,
        "dilate": 1,
    }

    result = denoise_batch([frame], config)

    assert result["count"] == 1
    _assert_frame_close(result["frames"][0], frame)
