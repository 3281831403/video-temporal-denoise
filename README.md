# video-temporal-denoise

## 题面

夜间素材做时域降噪后，暗部糊成一团、走动人还有拖影：① 帧差使用有符号相减，负差被当成 0，运动漏检；② 掩码不做膨胀，物体边缘残留噪点；③ 运动区也参与时域平均，走动人出现拖影；④ 累积历史不设上限，内存线性增长；⑤ 混合权重不按噪声水平自适应，暗部过平滑；⑥ 残影评分按全帧平均，局部残影被稀释；⑦ 配置不校验 alpha，超过 1 后画面发灰；⑧ 首帧没有历史，直接用 0 帧混合，整体变暗；⑨ 批量共用历史队列，镜头之间串；⑩ 尺寸不一致不报错，逐像素越界。denoise/temporal.py 是核心，tests/test_temporal.py 15个红。要修的是：frame_diff 的绝对差与尺寸校验、motion_mask 的阈值与膨胀、blend_temporal 的运动区零权重、accumulate 的滑窗上限、denoise 的自适应权重与首帧回退、ghost_score 的局部最大评分、validate_config 的 alpha 校验、denoise_batch 的独立历史与整批回滚。

## 运行方式

```powershell
cd D:\gsb-20261119\video-temporal-denoise
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `denoise/temporal.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_temporal.py` 是评测基准，**不可修改**；只允许修改 `denoise/temporal.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数，不调用 `sleep`。
