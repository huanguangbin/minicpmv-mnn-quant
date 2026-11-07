# 量化对比实测结果

- **平台**：Apple MacBook Air（M5，16GB 统一内存），macOS 27.2
- **框架**：MNN main 分支（commit b6cb126），Metal 后端，thread_num=4，precision=low
- **模型**：MiniCPM-V 4.7-1B（1.55B，dense）
- **日期**：2026-11-07
- **复现方式**：`scripts/01_build_mnn.sh` → `02_download_models.sh` → `03_export_models.sh` → `04_bench_all.sh`

## 核心数据

| 模型 | 语言权重 | 总体积 | prefill (tok/s) | decode (tok/s) | 文本正确率* |
|---|---|---|---|---|---|
| FP16（自导出） | 1.40 GiB | 2.9G | 2182 | 34.3 | 80.0% |
| INT8（自导出） | 762 MiB | 1.8G | 2346 | 123.0 | 83.3% |
| HQQ4（自导出） | 404 MiB | 1.4G | 2349 | **209.3** | 76.7% |
| 官方 HQQ 版 | 404 MiB | 1.4G | 2351 | 210.2 | 76.7% |

\* 30 道中文常识题，greedy 采样，关键词匹配判分（详见 `eval/README.md` 的局限性说明）

## 结论

1. **decode 是内存带宽受限（memory-bound）**：权重从 1.4G → 404M，decode 速度 34 → 209 tok/s（**6.1×**）；
2. **prefill 是计算受限（compute-bound）**：四版本均在 2182–2351 tok/s 区间，量化影响小；
3. **自导出 HQQ4 与官方版三项指标完全对齐**（体积逐文件一致、速度差 0.4%、精度逐题一致），验证量化复现正确；
4. 简单常识任务上，FP16→INT8→HQQ4 精度差异在噪声范围内（±1 题），需更难评测集才能体现量化掉点。

## 图像描述抽检（bus.jpg / zidane.jpg）

四版本均能正确描述（公交车/行人/球场人物），无幻觉；HQQ4 与官方版输出逐字一致。

## 推理时延参考（单次）

- 文本（17 token prompt）：prefill 0.65s + decode 0.11s（25 token）
- 图像（810×1080 jpg）：视觉编码 ~10s + prefill 1.2s + decode 0.11s（21 token）
- **视觉编码是端侧 VLM 最大耗时项**，优化优先级：视觉 token 压缩 > 视觉塔量化 > 算子融合
