# MiniCPM-V 4.7-1B 端侧量化实战（MNN）

> 从原始 PyTorch 权重出发，独立完成多模态大模型（VLM）的 **FP16 / INT8 / HQQ4** 三档量化导出，
> 在 Apple Silicon（Metal）上部署并实测「精度–速度–体积」三维权衡。

## TL;DR — 实测结果（M5 / Metal）

| 模型 | 语言权重 | decode 速度 | 文本正确率 |
|---|---|---|---|
| FP16 | 1.40 GiB | 34.3 tok/s | 80.0% |
| INT8 | 762 MiB | 123.0 tok/s | 83.3% |
| **HQQ4（本项目导出）** | **404 MiB** | **209.3 tok/s（6.1×）** | 76.7% |
| 官方 HQQ 版（对照） | 404 MiB | 210.2 tok/s | 76.7% |

**核心发现**：

- **decode 是 memory-bound**：权重每生成一个 token 都要整读一遍，压缩权重 = 直接提速（6.1×）；**prefill 是 compute-bound**，量化影响小（四版本均在 ~2300 tok/s）。
- 自导出 HQQ4 与官方版在体积、速度、精度上**完全对齐**，量化参数复现正确。
- 端侧 VLM 的最大耗时项是**视觉编码（~10s/图）**，而非语言 decode。

详细数据：[`results/benchmark.md`](results/benchmark.md)

## 完整复现（4 步）

```bash
git clone <本仓库> && cd minicpmv-mnn
brew install uv cmake

./scripts/00_setup_env.sh        # uv venv (Python 3.12) + torch/transformers 等
./scripts/01_build_mnn.sh        # 编译 MNN（Metal + LLM + 多模态 + 转换器）
./scripts/02_download_models.sh  # 下载官方量化版 + 原始权重（~4.4GB）
./scripts/03_export_models.sh    # 导出 FP16 / INT8 / HQQ4（可单选：fp16|int8|hqq4）

./scripts/04_bench_all.sh        # 速度基准（prefill/decode）
.venv/bin/python eval/run_text_eval.py   # 文本精度评测（30 题）
```

## 仓库结构

```
├── scripts/                    # 全流程可复现脚本（按序号执行）
│   ├── 00_setup_env.sh         # Python 3.12 venv + 依赖
│   ├── 01_build_mnn.sh         # MNN 编译（含 macOS rpath 修复）
│   ├── 02_download_models.sh   # 模型下载（HF）
│   ├── 03_export_models.sh     # ★ 三档量化导出
│   └── 04_bench_all.sh         # 速度基准
├── eval/                       # 精度评测
│   ├── run_text_eval.py        # 自动评测脚本（120 次推理）
│   ├── text_qa.json            # 30 道中文评测题
│   ├── text_eval_results.json  # 评测结果
│   └── images/                 # 图像评测样例（图源：ultralytics/yolov5）
├── results/
│   └── benchmark.md            # 完整实测数据 + 结论
├── docs/                       # 部署与学习文档
│   ├── 01-...部署实战.md       # 从零编译到推理的手册（含踩坑）
│   └── 02-PTQ量化项目-学习日志.md
└── models/                     # 模型目录（.gitignore 排除大文件，由脚本下载生成）
```

## 量化参数速查（llmexport.py）

| 目标 | 关键参数 |
|---|---|
| FP16（不量化） | `--quant_bit 16 --visual_quant_bit 16 --lm_quant_bit 16` |
| INT8 | `--quant_bit 8 --visual_quant_bit 8 --lm_quant_bit 8` |
| HQQ4 | `--hqq --quant_bit 4 --quant_block 64 --scale_bit 16` |

要点：MNN 的 VLM 量化支持**语言骨干 / 视觉编码器 / lm_head 三部分独立配置**；
视觉部分默认走 HQQ8（block 128），语言部分 HQQ4（block 64）。

## 已知局限（诚实声明）

1. 评测集仅 30 道中文常识题，**区分度有限**，量化真实掉点需更难的推理/数学评测集（后续计划接入 C-Eval 子集）；
2. 关键词判分存在少量误伤（如光速答 "3×10^8 m/s" 判错）；
3. 图像评测为小样本定性对比（2 张图），计划扩充。

## 技术要点备忘

- **macOS rpath 坑**：MNN 编译产物默认烙绝对路径 rpath，移动目录即失效；用 `install_name_tool` 加 `@executable_path` 相对 rpath 一劳永逸（见 `01_build_mnn.sh`）。
- **CMakeCache 坑**：移动过目录的 build 无法增量重配，只能删掉重建。
- **运行配置**：主配置与 `mllm` 段的 `backend_type` 必须一致，否则视觉/语言跨设备执行。

## 致谢与许可

- 模型：[OpenBMB/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)、[taobao-mnn/MiniCPM-V-4.7-1B-MNN](https://huggingface.co/taobao-mnn/MiniCPM-V-4.7-1B-MNN)（Apache-2.0）
- 框架：[alibaba/MNN](https://github.com/alibaba/MNN)
- 评测图片：[ultralytics/yolov5](https://github.com/ultralytics/yolov5) data/images
