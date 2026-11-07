# PTQ 量化项目 · 学习日志（阶段 0：环境准备）

> 目标：独立完成 MiniCPM-V 4.7-1B 的端侧量化对比（FP16 / INT8 / HQQ），产出可复现的流程 + 数据。
> 本日志记录**每一步做了什么、为什么、命令和路径**，供学习复现。
> 项目根目录：`/Users/indef/Projects/demo/minicpmv-mnn/`（下文用 `$PROJ` 代指）

---

## 目录总览（截至阶段 0）

```
$PROJ = /Users/indef/Projects/demo/minicpmv-mnn/
├── .venv/                          # uv 建的 Python 3.12 虚拟环境（隔离所有 Python 包）
├── .gitignore                      # 已加 .venv/
├── docs/                           # 文档
│   ├── 01-MNN-MiniCPM-V-4.7-1B-部署实战.md   # 上一阶段：跑通推理
│   └── 02-PTQ量化项目-学习日志.md            # 本文件
├── mnn/                            # MNN 源码 + build/（编译产物）
│   ├── build/llm_demo              # 推理 demo
│   ├── build/llm_bench             # 测速工具
│   ├── build/MNNConvert            # 模型转换工具（本阶段新增）
│   └── transformers/llm/export/llmexport.py  # 导出+量化脚本
└── models/
    ├── MiniCPM-V-4.7-1B-MNN/       # 已量化的 MNN 模型（官方 HQQ 版，1.5GB）
    └── MiniCPM-V-4.7-1B-hf/        # 原始 PyTorch 权重（本阶段新增，2.9GB）
```

---

## 一、下载了什么（全部文件 + 来源 + 大小）

| 内容 | 来源 | 大小 | 下载方式 |
|---|---|---|---|
| MNN 源码 | `https://github.com/alibaba/MNN.git`（浅克隆 main） | ~100MB | `git clone --depth 1` |
| 官方 MNN 量化模型 | HF `taobao-mnn/MiniCPM-V-4.7-1B-MNN` | 1.5GB | `curl` 逐文件 |
| 原始 PyTorch 权重 | HF `openbmb/MiniCPM-V-4.7-1B` | 2.9GB | `huggingface_hub.snapshot_download` |

### 1.1 官方 MNN 量化模型（10 个文件）

路径：`$PROJ/models/MiniCPM-V-4.7-1B-MNN/`

| 文件 | 作用 |
|---|---|
| `llm.mnn` + `llm.mnn.weight` | 语言模型计算图 + 权重（HQQ4 量化） |
| `visual.mnn` + `visual.mnn.weight` | 视觉编码器计算图 + 权重（HQQ8 量化） |
| `embeddings_bf16.bin` | 输入词嵌入（BF16） |
| `tokenizer.mtok` | MNN 专用分词器 |
| `config.json` / `llm_config.json` | 主配置 / 语言结构配置 |
| `conversion.json` / `export_args.json` | 转换溯源（记录了量化参数） |

**关键学习点**：这个模型的语言部分是 **HQQ4**、视觉部分是 **HQQ8** —— 说明 MNN 的 VLM 量化是"语言、视觉分开量化"的。我们自己做 PTQ 时也要搞清这两部分分别怎么设。

### 1.2 原始 PyTorch 权重（9 个文件）

路径：`$PROJ/models/MiniCPM-V-4.7-1B-hf/`

| 文件 | 作用 |
|---|---|
| `model.safetensors`（2.9GB） | 模型全部权重（单文件） |
| `tokenizer.json` / `tokenizer_config.json` | 分词器 |
| `config.json` | 模型结构配置 |
| `preprocessor_config.json` / `processor_config.json` | 图像预处理配置 |
| `chat_template.jinja` | 对话模板 |
| `generation_config.json` | 生成配置 |

**关键学习点**：这是 `llmexport.py` 的输入，也是你做量化的"源头"。从它出发，你可以导出任意精度的 MNN 模型。

---

## 二、安装了什么（全部 + 位置 + 版本）

### 2.1 系统级（影响全局，装之前已报备）

| 工具 | 版本 | 安装方式 | 位置 |
|---|---|---|---|
| cmake | 4.4.5 | `brew install cmake` | `/opt/homebrew/bin/cmake` |

### 2.2 Python 虚拟环境（项目内，不影响系统）

**创建命令**：
```bash
cd $PROJ
uv venv --python 3.12 .venv     # 用 uv 建，Python 3.12.16
```

- 位置：`$PROJ/.venv/`
- 原理：`.venv` 只是个小目录，第三方包装在这里，解释器和标准库共享 uv 中央 Python（不复制）
- 已加 `.venv/` 到 `.gitignore`（不提交，靠依赖清单重建）

### 2.3 Python 包（全部装在 .venv 里）

**安装命令**：
```bash
uv pip install torch transformers onnx onnxslim onnxruntime \
  safetensors numpy sentencepiece tqdm yaspin \
  Pillow requests huggingface_hub accelerate einops datasets peft
```

**装了什么、为什么**（对应 `llmexport.py` 的依赖）：

| 包 | 版本 | 作用 |
|---|---|---|
| torch | 2.15.0 | 深度学习框架，MPS 后端可用 |
| transformers | 5.22.2 | 加载原始 HF 权重、分词器 |
| onnx | 1.23.2 | 中间格式（PyTorch → ONNX → MNN） |
| onnxslim / onnxruntime | — | ONNX 图优化 / 校验 |
| safetensors | 0.8.0 | 读 `.safetensors` 权重文件 |
| sentencepiece | 0.2.3 | 分词器 |
| numpy / Pillow | — | 数值 / 图像 |
| huggingface_hub | — | 下载 HF 模型 |
| accelerate / einops | — | transformers 加速 / 张量操作 |
| datasets / peft | — | AWQ/SmoothQuant 校准数据、PeftModel 加载 |
| tqdm / yaspin | — | 进度条 |

**关键学习点**：`llmexport.py` 的官方依赖清单在 `mnn/transformers/llm/export/requirements.txt`。另外注意——**hqq 不用单独装**，因为 `llmexport.py` 里用 `--hqq` 开关，量化逻辑是脚本内置的（`torch_quant()` 函数）。

---

## 三、编译了什么（MNN 三个工具）

### 3.1 编译命令（关键开关要理解）

```bash
cd $PROJ/mnn
cmake -S . -B build \
  -DMNN_BUILD_LLM=ON \          # LLM 推理引擎
  -DMNN_LOW_MEMORY=ON \         # 低内存模式（端侧必需）
  -DMNN_BUILD_LLM_OMNI=ON \     # 多模态（视觉/语音）支持
  -DMNN_IMGCODECS=ON \          # 内置图像编解码
  -DMNN_METAL=ON \              # Apple GPU 加速
  -DMNN_BUILD_OPENCV=OFF \      # 不需要 OpenCV
  -DMNN_BUILD_CONVERTER=ON      # 【本阶段新增】生成 MNNConvert

cmake --build build --target llm_demo llm_bench MNNConvert -j8
```

### 3.2 产物（都在 `$PROJ/mnn/build/`）

| 工具 | 用途 |
|---|---|
| `llm_demo` | 跑推理（读 config + prompt，输出文本） |
| `llm_bench` | 测 prefill/decode 速度（量化对比的基线数据来源） |
| `MNNConvert` | **模型转换工具**（ONNX → MNN，PTQ 导出必经之路） |

---

## 四、踩过的坑（重要学习点）

### 坑 1：macOS 动态库 rpath 烙死绝对路径

- **现象**：移动项目目录后运行报 `Library not loaded: @rpath/libllm.dylib`
- **原因**：编译链接时，`@rpath` 里写的是当时的绝对路径（如 `.../mnn/build`），移动后失效
- **两个解法**：
  - 推荐：`install_name_tool` 加相对路径（`@executable_path` / `@loader_path`），一劳永逸
  - 备选：删 `build/` 重新编译

### 坑 2：CMakeCache.txt 也烙死路径（比 rpath 更麻烦）

- **现象**：移动后 `cmake` 重配报 `CMakeCache.txt directory is different ...`
- **原因**：`CMakeCache.txt` 记录了编译时的源目录绝对路径，移动后不一致
- **解决**：只能 `rm -rf build` 重新 configure + 编译
- **结论**：`install_name_tool` 只能救运行时的 rpath，救不了 CMakeCache；**目录定好后尽量别动**

---

## 五、当前状态检查

```bash
# 验证 venv
$PROJ/.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# → 2.15.0 True

# 验证三个工具
ls $PROJ/mnn/build/{llm_demo,llm_bench,MNNConvert}
# → 三个都在

# 验证推理仍正常
cd $PROJ/models/MiniCPM-V-4.7-1B-MNN
$PROJ/mnn/build/llm_demo ./config-test-metal.json ./prompts.txt 64
# → 正常输出，decode ~210 tok/s
```

---

## 六、阶段 1：导出多精度模型（已完成）

### 6.1 量化参数速查（`llmexport.py --help` 解读）

| 参数 | 取值 | 含义 |
|---|---|---|
| `--quant_bit` | 2/3/4/8/**16** | 语言权重位宽，**16 = FP16 不量化**，8 = INT8，4 = INT4 |
| `--quant_block` | 0=逐通道，默认 64 | 量化分块大小（HQQ 用 64） |
| `--visual_quant_bit` | 4/8/16 | 视觉编码器位宽（默认按 vit 模型定） |
| `--lm_quant_bit` | 4/8/16 | 输出层 lm_head 位宽（默认=quant_bit） |
| `--hqq` | 开关 | 用 HQQ 量化（否则是普通 RTN） |
| `--scale_bit` | 16/32 | 量化 scale 的存储精度（FP16/FP32） |
| `--sym` | 开关 | 对称量化（无零点） |
| `--export` | onnx/mnn | 导出格式 |
| `--mnnconvert` | 路径 | 指定 MNNConvert 二进制 |

**核心理解**：`--quant_bit 16` = FP16（不量化）；`--quant_bit 8` = INT8；`--quant_bit 4 --hqq` = HQQ4。语言/视觉/lm_head 三部分可以**独立设精度**。

### 6.2 三条导出命令（在 `mnn/transformers/llm/export/` 下执行）

```bash
PY=../.venv/bin/python
SRC=../models/MiniCPM-V-4.7-1B-hf
MNNConvert=../mnn/build/MNNConvert

# ① FP16 基线（不量化）
$PY llmexport.py --path $SRC --export mnn \
  --quant_bit 16 --visual_quant_bit 16 --lm_quant_bit 16 \
  --mnnconvert $MNNConvert --dst_path ../models/MiniCPM-V-4.7-1B-FP16

# ② INT8（weight-only RTN）
$PY llmexport.py --path $SRC --export mnn \
  --quant_bit 8 --visual_quant_bit 8 --lm_quant_bit 8 \
  --mnnconvert $MNNConvert --dst_path ../models/MiniCPM-V-4.7-1B-INT8

# ③ HQQ4（对标官方：语言 HQQ4 + 视觉默认 HQQ8）
$PY llmexport.py --path $SRC --export mnn \
  --hqq --quant_bit 4 --quant_block 64 --scale_bit 16 \
  --mnnconvert $MNNConvert --dst_path ../models/MiniCPM-V-4.7-1B-HQQ4
```

**踩坑记录**：首次导出报 `requires torchvision to be installed`，因为 VLM 的图像/视频预处理用到了 torchvision。补装 `uv pip install torchvision`（0.29.2，与 torch 2.15 匹配）后通过。

### 6.3 体积对比结果

| 模型 | 语言权重 | 视觉权重 | 总计 |
|---|---|---|---|
| FP16 | 1.4G | 1.0G | 2.9G |
| INT8 | 762M | 564M | 1.8G |
| **HQQ4（我们导的）** | 404M | 564M | **1.4G** |
| 官方版 | 404M | 564M | 1.4G |

✅ 我们导出的 HQQ4 与官方版大小**完全一致**，证明量化参数复现正确。

---

## 七、阶段 2：速度基准（已完成）

### 7.1 命令

```bash
llm_bench -m config-test-metal.json -a metal -c 1 -t 4 -p 512 -n 128 -rep 2
# -p 512 测 prefill（处理 512 token）
# -n 128 测 decode（生成 128 token）
```

### 7.2 速度对比结果（Metal / M5）

| 模型 | 语言权重 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|
| FP16 | 1.40 GiB | 2182 | **34.3** |
| INT8 | 762 MiB | 2345 | 123.0 |
| **HQQ4（我们导的）** | 404 MiB | 2349 | **209.3** |
| 官方版 | 404 MiB | 2351 | 210.2 |

### 7.3 结论（面试能讲的核心洞察）

1. **decode 是内存带宽受限的**：权重越小，decode 越快。FP16(1.4G)→INT8(762M)→HQQ4(404M)，decode 速度 34→123→209 tok/s，**HQQ4 是 FP16 的 6.1 倍**。
2. **prefill 是计算受限的**：四个版本 prefill 都在 2182–2351 tok/s 之间，差异小；INT8/HQQ4 略快是因为 Metal 的 int8 点积更快。
3. **我们导出的 HQQ4 与官方版速度完全一致**（209 vs 210），再次证明复现正确。
4. **这组数据直接回答了"为什么要量化"**：同等精度下，量化带来 6 倍 decode 加速 + 2 倍体积压缩。

---

## 八、阶段 3：精度评测（已完成）

### 8.1 文本正确率（30 道常识题，关键词匹配判分）

| 模型 | 正确率 |
|---|---|
| FP16 | 80.0% |
| INT8 | 83.3% |
| **HQQ4（我们导的）** | **76.7%** |
| 官方版 | 76.7% |

**评测脚本**：`eval/run_text_eval.py`（每题单独跑、greedy 采样、关键词匹配，结果存 `eval/text_eval_results.json`）

### 8.2 图像描述（2 张图，定性对比）

四个模型都能正确描述图像（公交车、行人、球场上的人物），**无幻觉**；HQQ4 与官方版描述**逐字一致**。

### 8.3 诚实说明评测的局限性

1. **部分“错”是判分误伤**：Q7 光速（模型答“3×10^8米/秒”其实对，答案关键词写的是“30万公里每秒”）、Q8 公转周期同理；
2. **MAX_TOKENS=48 偏短**：模型有冗长 preamble，个别答案被截断（如 10×10 没等说出 100 就截断）；
3. **题目太简单**：常识题区分度低，量化掉点体现不出来——要看到真实精度差异，需要更难的推理/数学/长文本任务。

### 8.4 结论

- **HQQ4 与官方版在正确率、逐题答案、图像描述上完全一致**（第三次验证复现正确）；
- **在简单任务上，FP16→INT8→HQQ4 的精度差异在噪声范围内**（76.7%–83.3%，30 题差异 ±1 题），无法区分；
- 这本身是个有价值的结论：**量化带来的速度/体积收益（6× 加速、2× 压缩）在简单任务上几乎免费**。

---

## 九、完整对比总表（项目核心产出）

| 模型 | 语言权重 | 总计体积 | decode 速度 | 文本正确率 |
|---|---|---|---|---|
| FP16 | 1.4G | 2.9G | 34 tok/s | 80.0% |
| INT8 | 762M | 1.8G | 123 tok/s (3.6×) | 83.3% |
| HQQ4（我们） | 404M | 1.4G | 209 tok/s (6.1×) | 76.7% |
| 官方版 | 404M | 1.4G | 210 tok/s | 76.7% |

**一句话结论**：HQQ4 相对 FP16 带来 **6.1× decode 加速 + 2× 体积压缩**，简单任务上精度无显著损失；我们独立导出的 HQQ4 与官方版在体积、速度、精度上完全一致。

---

## 十、剩余交付物（阶段 4）

1. 把评测脚本 + 导出脚本整理进 GitHub 仓库（可复现）；
2. 写一篇博客《端侧 VLM 量化实战：FP16/INT8/HQQ 怎么选》；
3. 简历话术（我来帮你写）。

*本日志所有命令均已在 Apple M5 / macOS 27.2 实测通过。*
