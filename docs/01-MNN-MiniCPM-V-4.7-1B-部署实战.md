# 用 MNN 在 Mac 上部署 MiniCPM-V 4.7-1B（端侧多模态 VLM）

> 目标：从零开始，在 Apple Silicon Mac 上用 MNN 框架跑通端侧多模态大模型 **MiniCPM-V 4.7-1B**，并理解每一步在做什么。
>
> 完成日期：2026-11-07
> 机器：MacBook Air，Apple M5，16GB 统一内存，macOS 27.2
> 最终结果：✅ 文本推理 + 图像推理均跑通（Metal GPU 后端）

---

## 目录结构

```
~/Projects/demo/minicpmv-mnn/
├── docs/                          # 文档（本文件）
├── mnn/                           # MNN 源码 + build/ 编译产物
│   └── build/llm_demo             # 编译出的推理可执行文件
└── models/
    └── MiniCPM-V-4.7-1B-MNN/      # 下载的 MNN 格式模型
        ├── config.json            # 主配置（模型文件路径、采样参数等）
        ├── llm_config.json        # 语言模型结构配置
        ├── llm.mnn / llm.mnn.weight         # 语言模型图 + 权重（HQQ4 量化）
        ├── visual.mnn / visual.mnn.weight   # 视觉编码器图 + 权重（HQQ8 量化）
        ├── embeddings_bf16.bin    # 输入嵌入（BF16）
        ├── tokenizer.mtok         # 分词器
        └── config-test-metal.json # 我们自己生成的运行配置（Metal GPU）
```

---

## 第 0 步：环境准备

MNN 编译需要 `cmake` 和 `clang`（Xcode Command Line Tools）。

```bash
# 1. 确认编译器存在（一般 Xcode 已自带）
xcode-select -p          # 输出 /Applications/Xcode.app/Contents/Developer 即正常

# 2. 安装 cmake（MNN 的构建系统）
brew install cmake
cmake --version           # 实测：4.4.5
```

**说明**：MNN 用 CMake 构建，Apple Silicon 上默认走 clang + Metal，不需要额外装 CUDA 之类的东西。

---

## 第 1 步：编译 MNN（带 LLM + 多模态 + Metal 支持）

MNN 默认编译的是传统 CV 推理；要跑大模型/多模态，必须显式打开 LLM 相关开关。

```bash
cd ~/Projects/demo
git clone --depth 1 https://github.com/alibaba/MNN.git mnn
cd mnn
```

### 1.1 确认 MiniCPM-V 4.7 支持已合入

主分支源码里有 `minicpmv4_7` 这个模型类型，说明已支持：

```bash
grep -n "minicpmv4_7" transformers/llm/export/utils/model_mapper.py
# 输出类似：if model_type in ('minicpmv4_7', 'minicpm_ocr'):
grep -n "MiniCPM47" transformers/llm/engine/src/omni.cpp
# 输出：struct MiniCPM47 ...（视觉处理器，含 canvas 位置计算）
```

### 1.2 CMake 配置（关键开关解释）

```bash
cmake -S . -B build \
  -DMNN_BUILD_LLM=ON \          # 编译 LLM 推理引擎（跑大模型必需）
  -DMNN_LOW_MEMORY=ON \         # 低内存模式（端侧必需，权重按需加载）
  -DMNN_BUILD_LLM_OMNI=ON \     # 多模态（Omni）支持：图像/语音输入（VLM 必需）
  -DMNN_IMGCODECS=ON \          # 内置图像编解码（读 jpg/png，无需 OpenCV）
  -DMNN_METAL=ON \              # Apple GPU 加速（M 系列芯片必需）
  -DMNN_BUILD_OPENCV=OFF \      # 不需要 OpenCV（用上面的内置编解码即可）
  -DMNN_BUILD_CONVERTER=OFF     # 本次只跑现成模型，不需要模型转换器
```

> 对比：官方 README 里给的转换命令是 `-DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_OPENCV=ON`，
> 那是「自己从 PyTorch 转 MNN + 自己做量化」时才需要。我们这次先跑现成的量化模型，
> 所以关掉这两项，编译更快、依赖更少。

### 1.3 编译

```bash
cmake --build build --target llm_demo -j8
```

- 产物：`build/llm_demo`（多模态推理 demo，读 config + prompt 文件，输出结果）。
- 后续做 benchmark 还可以加：`cmake --build build --target llm_bench -j8`。

---

## 第 2 步：下载 MNN 格式的模型

模型来源：HuggingFace 上的 `taobao-mnn/MiniCPM-V-4.7-1B-MNN`（阿里 MNN 团队转换好的官方版）。

**为什么用这个而不是原始 PyTorch 权重**：它已经是「MNN 图 + 量化权重 + 专用 tokenizer」的成品，省去自己转换的步骤；量化方案是语言部分 HQQ4、视觉部分 HQQ8。

### 2.1 模型文件清单（下载前先了解）

| 文件 | 大小 | 作用 |
|---|---|---|
| `llm.mnn` | 2.5 MB | 语言模型计算图（算子拓扑） |
| `llm.mnn.weight` | 404 MB | 语言模型权重（HQQ4，块 64，FP16 scale） |
| `embeddings_bf16.bin` | 485 MB | 输入词嵌入（BF16 存） |
| `visual.mnn` | 0.5 MB | 视觉编码器计算图 |
| `visual.mnn.weight` | 564 MB | 视觉编码器权重（HQQ8，块 128） |
| `tokenizer.mtok` | 7 MB | MNN 专用分词器 |
| `config.json` | — | 主配置 |
| `llm_config.json` | — | 语言模型结构参数 |
| `conversion.json` / `export_args.json` | — | 转换溯源信息 |

总计约 **1.5 GB**。

### 2.2 下载（不装 git-lfs，直接 curl）

HuggingFace 的下载 URL 规律：`https://huggingface.co/{仓库}/resolve/main/{文件名}`

```bash
cd ~/Projects/demo/minicpmv-mnn/models && mkdir -p MiniCPM-V-4.7-1B-MNN && cd $_
BASE="https://huggingface.co/taobao-mnn/MiniCPM-V-4.7-1B-MNN/resolve/main"
for f in config.json llm_config.json conversion.json export_args.json \
         tokenizer.mtok llm.mnn visual.mnn \
         llm.mnn.weight visual.mnn.weight embeddings_bf16.bin; do
  curl -L -C - --retry 3 -o "$f" "$BASE/$f"
done
```

> 提示：如果直连慢，把 `https://huggingface.co` 换成镜像 `https://hf-mirror.com`。
> `-C -` 支持断点续传，中断后重跑同一条命令即可继续。

---

## 第 3 步：创建运行配置

`config.json` 是模型自带的默认配置（CPU、随机采样）。我们要生成一份针对 GPU + 确定性输出的测试配置。

### 3.1 配置字段含义

| 字段 | 含义 |
|---|---|
| `backend_type` | 计算后端：`cpu` / `metal`（Apple GPU）/ `opencl` / `cuda` |
| `precision` | 精度：`low`（低精度/量化）/ `normal` / `high` |
| `thread_num` | CPU 线程数（Metal 时影响较小） |
| `memory` | 内存模式：`low`（低内存，端侧必需） |
| `sampler_type` | 采样方式：`greedy`（贪心，最确定）/ `mixed`（混合采样） |
| `temperature` / `top_k` / `top_p` | 采样温度与截断参数 |
| `mllm` | 视觉/多模态部分的独立后端配置（要和主配置保持一致） |

### 3.2 生成配置文件（Python）

```python
import json
from pathlib import Path
base = json.loads(Path('config.json').read_text())

# Metal GPU 版
cfg = dict(base)
cfg.update(backend_type='metal', precision='low', thread_num=4,
           sampler_type='greedy', temperature=0.0, top_k=1)
cfg['mllm'] = dict(backend_type='metal', precision='low', thread_num=4, memory='low')
Path('config-test-metal.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
```

> 关键点：主配置和 `mllm` 里的 `backend_type` **必须一致**（都 metal 或都 cpu），
> 否则视觉部分和语言部分在不同设备上跑，会出问题。

---

## 第 4 步：准备测试输入

### 4.1 纯文本 prompt（`prompts.txt`）

每行一条 prompt：

```
请用一句话介绍北京。
```

### 4.2 图像 prompt（`prompts_img.txt`）

图片用 `<img>绝对路径</img>` 包起来，后面跟问题：

```
<img>/Users/indef/Projects/demo/minicpmv-mnn/models/MiniCPM-V-4.7-1B-MNN/test_bus.jpg</img>请用一句话描述图中的内容。
```

---

## 第 5 步：运行推理

### 5.1 命令格式

```bash
~/Projects/demo/minicpmv-mnn/mnn/build/llm_demo <config.json> <prompt文件> <最大生成token数>
```

第三个参数是最大生成的 token 数（比如 128）。

### 5.2 文本推理

```bash
cd ~/Projects/demo/minicpmv-mnn/models/MiniCPM-V-4.7-1B-MNN
~/Projects/demo/minicpmv-mnn/mnn/build/llm_demo ./config-test-metal.json ./prompts.txt 128
```

**输出结果**：

```
北京，作为中国的政治、文化、国际交往中心，是中华文明的摇篮，承载着历史的厚重与未来的希望。

prefill speed = 26.32 tok/s
 decode speed = 233.02 tok/s
```

### 5.3 图像推理

```bash
~/Projects/demo/minicpmv-mnn/mnn/build/llm_demo ./config-test-metal.json ./prompts_img.txt 128
```

**输出结果**：

```
[MetalAttention] prefill-flash-attn-tc kernel active (seq=484, ...)
在西班牙马德里，一辆蓝色电动公交车停靠在街道旁，几位行人正在人行道上行走。

prompt tokens num = 484        # 视觉 token + 文本 token 总数
 vision time = 10.19 s          # 视觉编码耗时（最耗时的部分）
prefill time = 1.20 s
 decode time = 0.11 s
prefill speed = 402.68 tok/s
 decode speed = 190.17 tok/s
 vision speed = 0.020 MP/s
```

**说明**：
- 模型正确识别了图中内容（街景、公交车、行人），证明图像链路正常；
- `vision time`（约 10s）是整个推理的大头——因为视觉编码器要在 Metal 上跑 HQQ8 的 ViT，且首次有 tuning 预热。这是端侧 VLM 优化的关键目标之一。

---

## 第 6 步：基准测试（llm_bench）

`llm_bench` 可以分离测量 prefill（首字前处理）和 decode（逐 token 生成）的速度，这是量化前后对比的基线数据来源。

```bash
cd ~/Projects/demo/minicpmv-mnn/mnn
cmake --build build --target llm_bench -j8

cd ~/Projects/demo/minicpmv-mnn/models/MiniCPM-V-4.7-1B-MNN
~/Projects/demo/minicpmv-mnn/mnn/build/llm_bench -m ./config-test-metal.json \
  -a metal -c 1 -t 4 -p 512 -n 128 -rep 2
```

**参数说明**：`-a` 后端、`-c` 并发、`-t` 线程、`-p 512` 测 prefill 512 token、`-n 128` 测 decode 128 token、`-rep 2` 重复 2 次。

**实测结果（M5 / Metal / 语言部分 403.63 MiB）**：

```
| test  |        t/s |
| pp512 | 2296.59 ± 37.79   # prefill 吞吐
| tg128 |  201.20 ±  1.82   # decode 吞吐
```

> 这就是你做 PTQ 项目时的「量化前基线」。之后自己量化，用同样的命令测「量化后」，
> 得到一组可对比的 (速度, 内存, 精度) 数据，就是简历上最硬的部分。

---

## 结果小结

| 项 | 结果 |
|---|---|
| MNN 编译（Metal + LLM + Omni） | ✅ 成功 |
| 模型下载 | ✅ 1.5 GB |
| 纯文本推理 | ✅ 输出正确，decode 233 tok/s |
| 图像推理 | ✅ 正确描述图片内容 |
| 视觉编码耗时 | ~10 s（后续量化/优化空间所在） |
| llm_bench 基线 | prefill 2297 t/s，decode 201 t/s |

---

## 常见问题排查

1. **`Can't open file:tmp_.../mnn_cachefile.bin`**：这是 MNN 的算子 tuning 缓存，第一次运行会警告，属于正常现象，跑完会自动生成缓存文件，第二次更快。

2. **`backend_type` 不一致**：主配置写 metal、`mllm` 写 cpu，会导致视觉/语言跨设备、报错或极慢。两者必须一致。

3. **图片读不了**：确认编译时开了 `-DMNN_IMGCODECS=ON`（内置编解码）；若还不行再考虑装 OpenCV 重编。

4. **Metal 跑不起来**：确认是 Apple Silicon（M 系列），且 `cmake` 配置时 `-DMNN_METAL=ON` 确实生效（配置输出里应有 Metal 相关行）。

5. **项目目录移动后运行报 `Library not loaded: @rpath/libllm.dylib`**：这是因为 macOS 动态库的 rpath 在链接时烙死了绝对路径，移动目录后找不到 dylib。两个解决办法：
   - **推荐（无需重新编译）**：用 `install_name_tool` 加相对 rpath，以后怎么移动都不会再断：
     ```bash
     cd mnn/build
     for bin in llm_demo llm_bench; do
       for p in @executable_path @executable_path/tools/cv @executable_path/tools/audio @executable_path/express; do
         install_name_tool -add_rpath "$p" "$bin"
       done
     done
     for p in @loader_path @loader_path/tools/cv @loader_path/tools/audio @loader_path/express; do
       install_name_tool -add_rpath "$p" libllm.dylib
     done
     ```
   - **替代方案**：删除 `build/` 重新 `cmake` configure + 重新编译（编译产物重新链接后 rpath 会指向新路径）。

---

## 下一步（通往你的 PTQ 项目）

现在你跑的是**别人量化好的模型**。要变成「自己的项目」，接下来做：

1. ✅ **构建 llm_bench**（已完成，见第 6 步），量化前后用它测速得到可比数据；
2. **从原始权重导出 FP16 MNN 模型**（需要开 `-DMNN_BUILD_CONVERTER=ON`，用 `llmexport.py`）——这是自己做 PTQ 的起点；
3. **自己做 INT8/INT4 PTQ**：在导出的 FP16 模型上跑 MNN 量化，对比「你的量化版 vs 官方 HQQ 版」的精度/速度/内存；
4. **把整条链路写成可复现脚本 + 博客**，就是简历上「端侧多模态大模型量化部署」的硬项目。

---

*本文档所有命令均已在 Apple M5 / macOS 27.2 / MNN main(commit b6cb126) 上实测通过。*
