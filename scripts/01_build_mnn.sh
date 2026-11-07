#!/usr/bin/env bash
# 编译 MNN（Metal + LLM + 多模态 + 转换器）
# 产物：mnn/build/{llm_demo, llm_bench, MNNConvert}
set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# 1. 克隆 MNN（已存在则跳过）
if [ ! -d mnn ]; then
  echo "===== 克隆 MNN 源码 ====="
  git clone --depth 1 https://github.com/alibaba/MNN.git mnn
fi

# 2. CMake 配置
# 开关说明：MNN_BUILD_LLM=LLM引擎  MNN_LOW_MEMORY=低内存(端侧必需)
#          MNN_BUILD_LLM_OMNI=多模态(VLM)  MNN_IMGCODECS=内置图像编解码
#          MNN_METAL=Apple GPU  MNN_BUILD_CONVERTER=转换器(量化导出必需)
echo "===== CMake 配置 ====="
cmake -S mnn -B mnn/build \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_IMGCODECS=ON \
  -DMNN_METAL=ON \
  -DMNN_BUILD_OPENCV=OFF \
  -DMNN_BUILD_CONVERTER=ON

# 3. 编译三个工具
echo "===== 编译 llm_demo / llm_bench / MNNConvert ====="
cmake --build mnn/build --target llm_demo llm_bench MNNConvert -j8

# 4. 打相对 rpath（防止移动目录后 dyld 找不到库，macOS 专属坑）
echo "===== 修复 rpath（相对路径，移动目录不失效）====="
cd mnn/build
for bin in llm_demo llm_bench; do
  for p in @executable_path @executable_path/tools/cv @executable_path/tools/audio @executable_path/express; do
    install_name_tool -add_rpath "$p" "$bin" 2>/dev/null || true
  done
done
for p in @loader_path @loader_path/tools/cv @loader_path/tools/audio @loader_path/express; do
  install_name_tool -add_rpath "$p" libllm.dylib 2>/dev/null || true
done
cd "$PROJ_ROOT"

echo "✅ 编译完成: mnn/build/{llm_demo, llm_bench, MNNConvert}"
