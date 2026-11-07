#!/usr/bin/env python3
"""文本精度评测：对四个模型跑 30 道常识题，关键词匹配判分。

用法：.venv/bin/python eval/run_text_eval.py
"""
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path("/Users/indef/Projects/demo/minicpmv-mnn")
BIN = PROJ / "mnn/build/llm_demo"
MODELS = {
    "FP16": PROJ / "models/MiniCPM-V-4.7-1B-FP16",
    "INT8": PROJ / "models/MiniCPM-V-4.7-1B-INT8",
    "HQQ4": PROJ / "models/MiniCPM-V-4.7-1B-HQQ4",
    "官方版": PROJ / "models/MiniCPM-V-4.7-1B-MNN",
}
MAX_TOKENS = "48"  # 答案都短，48 token 足够，也减少 preamble 干扰

def normalize(s: str) -> str:
    """统一大小写、处理 H₂O 下标等，便于匹配"""
    s = s.lower()
    s = s.replace("h₂o", "h2o").replace("h2o", "h2o")
    s = s.replace("２", "2").replace("１", "1")
    return s

def run_one(model_dir: Path, question: str) -> str:
    """跑单个问题，返回模型原始输出"""
    prompt_file = "/tmp/eval_prompt.txt"
    Path(prompt_file).write_text(question + "\n")
    cfg = model_dir / "config-test-metal.json"
    r = subprocess.run(
        [str(BIN), str(cfg), prompt_file, MAX_TOKENS],
        capture_output=True, text=True, timeout=120,
    )
    out = r.stdout
    # 答案在 "prompt file is" 之后、"######" 之前
    idx = out.find("prompt file is")
    if idx >= 0:
        out = out[idx:]
        out = out.split("\n", 1)[1] if "\n" in out else out
    answer = out.split("#################################")[0].strip()
    return answer

def score(answer: str, key: str) -> bool:
    """关键词匹配：参考答案（或其变体）是否出现在模型输出中"""
    ans, key = normalize(answer), normalize(key)
    return key in ans

def main():
    questions = json.loads((PROJ / "eval/text_qa.json").read_text())
    results = {}
    print(f"共 {len(questions)} 题，{len(MODELS)} 个模型\n")

    for name, mdir in MODELS.items():
        correct = 0
        detail = []
        print(f"========== {name} ==========")
        for item in questions:
            try:
                ans = run_one(mdir, item["q"])
                ok = score(ans, item["a"])
                correct += ok
                detail.append({"id": item["id"], "q": item["q"],
                               "a": item["a"], "model_ans": ans[:80], "ok": ok})
                mark = "✅" if ok else "❌"
                print(f"  [{mark}] Q{item['id']} {item['q']} → {ans[:40]}")
            except Exception as e:
                print(f"  [⚠️] Q{item['id']} 出错: {e}")
        acc = correct / len(questions)
        results[name] = {"correct": correct, "total": len(questions), "accuracy": acc}
        print(f"  → {name} 正确率: {correct}/{len(questions)} = {acc:.1%}\n")

    out = PROJ / "eval/text_eval_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"结果已保存: {out}")
    print("\n===== 汇总 =====")
    for name, r in results.items():
        print(f"  {name}: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")

if __name__ == "__main__":
    main()
