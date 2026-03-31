import json
import re
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


# ─── 任务列表（与 GraphInstruct-Test 文件名对应）─────────────────────────────
TASKS = [
    'connectivity',
    'cycle',
    'shortest',
    'bipartite',
    'flow',
    'topology',
    'triangle',   # 文件名: triangle_test.json，论文里叫 Triangle
    'hamilton',
    'substructure',
]

# demo 文件名映射（task -> demos/{name}.txt）
DEMO_FILES = {
    'connectivity': 'connectivity.txt',
    'cycle':        'cycle.txt',
    'shortest':     'shortest.txt',
    'bipartite':    'bipartite.txt',
    'flow':         'flow.txt',
    'topology':     'topology.txt',
    'triangle':     'triplet.txt',
    'hamilton':     'hamilton.txt',
    'substructure': 'substructure.txt',
}


def load_demo(demos_dir: str, task: str) -> str:
    """加载 few-shot demo 文本，找不到则返回空字符串"""
    fname = DEMO_FILES.get(task)
    if not fname:
        return ""
    fpath = Path(demos_dir) / fname
    if not fpath.exists():
        print(f"[demo] 未找到: {fpath}，使用 zero-shot")
        return ""
    with open(fpath, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ─── 答案评估 ─────────────────────────────────────────────────────────────────

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    return float(res[-1][0]) if res else 0.0


def check(key, truth, predict):
    truth   = truth.lower().strip()
    predict = predict.lower().strip()
    after   = predict.split('###')[-1]

    if key in ['cycle', 'connectivity', 'bipartite', 'hamilton', 'substructure']:
        if 'yes' in truth:
            return 'yes' in after
        else:
            return 'no' in after

    elif key == 'flow':
        return abs(extract_last_num(truth) - extract_last_num(after)) < 1e-2

    elif key == 'triangle':
        return abs(extract_last_num(truth) - extract_last_num(after)) < 1e-2

    elif key == 'shortest':
        # truth 格式: "### X" 或 "the shortest path is X"
        t = truth.split('is')[-1].split('with')[0].strip().strip('.')
        return t in predict

    elif key == 'topology':
        t = truth.split('is: ')[-1].strip().strip('.')
        return t in predict

    return False


# ─── 数据加载（JSONL 格式）────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ─── 模型加载 ─────────────────────────────────────────────────────────────────

def get_model(model_path: str):
    print(f"[model] 加载: {model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 均衡分配到两张 V100，各占约 7GB
    # 'balanced' 让 accelerate 自动按层数平均分配到所有可见 GPU
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='balanced',
    )
    model.eval()

    # 打印每张卡的显存占用
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"[model] GPU {i}: {mem:.2f} GB allocated")

    return model, tokenizer


# ─── 批量推理 ─────────────────────────────────────────────────────────────────

def get_batch_llama(model, tokenizer, max_tokens: int):
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )

    @torch.inference_mode()
    def batch_llama(input_prompts: list, demo: str = "") -> list:
        # few-shot demo 拼接在 instruction 最前面
        if demo:
            queries = [f"{demo}\n\n{p}" for p in input_prompts]
        else:
            queries = input_prompts

        formatted = [PROMPT_TEMPLATE.format(query=q) for q in queries]
        enc = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
            ),
        ).tolist()

        real_output_ids = [
            out[len(enc.input_ids[i]):]
            for i, out in enumerate(output_ids)
        ]
        return tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)

    return batch_llama


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_task(task: str, args, batch_llama, save_dir: Path, demo: str = ""):
    data_file = Path(args.data_dir) / f"{task}_test.json"
    if not data_file.exists():
        print(f"[skip] 文件不存在: {data_file}")
        return None

    datas = load_jsonl(str(data_file))
    print(f"\n[{task}] 样本数: {len(datas)}, demo={'有' if demo else '无'}")

    out_file = save_dir / f"gen_{task}.jsonl"
    # 断点续跑：已跑过的跳过
    start = len(open(out_file).readlines()) if out_file.exists() else 0
    print(f"[{task}] 从第 {start} 条开始")

    # 推理
    for i in tqdm(range(start, len(datas), args.batch_size), desc=task):
        batch = datas[i: i + args.batch_size]
        prompts = [item['input_prompt'] for item in batch]
        outputs = batch_llama(prompts, demo=demo)

        with open(out_file, 'a', encoding='utf-8') as f:
            for j, (item, out) in enumerate(zip(batch, outputs)):
                json.dump({
                    'index':        i + j,
                    'input_prompt': item['input_prompt'],
                    'answer':       item['answer'],
                    'prediction':   out,
                    'task':         task,
                    'node_range':   item.get('node_range'),
                    'edge_range':   item.get('edge_range'),
                }, f, ensure_ascii=False)
                f.write('\n')

    # 评估
    records = load_jsonl(str(out_file))
    correct, wrong = [], []
    for r in records:
        ok = check(task, r['answer'], r['prediction'])
        r['is_correct'] = ok
        (correct if ok else wrong).append(r)

    total = len(correct) + len(wrong)
    acc   = len(correct) / total if total > 0 else 0
    print(f"[{task}] Accuracy = {len(correct)}/{total} = {acc:.4f}")

    # 保存分类结果
    with open(save_dir / f"{task}_correct.json", 'w', encoding='utf-8') as f:
        json.dump(correct, f, ensure_ascii=False, indent=2)
    with open(save_dir / f"{task}_wrong.json", 'w', encoding='utf-8') as f:
        json.dump(wrong, f, ensure_ascii=False, indent=2)

    return acc


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model(args.model_path)
    batch_llama = get_batch_llama(model, tokenizer, args.max_tokens)

    tasks = [args.task] if args.task else TASKS
    results = {}

    for task in tasks:
        # 加载 few-shot demo
        demo = load_demo(args.demos_dir, task) if args.demos_dir else ""
        acc = run_task(task, args, batch_llama, save_dir, demo=demo)
        if acc is not None:
            results[task] = acc

    # 汇总
    print("\n" + "=" * 50)
    print(f"{'Task':<20} {'Accuracy':>10}")
    print("-" * 32)
    for task, acc in results.items():
        print(f"{task:<20} {acc:>10.2%}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"{'Average':<20} {avg:>10.2%}")
    print("=" * 50)

    # 保存汇总 CSV
    import csv
    with open(save_dir / "summary.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Accuracy'])
        for task, acc in results.items():
            w.writerow([task, f"{acc:.4f}"])
        if results:
            w.writerow(['Average', f"{avg:.4f}"])
    print(f"\n[done] 结果保存至: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="GraphInstruct-Test 目录路径")
    parser.add_argument("--save_dir",   type=str, required=True,
                        help="结果保存目录")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size（V100 32GB 建议 4）")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="最大生成 token 数")
    parser.add_argument("--demos_dir",  type=str, default=None,
                        help="few-shot demos 目录路径，如 ../dataset/demos")
    parser.add_argument("--task",       type=str, default=None,
                        help="只跑某一个任务，不填则跑全部")
    args = parser.parse_args()
    main(args)