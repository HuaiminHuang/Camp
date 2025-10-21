import json

def convert_to_sft_format(Data):
    sft_data = []
    for idx, sample in Data.items():
        sentence = sample["sentence"]
        labels = sample["labels"]

        # 构造output
        output = [
            {"text": text, "type": tag}
            for start, end, text, tag in labels
        ]
        prompt = "你是一个专业的生物医学信息抽取助手。请从用户提供的句子识别所有的实体，并输出标准 JSON。"

        sft_data.append({
            "instruction": prompt,
            "input": sentence,
            "output": json.dumps({"entities": output}, ensure_ascii=False)
        })
        
    return sft_data

def compute_f1(predictions, references, match_mode="soft"):
    assert len(predictions) == len(references)

    entity_counter = {}
    all_types = set()

    def load_json_safe(x):
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x)
        except Exception:
            return {"entities": []}

    def overlap(e1, e2, mode="lenient"):
        if e1["type"] != e2["type"]:
            return False
        if mode == "strict":
            return e1["text"] == e2["text"]
        elif mode == "soft":
            return (e1["text"] in e2["text"]) or (e2["text"] in e1["text"])
        else:
            raise ValueError(f"Unknown match_mode: {mode}")

    for pred, ref in zip(predictions, references):
        pred_json = load_json_safe(pred)
        ref_json = load_json_safe(ref)

        pred_ents = pred_json.get("entities", [])
        ref_ents = ref_json.get("entities", [])
        types = {e["type"] for e in pred_ents + ref_ents}

        for t in types:
            preds_t = [e for e in pred_ents if e["type"] == t]
            refs_t = [e for e in ref_ents if e["type"] == t]
            all_types.add(t)

            tp = 0
            for pe in preds_t:
                for re_ in refs_t:
                    if overlap(pe, re_, match_mode):
                        tp += 1
                        break

            counter = entity_counter.get(t, {"tp": 0, "pred": 0, "ref": 0})
            counter["tp"] += tp
            counter["pred"] += len(preds_t)
            counter["ref"] += len(refs_t)
            entity_counter[t] = counter

    # 汇总结果
    results = {"by_type": {}, "macro_f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0}
    precisions, recalls, f1s = [], [], []
    total_tp = total_pred = total_ref = 0

    for t, v in entity_counter.items():
        p = v["tp"] / v["pred"] if v["pred"] > 0 else 0
        r = v["tp"] / v["ref"] if v["ref"] > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        total_tp += v["tp"]
        total_pred += v["pred"]
        total_ref += v["ref"]
        results["by_type"][t] = {"p": p, "r": r, "f1": f1}

    results["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0
    
    micro_p = total_tp / total_pred if total_pred > 0 else 0
    micro_r = total_tp / total_ref if total_ref > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    results.update({"micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1})

    return results


def format_evaluation_results(metrics):
    """
    格式化评估结果为美观的输出
    """
    # 打印标题
    print("=" * 80)
    print("🎯 医学命名实体识别评估结果")
    print("=" * 80)
    
    # 按实体类型详细结果
    print("\n📊 按实体类型详细结果:")
    print("-" * 60)
    print(f"{'实体类型':<12} {'精确率(P)':<10} {'召回率(R)':<10} {'F1分数':<10}")
    print("-" * 60)
    
    by_type = metrics['by_type']
    for entity_type, scores in sorted(by_type.items()):
        p = scores['p'] * 100
        r = scores['r'] * 100
        f1 = scores['f1'] * 100
        print(f"{entity_type:<12} {p:>8.2f}% {r:>8.2f}% {f1:>8.2f}%")
    
    # 总体指标
    print("\n📈 总体评估指标:")
    print("-" * 40)
    macro_f1 = metrics['macro_f1'] * 100
    micro_p = metrics['micro_precision'] * 100
    micro_r = metrics['micro_recall'] * 100
    micro_f1 = metrics['micro_f1'] * 100
    
    print(f"🏆 宏平均F1:     {macro_f1:>8.2f}%")
    print(f"🎯 微平均精确率: {micro_p:>8.2f}%")
    print(f"🔍 微平均召回率: {micro_r:>8.2f}%")
    print(f"⚡ 微平均F1:     {micro_f1:>8.2f}%")
    
    # 性能分析
    print("\n📋 性能分析:")
    print("-" * 40)
    
    # 找出表现最好和最差的实体类型
    f1_scores = {etype: scores['f1'] for etype, scores in by_type.items()}
    best_entity = max(f1_scores.items(), key=lambda x: x[1])
    worst_entity = min(f1_scores.items(), key=lambda x: x[1])
    
    print(f"✅ 最佳表现: {best_entity[0]} (F1: {best_entity[1]*100:.2f}%)")
    print(f"❌ 最差表现: {worst_entity[0]} (F1: {worst_entity[1]*100:.2f}%)")
    
    # 统计不同F1分数区间的实体类型数量
    excellent = sum(1 for f1 in f1_scores.values() if f1 >= 0.8)
    good = sum(1 for f1 in f1_scores.values() if 0.6 <= f1 < 0.8)
    poor = sum(1 for f1 in f1_scores.values() if f1 < 0.6)
    
    print(f"\n📈 F1分数分布:")
    print(f"  优秀 (≥80%): {excellent} 个类型")
    print(f"  良好 (60-80%): {good} 个类型")
    print(f"  需改进 (<60%): {poor} 个类型")
    
    print("=" * 80)

    import torch
import numpy as np

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=-1)

    preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds.cpu().tolist()]
    labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels.cpu().tolist()]

    res = compute_f1(preds, labels)
    return {
        "macro_f1": res["macro_f1"], 
        "micro_precision": res["micro_precision"], 
        "micro_recall": res["micro_recall"], 
        "micro_f1": res["micro_f1"],
    }
