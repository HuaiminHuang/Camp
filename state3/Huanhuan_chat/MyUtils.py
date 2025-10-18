import os
import json
from collections import defaultdict

# exmaple path = "./state3/Huanhuan_chat/chat_json/甄嬛传剧本01-10_dialogues_concurrent.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

    data = list(load_jsonl(path))
    return data

def proc_func(data: list, up_nums: int) -> list:
    """
    args:
        data: 需要处理的数据
        up_num: 需要传输的上下文
    return:
        sample like [{'role': '', 'content': ''}])
    """
    scenes = defaultdict(list)
    roles = {item["role"] for item in data}

    for d in data:
        scenes[d["chunk_id"]].append(d)

        samples = []
    up_nums = 3
    key_roles = list(roles)

    for chunk_id, lines in scenes.items():
        for i, line in enumerate(lines):
            if line["role"] == "甄嬛":
                # print(line)
                context_lines = []
                for j in range(max(0, i - up_nums), i):
                    print(lines[j]["role"], lines[j]["dialogue"])
                    r = lines[j]["role"]
                    d = lines[j]["dialogue"]
                    if key_roles and r not in key_roles and r != "甄嬛":
                        continue
                    context_lines.append(f"{r}:{d}")
                print("context_lines", context_lines)

                input_text = "\n".join(context_lines)
                output_text = line["dialogue"]

                messages = [
                    {"role": "system", "content": "你是甄嬛，深谙宫闱权谋，言辞婉转含蓄，心思缜密，进退有度。你以柔克刚、以智取胜，既有诗书才情，亦怀果决狠厉。言语间常引经据典，暗藏机锋，表面温婉谦和，内里清醒自持。"},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text}
                ]
                samples.append({"messages": messages})

    return samples