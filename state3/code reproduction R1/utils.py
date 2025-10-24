from typing import Optional

from datasets import IterableDataset
from datasets import load_dataset

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""


def extract_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_cot(text: str) -> str:
    if "####" not in text:
        return ""
    cot = text.split("####")
    return XML_COT_FORMAT.format(think=cot[0].strip(), answer=cot[1].strip())


def get_gsm8k_dataset(split="train", sft=False, cache_dir=None, first_half=False, second_half=False) -> IterableDataset:
    # 这里使用hf的 dataset 进行安装，并且指定分支为 main
    data = load_dataset('data/gsm8k', "main", split=split, cache_dir=cache_dir)

    if first_half:
        data = data.select(range(0, len(data)//2))
    elif second_half:
        data = data.select(range(len(data)//2, len(data)))

    if not sft:
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_answer(x['answer'])
        })
    else:
        data = data.map(lambda x: {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']},
                {'role': 'assistant', 'content': extract_cot(x['answer'])},
            ]
        })
    return data
