"""
https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html
## 这里的默认设置没有进行优化，请使用vllm等加速推理框架进行使用
"""

from pathlib import Path
from paddleocr import PaddleOCRVL

input_file = [
    # "./input_pdf/Financial Risk.pdf", 
    "./input_pdf/国家金融监督管理总局关于 印发《银行保险机构涉刑案件风险防控 管理办法》的通知.pdf",
    "./input_pdf/国家金融监督管理总局关于印发《银行业 金融机构国别风险管理办法》的通知.pdf",
    "./input_pdf/JRT+0288—2023《银行电子凭证技术规范》0802.pdf",
]
output_path = Path("./VLoutput")

# 英伟达 GPU
pipeline = PaddleOCRVL()
# 昆仑芯 XPU
# pipeline = PaddleOCRVL(device="xpu")
# 海光 DCU
# pipeline = PaddleOCRVL(device="dcu")
# 沐曦 GPU
# pipeline = PaddleOCRVL(device="metax_gpu")

output = pipeline.predict(input=input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)