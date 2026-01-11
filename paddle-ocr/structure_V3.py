"""
https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PP-StructureV3.html
"""
import time
from pathlib import Path
from paddleocr import PPStructureV3

input_file = [
    "./input_pdf/Financial Risk.pdf", 
    "./input_pdf/国家金融监督管理总局关于 印发《银行保险机构涉刑案件风险防控 管理办法》的通知.pdf",
    "./input_pdf/国家金融监督管理总局关于印发《银行业 金融机构国别风险管理办法》的通知.pdf",
    "./input_pdf/JRT+0288—2023《银行电子凭证技术规范》0802.pdf",
]
output_path = Path("./PPStructureV3output")

"""在示例代码中，use_doc_orientation_classify、use_doc_unwarping、use_textline_orientation 
参数默认均设置为 False，分别表示关闭文档方向分类、文本图像矫正、文本行方向分类功能，
如果需要使用这些功能，可以手动设置为 True。"""
pipeline = PPStructureV3()

for name in input_file:
    t0 = time.perf_counter()
    print("="*100)
    print(f"start processing {name}")
    print("="*100)

    output = pipeline.predict(input=name)

    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)


    mkd_file_path = output_path / f"{Path(name).stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)
    t1 = time.perf_counter()
    print(f"[DONE] {mkd_file_path} | time = {t1 - t0:.3f}s")

"""
PP-StructureV3 产线使用的默认文本识别模型为 中英文识别模型，对于纯英文的识别能力有限，
对于全英文场景，您可以设置text_recognition_model_name参数为 en_PP-OCRv4_mobile_rec 
等英文识别模型以取得更好的识别效果。对应其他语言场景，也可以参考前文的模型列表，选择对
应的语言识别模型进行替换。
"""