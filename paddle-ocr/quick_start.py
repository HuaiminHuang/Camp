import paddle

print(paddle.__version__)

# Initialize PaddleOCR instance
def  PP_OCRv5_Example():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Run OCR inference on a sample image 
    result = ocr.predict(
        input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

    # Visualize the results and save the JSON results
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")

# 这里需要根据提示进行安装VL模型
# pip install paddlex[ocr] <opntional: version>
def Paddle_OCR_VL():
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL()
    output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")
    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output")

if __name__ == "__main__":
    # PP_OCRv5_Example()
    Paddle_OCR_VL()
