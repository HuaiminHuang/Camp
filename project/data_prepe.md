# 数据处理部分
1. 构建数据的索引：
    - 读入PFDF文件，对数据进行一个基本的清洗（去除无效部分：页眉目录等）并且关联图片和文本。并且使用hash md5 生成 unique id作为 meta 信息
    ```python
    def handle_image(img: Tuple, img_index: int, page: fitz.Page) -> ManualImages | None:
        """处理单个图片"""
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        ...
        # 跳过小图标
        # 保存图片并获取路径
        # 获取扩展后的图片区域
        # 获取关联文本块
        related_blocks = get_related_text_blocks(page, expanded_rect, img_rect.y0)
        title_blocks = [text for is_title, text in related_blocks if is_title]

        return ManualImages(
            image_path=image_path,
            page=page.number + 1,
            title="\n".join(title_blocks)
        )
    ```    
    
    - 借助LLM进行文档的规整和清洗（`llm_clean_client.py`），更方便后续的清洗逻辑
    ```py
        LLM_CLEAN_PROMPT = """
    你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

    1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
    2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。

    请根据以下文档内容进行整理：
    {}
    整理后的输出：
    """
    ```
    - 在进行文档的切分，这里采用的是语义切分加滑动窗口切分，并且关联上父子文档（防止语义被二次切碎）`utils.py`:
    ```py
    def texts_split(raw_docs):
        """语义切分 + 父子文档层次构建（简化示意版）"""
        all_split_docs = []

        for doc in raw_docs:
            # === 1. 语义级切分 ===
            semantic_groups = semantic_split(doc.page_content)

            parent_docs = []
            for group in semantic_groups:
                parent_doc = Document(page_content=group, metadata={"type": "parent"})
                parent_docs.append(parent_doc)
                all_split_docs.append(parent_doc)

            # === 2. 句子级再切分（子文档）===
            for parent in parent_docs:
                child_chunks = sentence_split(parent.page_content)
                for chunk in child_chunks:
                    child_doc = Document(
                        page_content=chunk,
                        metadata={"type": "child", "parent": parent},
                    )
                    all_split_docs.append(child_doc)

        return all_split_docs
    ```
    - 最后为索引入库

## 生成qa数据集
使用Deepseek进行加载的 PDF 文档进行生成式数据抽取，根据提示词要求模型生成 **Json** 格式。这样就获得了一个原始的数据集。
```py 
CONTEXT_PROMPT_TPL = """
我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成5个问题，和基于这段文本对问题的回答，回答请保持完整，无须重复问题。

对问题、答案的要求：
1.问题：问题要与这段文本相关，不要询问类似“这个问题的答案在哪一章”这样的问题;
2.答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用其他章节和页码，例如答案内容不能出现请参阅xx页码;
3.5个问题里面至少要包含一个需要综合*大段*文本才能回答的问题，但不要问类似“这一段主要讲了什么内容”这样的问题;

对输出的要求：
1.返回结果以JSON形式组织，格式为[{"question": "...", "answer": "..."}, ...]。
2.如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。

下方是文本：
<document>
{{document}}
</document>

请生成结果：
"""
```

但是对于这个数据样本还是比较少，接着再次进行现有数据集问题的拓展增强数据集的泛化性和丰富度，涵盖更广的场景（例如一些口语化场景等）。这一词同样使用了Deepseek作为生成助手，对问题进行拓展。







