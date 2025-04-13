
import os
import hashlib
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.indexes import SQLRecordManager
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, RePhraseQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseLanguageModel

# 自定义模块：用于对接本地微调后的 Qwen2.5 模型
from my_local_qwen import LocalQwenLLM

# 系统角色 prompt，用于限定回答语气和风格
SYSTEMPL = """你是一名智能医疗数字人，帮助用户进行医疗健康咨询，请结合上下文提供准确的医疗信息，避免夸大、不确定、不科学的建议。
{context}
"""

# QA prompt 模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEMPL),
        ("human", "{input}"),
    ]
)

# 知识库存储路径
KNOWLEDGE_DIR = './chroma/knowledge/'
# 向量化模型路径
embedding_model = './BAAI/bge-large-zh-v1.5'
# rerank 模型路径
rerank_model = './BAAI/bge-reranker-large'

# 自定义文档加载器：支持自动识别文件类型并切分
class MyCustomLoader(BaseLoader):
    file_type = {
        "pdf": ("PyPDFLoader", {}),
        "txt": ("TextLoader", {}),
        "doc": ("UnstructuredWordDocumentLoader", {}),
        "docx": ("UnstructuredWordDocumentLoader", {}),
        "md": ("UnstructuredMarkdownLoader", {}),
        "csv": ("CSVLoader", {"autodetect_encoding": True}),
    }

    def __init__(self, file_path: str):
        from unstructured.file_utils.filetype import detect_filetype, FileType
        loader_class, params = self.file_type[detect_filetype(file_path).name.lower()]
        self.loader: BaseLoader = loader_class(file_path, **params)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=200,
            chunk_overlap=60,
            length_function=len,
        )

    def load(self):
        return self.loader.load_and_split(self.text_splitter)

# 获取字符串的 MD5 值，用于统一知识库标识
def get_md5(input_string):
    return hashlib.md5(input_string.encode('utf-8')).hexdigest()

# 创建文档索引并生成混合检索器
def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    db = Chroma(collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=os.path.join('./chroma', collection_name))

    # 创建记录管理器（避免重复索引）
    record_manager = SQLRecordManager(f"chromadb/{collection_name}", db_url="sqlite:///record_manager_cache.sql")
    record_manager.create_schema()

    # 加载并切分文档
    documents = loader.load()

    # 将文档存入向量数据库
    index(documents, record_manager, db, cleanup="full", source_id_key="source")

    # 使用向量检索 + 关键词 BM25 检索融合
    ensemble_retriever = EnsembleRetriever(
        retrievers=[db.as_retriever(search_kwargs={"k": 3}), BM25Retriever.from_documents(documents)]
    )

    return ensemble_retriever

# 知识库管理类
class MyKnowledge:
    __embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)
    __retrievers = {}
    __llm = LocalQwenLLM()  # 本地模型实例

    os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)
    collections = [None]

    # 遍历知识库目录，为每个文件构建索引和检索器
    for file in os.listdir(KNOWLEDGE_DIR):
        collections.append(file)
        collection_name = get_md5(file)
        file_path = os.path.join(KNOWLEDGE_DIR, file)
        loader = MyCustomLoader(file_path)
        __retrievers[collection_name] = create_indexes(collection_name, loader, __embeddings)

    # 获取对应知识库的压缩型检索器（包含 rerank）
    def get_retrievers(self, collection):
        collection_name = get_md5(collection)
        if collection_name not in self.__retrievers:
            return None
        retriever = self.__retrievers[collection_name]

        # 文档 rerank 排序器：提升语义相关性
        model = HuggingFaceCrossEncoder(model_name=rerank_model)
        reranker = CrossEncoderReranker(model=model, top_n=3)

        # 返回压缩型检索器（结合原检索器 + reranker）
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever
        )
        return retriever

# 问答类，结合 retriever + LLM 构建 RAG chain
class MyLLM(MyKnowledge):
    def get_chain(self, collection, model=None, max_length=512, temperature=0):
        retriever = self.get_retrievers(collection)
        question_answer_chain = create_stuff_documents_chain(self.__llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    # 执行问答调用
    def invoke(self, question, collection, model="qwen2.5-7b", max_length=512, temperature=0):
        return self.get_chain(collection, model, max_length, temperature).invoke({"input": question})
