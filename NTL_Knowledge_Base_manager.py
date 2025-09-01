import getpass
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PythonLoader
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PythonLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RAGDatabase:
    def __init__(self, persistent_directory, collection_name="knowledge-chroma"):
        """
        初始化 RAG 数据库
        :param persistent_directory: 持久化存储目录
        :param collection_name: Chroma 集合名称
        """
        self.persistent_directory = persistent_directory
        self.collection_name = collection_name

        os.makedirs(persistent_directory, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2048, chunk_overlap=512
        )
        self.embeddings = OpenAIEmbeddings()

        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persistent_directory,
                embedding_function=self.embeddings
            )
            print(f"成功加载现有 RAG 知识库，当前文档数：{self.vector_store._collection.count()}")
        except Exception as e:
            print(f"加载现有知识库失败：{e}")
            self.vector_store = None

    def create_database(
        self,
        url_list=None,
        pdf_folder=None,
        json_folder=None,
        py_folder=None,
        txt_folder=None
    ):
        """
        初始化数据库并加载指定的文档。
        """
        knowledge_docs_list = []

        # 在线文档
        if url_list:
            docs = [WebBaseLoader(url).load() for url in url_list]
            knowledge_docs_list.extend([item for sublist in docs for item in sublist])

        # PDF
        if pdf_folder:
            for file_name in os.listdir(pdf_folder):
                if file_name.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_folder, file_name)
                    loader = PyPDFLoader(pdf_path)
                    knowledge_docs_list.extend(loader.load())

        # JSON（按 task_id 拆分）
        if json_folder:
            for file_name in os.listdir(json_folder):
                if file_name.endswith(".json"):
                    json_path = os.path.join(json_folder, file_name)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # 如果是任务列表
                        if isinstance(data, list):
                            for task in data:
                                page_content = json.dumps(task, ensure_ascii=False, indent=2)
                                metadata = {"source": json_path, "task_id": task.get("task_id")}
                                knowledge_docs_list.append(Document(page_content=page_content, metadata=metadata))
                        else:
                            # 其他 JSON 直接存为一条文档
                            page_content = json.dumps(data, ensure_ascii=False, indent=2)
                            knowledge_docs_list.append(Document(page_content=page_content, metadata={"source": json_path}))
                    except Exception as e:
                        print(f"读取 {json_path} 失败，跳过。原因：{e}")

        # Python
        if py_folder:
            for file_name in os.listdir(py_folder):
                if file_name.endswith(".py"):
                    py_path = os.path.join(py_folder, file_name)
                    try:
                        knowledge_docs_list.extend(PythonLoader(py_path).load())
                    except Exception as e:
                        print(f"读取 {py_path} 失败，跳过。原因：{e}")

        # TXT
        if txt_folder:
            for file_name in os.listdir(txt_folder):
                if file_name.endswith(".txt"):
                    txt_path = os.path.join(txt_folder, file_name)
                    try:
                        knowledge_docs_list.extend(TextLoader(txt_path, encoding="utf-8").load())
                    except Exception as e:
                        try:
                            knowledge_docs_list.extend(TextLoader(txt_path, encoding="gbk").load())
                        except Exception as e2:
                            print(f"读取 {txt_path} 失败，跳过。原因：{e2}")

        print(f"共加载文档数量：{len(knowledge_docs_list)}")
        for i, doc in enumerate(knowledge_docs_list[:3]):
            print(f"文档 {i + 1} 预览：{doc.page_content[:100]}")

        # 文本拆分（JSON 按 task_id 已经拆过，不再切）
        non_json_docs = [doc for doc in knowledge_docs_list if not doc.metadata.get("task_id")]
        json_task_docs = [doc for doc in knowledge_docs_list if doc.metadata.get("task_id")]

        knowledge_splits = self.text_splitter.split_documents(non_json_docs) + json_task_docs
        print(f"文档切分后总块数：{len(knowledge_splits)}")

        # 写入 Chroma
        batch_size = 500
        for i in range(0, len(knowledge_splits), batch_size):
            batch = knowledge_splits[i:i + batch_size]
            if i == 0 or self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    persist_directory=self.persistent_directory,
                )
            else:
                self.vector_store.add_documents(batch)

        print("RAG 知识库已初始化并保存到磁盘。")








    def add_documents(self, folder_path):
        """
        向现有知识库批量添加新文档，支持 PDF、TXT、PY、JSON 文件，自动编码容错，并分批写入防止 token 超限。
        :param folder_path: 包含待添加文档的文件夹路径
        """
        if not self.vector_store:
            raise ValueError("请先初始化 RAG 数据库。")

        new_docs = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if file_name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    new_docs.extend(loader.load())
                elif file_name.endswith(".txt"):
                    try:
                        new_docs.extend(TextLoader(file_path, encoding="utf-8").load())
                    except Exception as e:
                        try:
                            new_docs.extend(TextLoader(file_path, encoding="gbk").load())
                        except Exception as e2:
                            print(f"读取 {file_path} 失败，跳过。原因：{e2}")
                elif file_name.endswith(".py"):
                    try:
                        new_docs.extend(PythonLoader(file_path).load())
                    except Exception as e:
                        print(f"读取 {file_path} 失败，跳过。原因：{e}")
                elif file_name.endswith(".json"):
                    try:
                        new_docs.extend(TextLoader(file_path, encoding="utf-8").load())
                    except Exception as e:
                        try:
                            new_docs.extend(TextLoader(file_path, encoding="gbk").load())
                        except Exception as e2:
                            print(f"读取 {file_path} 失败，跳过。原因：{e2}")
                else:
                    continue  # 跳过不支持的文件类型
            except Exception as e:
                print(f"处理 {file_path} 发生异常，已跳过。原因：{e}")

        if not new_docs:
            print("未找到可添加的新文档。")
            return

        # 拆分新文档
        new_splits = self.text_splitter.split_documents(new_docs)

        # 分批写入，防止token超限
        batch_size = 200
        total = len(new_splits)
        for i in range(0, total, batch_size):
            batch = new_splits[i:i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"已添加文档分块 {i + 1} - {min(i + batch_size, total)} 到知识库。")

        print(f"已添加 {len(new_docs)} 个新文档（共 {len(new_splits)} 个分块）到知识库，并保存到磁盘。")




class ChromaManager:
    def __init__(self, persistent_directory, collection_name="knowledge-chroma"):
        """
        初始化 Chroma 数据库管理器
        :param persistent_directory: 持久化存储目录
        :param collection_name: Chroma 集合名称
        """
        self.persistent_directory = persistent_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()

        # 确保目录存在
        os.makedirs(persistent_directory, exist_ok=True)

        # 加载数据库
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=persistent_directory,
                embedding_function=self.embeddings
            )
            print(f"成功加载知识库 '{collection_name}'")
        except Exception as e:
            print(f"加载知识库失败: {e}")
            self.vector_store = None

    def list_documents(self, limit=None):
        """
        列出数据库中的文档
        :param limit: 限制返回的文档数量
        """
        if not self.vector_store:
            print("知识库未初始化")
            return

        # 获取所有文档的 ID、内容和元数据
        ids = self.vector_store._collection.get(include=['documents', 'metadatas'])

        print(f"总文档数: {len(ids['ids'])}")

        # 如果设置了限制，只显示部分文档
        if limit:
            ids['ids'] = ids['ids'][:limit]
            ids['documents'] = ids['documents'][:limit]
            ids['metadatas'] = ids['metadatas'][:limit]

        # 打印文档详情
        for i, (doc_id, doc, metadata) in enumerate(zip(ids['ids'], ids['documents'], ids['metadatas']), 1):
            preview = doc[:100] + ("..." if len(doc) > 100 else "")
            print(f"\n文档 {i}:")
            print(f"ID: {doc_id}")
            print(f"内容预览: {preview}")
            print(f"元数据: {metadata}")

    def delete_documents(self, ids=None, filter_metadata=None):
        """
        删除文档
        :param ids: 要删除的文档 ID 列表
        :param filter_metadata: 根据元数据过滤并删除文档
        """
        if not self.vector_store:
            print("知识库未初始化")
            return

        if not ids and not filter_metadata:
            print("请提供要删除的文档 ID 或元数据过滤条件")
            return

        if ids:
            self.vector_store._collection.delete(ids=ids)
            print(f"已删除 {len(ids)} 个文档")

        if filter_metadata:
            delete_ids = self.vector_store._collection.get(
                where=filter_metadata,
                include=['ids']
            )['ids']

            if delete_ids:
                self.vector_store._collection.delete(ids=delete_ids)
                print(f"根据元数据删除了 {len(delete_ids)} 个文档")
            else:
                print("未找到符合条件的文档")

    def delete_all_documents(self):
        """
        删除知识库中的所有文档，并提供详细的操作反馈
        """
        try:
            # 检查知识库是否初始化
            if not self.vector_store:
                print("错误：知识库未初始化")
                return False

            # 获取所有文档的 ID
            collection_info = self.vector_store._collection.get(include=['metadatas'])

            # 检查是否有文档
            if not collection_info or 'ids' not in collection_info:
                print("知识库中没有文档需要删除")
                return True

            ids = collection_info['ids']

            # 检查文档数量
            if not ids:
                print("知识库中没有文档需要删除")
                return True

            # 执行删除操作
            try:
                self.vector_store._collection.delete(ids=ids)
                print(f"成功删除所有文档，共计 {len(ids)} 个")
                return True
            except Exception as delete_error:
                print(f"删除文档时发生错误：{delete_error}")
                return False

        except Exception as e:
            print(f"删除文档过程中发生未知错误：{e}")
            return False

    def search_documents(self, query, k=5):
        """
        语义搜索文档
        :param query: 搜索查询
        :param k: 返回的相似文档数量
        """
        if not self.vector_store:
            print("知识库未初始化")
            return

        results = self.vector_store.similarity_search(query, k=k)

        if not results:
            print(f"未找到与查询 '{query}' 相关的文档")
            return

        print(f"搜索查询: {query}")
        print(f"找到 {len(results)} 个相关文档:")

        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            print(f"\n相关文档 {i}:")
            print(f"内容预览: {preview}")
            print(f"元数据: {doc.metadata}")

    def count_documents(self):
        """
        统计文档数量
        """
        if not self.vector_store:
            print("知识库未初始化")
            return

        count = self.vector_store._collection.count()
        print(f"知识库 '{self.vector_store._collection.name}' 中共有 {count} 个文档")

# 使用示例
if __name__ == "__main__":
    # 指定持久化路径
    # persistent_directory1 = r"C:\NTL_Agent\RAG\Literature_RAG"
    # collection_name1 = "Literature_RAG"
    # manager1 = ChromaManager(persistent_directory1, collection_name1)
    # # 列出文档
    # # print("\n文档列表:")
    # # manager1.list_documents(limit=20)
    #
    import os
    import shutil

    persistent_directory2 = r"E:\NTL_Agent\RAG\Solution_RAG"

    # Remove all files and subfolders inside the directory, but keep the directory itself
    for item in os.listdir(persistent_directory2):
        item_path = os.path.join(persistent_directory2, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"Cleared all contents inside: {persistent_directory2}")
    collection_name2 = "Solution_RAG"
    manager2 = ChromaManager(persistent_directory2, collection_name2)
    #
    # persistent_directory3 = r"C:\NTL_Agent\RAG\Code_RAG"
    # collection_name3 = "Code_RAG"
    # manager3 = ChromaManager(persistent_directory3, collection_name3)
    #
    # # 初始化 RAG 数据库
    # rag_db1 = RAGDatabase(persistent_directory1, collection_name1)
    rag_db2 = RAGDatabase(persistent_directory2, collection_name2)
    # print("\n文档列表:")
    # manager2.list_documents(limit=20)
    # rag_db3 = RAGDatabase(persistent_directory3, collection_name2)
    #
    # 创建新的数据库
    # urls = ["https://rasterio.readthedocs.io"]
    json_folder = r"E:\NTL_Agent\workflow"
    # json_folder2 = r"C:\NTL-CHAT\tool\RAG\code_guide\GEE_dataset"
    # py_folder = r"C:\NTL-CHAT\tool\RAG\code_guide\Geospatial_Code_GEE"
    # txt_folder = r"C:\NTL-CHAT\tool\RAG\code_guide\Geospatial_Code_geopanda_rasterio"
    # pdf_folder = r"C:\NTL-CHAT\tool\RAG\文献查找\综述"  # PDF 文件夹路径

    # rag_db1.create_database(pdf_folder=pdf_folder)
    rag_db2.create_database(json_folder = json_folder)
    # rag_db3.create_database(py_folder=py_folder, txt_folder=txt_folder, json_folder= json_folder2)
    # 添加新文档
    # new_doc_folder = r"C:\NTL-CHAT\tool\RAG\add_document"
    # rag_db.add_documents(folder_path=new_doc_folder)
    # 统计文档数量
    # print("\n统计文档数量:")
    # manager1.count_documents()
    # manager2.count_documents()
    # manager3.count_documents()
    # query = "如何下载2020年上海市的夜间灯光遥感影像"
    # manager1.search_documents(query=query, k=3)
    # manager2.search_documents(query=query, k=3)


    # 列出文档
    # print("\n文档列表:")
    # manager2.list_documents(limit=20)

    # 删除所有文档
    # print("\n删除所有文档:")
    # manager.delete_all_documents()

    # 验证是否成功删除
    # print("\n统计文档数量:")
    # manager.count_documents()

    # 搜索文档
    # print("\n文档搜索:")
    # query = "夜间灯光遥感"
    # manager.search_documents(query=query, k=3)

    # 统计文档数量
    # print("\n统计文档数量:")
    # manager.count_documents()

    # 删除文档（根据 ID 或元数据）
    # print("\n删除文档:")
    # manager.delete_documents(ids=["example_doc_id"])

    # 删除文档（根据元数据过滤条件）
    # print("\n根据元数据删除文档:")
    # manager.delete_documents(filter_metadata={"source": "pdf"})
