import os
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


class StorageManager:
    def __init__(self, db_name: str, read_only: bool = False):
        self.db_name = db_name
        self.base_path = f"/app/deeplake_data"
        self.path_to_db = f"{self.base_path}/{db_name}"
        self.vector_store = DeepLakeVectorStore(
            dataset_path=self.path_to_db, overwrite=False, read_only=read_only
        )
        self.paths = {
            "docstore": f"{self.base_path}/{db_name}_docstore",
            "tree": f"{self.base_path}/{db_name}_tree_indexstore",
            "vector": f"{self.base_path}/{db_name}_vector_indexstore",
            "kg": f"{self.base_path}/{db_name}_kg_storage",
            "bm25": f"{self.base_path}/{db_name}_bm25_storage",
            "memory": f"{self.base_path}/memory",
        }

    def build_context(self, index_type: str) -> StorageContext:
        idx_path = self.paths[index_type]
        doc_file = os.path.join(self.paths["docstore"], "docstore.json")
        idx_file = os.path.join(idx_path, "index_store.json")

        os.makedirs(self.paths["docstore"], exist_ok=True)
        os.makedirs(idx_path, exist_ok=True)

        return StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=(
                SimpleDocumentStore.from_persist_dir(self.paths["docstore"])
                if os.path.exists(doc_file)
                else SimpleDocumentStore()
            ),
            index_store=(
                SimpleIndexStore.from_persist_dir(idx_path)
                if os.path.exists(idx_file)
                else SimpleIndexStore()
            ),
        )

    def persist_context(self, ctx: StorageContext, index_type: str) -> None:
        idx_path = self.paths[index_type]
        doc_file = os.path.join(self.paths["docstore"], "docstore.json")
        idx_file = os.path.join(idx_path, "index_store.json")
        ctx.docstore.persist(persist_path=doc_file)
        ctx.index_store.persist(persist_path=idx_file)
