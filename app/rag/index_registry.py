import os
import pickle
import re
import shutil
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.indices.tree import TreeIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.retrievers.bm25 import BM25Retriever

from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger

from .storage_manager import StorageManager


class IndexRegistry:
    def __init__(self, storage: StorageManager, prompts: dict):
        self.storage = storage
        self.prompts = prompts
        self.indices = {"vector": None, "tree": None, "kg": None, "bm25": None}
        self._morph_vocab = MorphVocab()
        self._segmenter = Segmenter()
        self._morph_emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._morph_emb)
        self._line_word_break_re = re.compile(
            r"(?<=[A-Za-zА-Яа-яЁё])\n(?=[A-Za-zА-Яа-яЁё])"
        )
        self._glued_sentence_re = re.compile(r"([)\].,;:!?])([A-ZА-ЯЁ])")
        self._lower_upper_glue_re = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")

    def _get_parser(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=384, chunk_overlap=48, paragraph_separator="\n\n"
        )

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00ad", "")
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = self._line_word_break_re.sub(" ", text)
        text = self._glued_sentence_re.sub(r"\1 \2", text)
        text = self._lower_upper_glue_re.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_batch(self, batch: list) -> list:
        for doc in batch:
            if getattr(doc, "text", None):
                doc.text = self._normalize_text(doc.text)
        return batch

    def _iter_normalized_batches(self, docs: SimpleDirectoryReader):
        for batch in docs.iter_data():
            yield self._normalize_batch(batch)

    def _reset_index(self, index_type: str) -> None:
        if index_type not in self.indices:
            raise ValueError(
                f"Неизвестный index_type: {index_type}. Доступны: tree, vector, kg, bm25"
            )
        self.indices[index_type] = None
        index_path = self.storage.paths[index_type]
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)

    def add(self, index_type: str, path_to_docs: str, overwrite: bool = False) -> None:
        if overwrite:
            self._reset_index(index_type)

        docs = SimpleDirectoryReader(path_to_docs, recursive=True)
        parser = self._get_parser()

        if index_type == "bm25":
            self._build_bm25(docs, parser)
            return

        ctx = self.storage.build_context(index_type)
        if index_type == "tree":
            self._build_tree(ctx, docs, parser)
        elif index_type == "vector":
            self._build_vector(ctx, docs, parser)
        elif index_type == "kg":
            self._build_kg(ctx, docs, parser)
        else:
            raise ValueError(
                f"Неизвестный index_type: {index_type}. Доступны: tree, vector, kg, bm25"
            )
        self.storage.persist_context(ctx, index_type)

    def _build_tree(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser: SentenceSplitter
    ) -> None:
        docs_iter = self._iter_normalized_batches(docs)
        if os.path.exists(os.path.join(self.storage.paths["tree"], "index_store.json")):
            self.indices["tree"] = load_index_from_storage(ctx, index_id="tree_index")
            for batch in docs_iter:
                for d in batch:
                    self.indices["tree"].insert(d)
        else:
            if first_doc := next(docs_iter, None):
                self.indices["tree"] = TreeIndex.from_documents(
                    first_doc,
                    storage_context=ctx,
                    transformations=[parser],
                    summary_template=self.prompts["summary_prompt"],
                )
                self.indices["tree"].set_index_id("tree_index")
                for batch in docs_iter:
                    for d in batch:
                        self.indices["tree"].insert(d)

    def _build_vector(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser: SentenceSplitter
    ) -> None:
        docs_iter = self._iter_normalized_batches(docs)
        if os.path.exists(
            os.path.join(self.storage.paths["vector"], "index_store.json")
        ):
            self.indices["vector"] = load_index_from_storage(
                ctx, index_id="vector_index"
            )
            for batch in docs_iter:
                for d in batch:
                    self.indices["vector"].insert(d)
        else:
            if first_doc := next(docs_iter, None):
                self.indices["vector"] = VectorStoreIndex.from_documents(
                    first_doc, storage_context=ctx, transformations=[parser]
                )
                self.indices["vector"].set_index_id("vector_index")
                for batch in docs_iter:
                    for d in batch:
                        self.indices["vector"].insert(d)

    def _build_kg(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser: SentenceSplitter
    ) -> None:
        docs_iter = self._iter_normalized_batches(docs)
        kg_path = self.storage.paths["kg"]
        if os.path.exists(os.path.join(kg_path, "graph_store.json")):
            self.indices["kg"] = load_index_from_storage(ctx, index_id="kg_index")
            for batch in docs_iter:
                for d in batch:
                    self.indices["kg"].insert(d)
        else:
            if first_doc := next(docs_iter, None):
                self.indices["kg"] = KnowledgeGraphIndex.from_documents(
                    first_doc,
                    storage_context=ctx,
                    transformations=[parser],
                    llm=Settings.llm,
                    embed_model=Settings.embed_model,
                    kg_triplet_extract_template=self.prompts[
                        "kg_triplet_extract_template"
                    ],
                    max_triplets_per_chunk=int(
                        os.getenv("KG_MAX_TRIPLETS_PER_CHUNK", "8")
                    ),
                    include_embeddings=True,
                )
                self.indices["kg"].set_index_id("kg_index")
                for batch in docs_iter:
                    for d in batch:
                        self.indices["kg"].insert(d)
                ctx.persist(persist_dir=kg_path)

    def _build_bm25(
        self, docs: SimpleDirectoryReader, parser: SentenceSplitter
    ) -> None:
        nodes = []
        for batch in self._iter_normalized_batches(docs):
            nodes.extend(parser.get_nodes_from_documents(batch))

        if not nodes:
            raise RuntimeError("Не найдено документов для построения BM25 индекса.")

        self.indices["bm25"] = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=10,
            tokenizer=self._ru_lemmatize,
        )

        bm25_path = self.storage.paths["bm25"]
        os.makedirs(bm25_path, exist_ok=True)
        self._persist_bm25_nodes(nodes, bm25_path)

    def _load_persisted(self, index_type: str):
        if index_type == "vector":
            idx_file = os.path.join(self.storage.paths["vector"], "index_store.json")
            if os.path.exists(idx_file):
                ctx = self.storage.build_context("vector")
                return load_index_from_storage(ctx, index_id="vector_index")
        elif index_type == "tree":
            idx_file = os.path.join(self.storage.paths["tree"], "index_store.json")
            if os.path.exists(idx_file):
                ctx = self.storage.build_context("tree")
                return load_index_from_storage(ctx, index_id="tree_index")
        elif index_type == "kg":
            kg_path = self.storage.paths["kg"]
            has_kg = all(
                os.path.exists(os.path.join(kg_path, filename))
                for filename in (
                    "docstore.json",
                    "index_store.json",
                    "graph_store.json",
                )
            )
            if has_kg:
                ctx = StorageContext.from_defaults(persist_dir=kg_path)
                return load_index_from_storage(ctx, index_id="kg_index")
        elif index_type == "bm25":
            bm25_path = self.storage.paths["bm25"]
            nodes = self._load_bm25_nodes(bm25_path)
            if nodes:
                return BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=10,
                    tokenizer=self._ru_lemmatize,
                )
        return None

    def _persist_bm25_nodes(self, nodes: list, bm25_path: str) -> None:
        with open(os.path.join(bm25_path, "bm25_nodes.pkl"), "wb") as f:
            pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bm25_nodes(self, bm25_path: str) -> list:
        nodes_file = os.path.join(bm25_path, "bm25_nodes.pkl")
        if not os.path.exists(nodes_file):
            return []
        with open(nodes_file, "rb") as f:
            return pickle.load(f)

    def _ru_lemmatize(self, text: str) -> list[str]:
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        lemmas: list[str] = []
        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)
            lemma = token.lemma.lower()
            if lemma and not lemma.isspace():
                lemmas.append(lemma)
        return lemmas

    def get(self, index_type: str):
        if self.indices[index_type] is None:
            loaded = self._load_persisted(index_type)
            if loaded is not None:
                self.indices[index_type] = loaded
            else:
                raise RuntimeError(
                    f"Индекс '{index_type}' не создан. Вызовите add() перед использованием."
                )
        return self.indices[index_type]
