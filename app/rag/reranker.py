import os
import logging
from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

from .prompt_config import PromptConfig


logger = logging.getLogger(__name__)


class SafeLLMRerank(LLMRerank):
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        try:
            return super().postprocess_nodes(
                nodes=nodes, query_bundle=query_bundle, query_str=query_str
            )
        except Exception as exc:
            logger.warning("LLMRerank fallback: %s", exc)
            return nodes[: self.top_n] if self.top_n > 0 else nodes


class Reranker:
    def __init__(self):
        self.enabled = os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"
        self.top_n = int(os.getenv("RAG_RERANK_TOP_N", "3"))
        self.choice_batch_size = int(os.getenv("RAG_RERANK_CHOICE_BATCH_SIZE", "8"))
        self._postprocessors = None
        self.choice_select_prompt = PromptConfig().rerank_choice_select_prompt

    def get_postprocessors(self):
        if not self.enabled:
            return []
        if self._postprocessors is None:
            self._postprocessors = [
                SafeLLMRerank(
                    llm=Settings.llm,
                    choice_select_prompt=self.choice_select_prompt,
                    top_n=self.top_n,
                    choice_batch_size=self.choice_batch_size,
                )
            ]
        return self._postprocessors
