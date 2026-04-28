from llama_index.core.prompts import PromptTemplate
from typing import Dict, Any
import textwrap


class PromptConfig:
    """
    Централизованное хранилище промптов для RAG-пайплайна.
    Ниже каждый шаблон подписан по этапу: индексация, retrieval, ответ, чат.
    """

    # Общий QA-шаблон для ответов в query_engine:
    # используется для vector/tree/kg/bm25 при генерации финального ответа.
    CUSTOM_TEMPLATE_STR = textwrap.dedent(
        """\
        Ты — помощник и консультант. Отвечай строго на основе предоставленного контекста.
        Правила:
        1. Если в контексте есть упоминание темы — подтверди это и приведи релевантные фрагменты.
        2. Если темы нет — скажи: "В документах нет информации по этому вопросу".
        3. Дай развернутый по сути ответ: с основными условиями, деталями и ограничениями из контекста.
        4. Пиши структурно и понятно, без воды и без служебных формулировок.
        5. Запрещено использовать фразы: "Ответ", "В соответствии с запросом", "Новый ответ", "Я должен предоставить информацию".
        Не используй markdown-заголовки и оформление (**...**).
        
        Контекст:
        {context_str}

        Вопрос: {query_str}
        Ответ:
    """
    )

    # Контекстный шаблон для stateful-чата (CondensePlusContextChatEngine):
    # задает стиль ответа на основе найденного контекста.
    CHAT_CONTEXT_PROMPT_STR = textwrap.dedent(
        """\
        Ты — помощник и консультант. Отвечай строго на основе предоставленного контекста.
        Правила:
        1. Если в контексте есть упоминание темы — подтверди это и приведи релевантные фрагменты.
        2. Если темы нет — скажи: "В документах нет информации по этому вопросу".
        3. Дай развернутый по сути ответ: с основными условиями, деталями и ограничениями из контекста.
        4. Пиши структурно и понятно, без воды и без служебных формулировок.
        5. Запрещено использовать фразы: "Ответ", "В соответствии с запросом", "Новый ответ", "Я должен предоставить информацию".
        Не используй markdown-заголовки и оформление (**...**).

        Контекст:
        {context_str}

        Инструкция: ответь на последний вопрос пользователя только по контексту выше.
    """
    )

    # Шаблон для "сжатия" истории чата в короткий вопрос для retrieval.
    CONDENSE_PROMPT_STR = (
        "Суммаризируй историю диалога на русском, сохранив ключевые факты, "
        "номера статей/пунктов и суть вопроса. Будь краток.\n"
        "История:\n{chat_history}\nВопрос: {question}"
    )

    # Refine-шаблон: используется, когда ответ уточняется по новому контексту.
    # Нужен прежде всего для tree/kg query_engine.
    REFINE_TEMPLATE_STR = textwrap.dedent(
        """\
        Ты уточняешь уже сформированный ответ по новому контексту.
        Отвечай только по контексту, без внешних знаний.
        Если новый контекст не добавляет фактов, верни исходный ответ без изменений.
        Пиши сразу по сути, без вступлений и служебных блоков.
        Не используй фразы "Original Answer", "Новый ответ", "В соответствии с запросом".

        Вопрос: {query_str}
        Текущий ответ: {existing_answer}
        Новый контекст:
        {context_msg}

        Обновленный ответ:
    """
    )

    # Summary-шаблон для построения TreeIndex:
    # применяется в index_registry при TreeIndex.from_documents(..., summary_template=...).
    SUMMARY_PROMPT_STR = "Суммаризируй максимально точно только по контексту.\n\n{context_str}\n\nСводка:"

    # Tree retrieval (single choice):
    # выбор одного наиболее релевантного узла дерева.
    TREE_QUERY_TEMPLATE_STR = textwrap.dedent(
        """\
        Ниже список вариантов (1..{num_chunks}), каждый вариант — это краткая сводка узла.
        ---------------------
        {context_list}
        ---------------------
        Используя только варианты выше и не используя внешние знания,
        выбери один самый релевантный вопросу: "{query_str}".
        Ответ верни в формате: ANSWER: <number>
    """
    )

    # Tree retrieval (multiple choice):
    # выбор нескольких релевантных веток дерева.
    TREE_QUERY_MULTIPLE_TEMPLATE_STR = textwrap.dedent(
        """\
        Ниже список вариантов (1..{num_chunks}), каждый вариант — это краткая сводка узла.
        ---------------------
        {context_list}
        ---------------------
        Используя только варианты выше и не используя внешние знания,
        выбери не более {branching_factor} самых релевантных вопросу "{query_str}".
        Упорядочи от более релевантного к менее релевантному.
        Ответ верни в формате: ANSWER: <numbers>
    """
    )

    # Шаблон извлечения триплетов при построении KGIndex.
    KG_TRIPLET_EXTRACT_TEMPLATE_STR = textwrap.dedent(
        """\
        Извлеки до {max_knowledge_triplets} триплетов знаний из текста.
        Формат: (субъект, отношение, объект)
        Не используй внешние знания, только текст ниже.
        ---------------------
        {text}
        ---------------------
        Триплеты:
    """
    )

    # Шаблон извлечения ключевых слов для query по графу знаний.
    KG_QUERY_KEYWORD_TEMPLATE_STR = textwrap.dedent(
        """\
        Из вопроса извлеки до {max_keywords} ключевых слов для поиска по графу знаний.
        Избегай стоп-слов.
        ---------------------
        {question}
        ---------------------
        Формат ответа: KEYWORDS: <слово1, слово2, ...>
    """
    )

    # Шаблон для LLM reranker (оценка релевантности кандидатов).
    RERANK_CHOICE_SELECT_PROMPT_STR = textwrap.dedent(
        """\
        Ниже список документов. У каждого документа есть номер и краткое содержание.
        Твоя задача — выбрать только релевантные документы для ответа на вопрос.
        Отвечай СТРОГО в формате строк:
        Doc: <номер>, Relevance: <оценка 1-10>
        Никакого дополнительного текста, пояснений, markdown и заголовков.
        Если релевантных документов нет, верни пустой ответ.

        {context_str}
        Question: {query_str}
        Answer:
    """
    )

    def __init__(self):
        self.custom_template = PromptTemplate(self.CUSTOM_TEMPLATE_STR)
        self.refine_template = PromptTemplate(self.REFINE_TEMPLATE_STR)
        self.summary_prompt = PromptTemplate(self.SUMMARY_PROMPT_STR)
        self.tree_query_template = PromptTemplate(self.TREE_QUERY_TEMPLATE_STR)
        self.tree_query_template_multiple = PromptTemplate(
            self.TREE_QUERY_MULTIPLE_TEMPLATE_STR
        )
        self.kg_triplet_extract_template = PromptTemplate(
            self.KG_TRIPLET_EXTRACT_TEMPLATE_STR
        )
        self.kg_query_keyword_template = PromptTemplate(
            self.KG_QUERY_KEYWORD_TEMPLATE_STR
        )
        self.rerank_choice_select_prompt = PromptTemplate(
            self.RERANK_CHOICE_SELECT_PROMPT_STR
        )

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает все шаблоны в виде словаря для удобной передачи в другие модули."""
        return {
            "custom_template": self.custom_template,
            "chat_context_prompt_str": self.CHAT_CONTEXT_PROMPT_STR,
            "condense_prompt_str": self.CONDENSE_PROMPT_STR,
            "refine_template": self.refine_template,
            "summary_prompt": self.summary_prompt,
            "tree_query_template": self.tree_query_template,
            "tree_query_template_multiple": self.tree_query_template_multiple,
            "kg_triplet_extract_template": self.kg_triplet_extract_template,
            "kg_query_keyword_template": self.kg_query_keyword_template,
            "rerank_choice_select_prompt": self.rerank_choice_select_prompt,
        }

    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Factory-метод для быстрого получения словаря промптов."""
        return cls().to_dict()
