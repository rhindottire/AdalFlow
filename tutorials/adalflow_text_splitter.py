"Code for tutorial: https://adalflow.sylph.ai/tutorials/text_splitter.html"

from adalflow.components.data_process.text_splitter import TextSplitter
from adalflow.core.types import Document
from typing import Optional, Dict


def split_by_words(
    text: str, chunk_size: int = 5, chunk_overlap: int = 1, doc_id: Optional[str] = None
) -> list:

    text_splitter = TextSplitter(
        split_by="word", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    doc = Document(text=text, id=doc_id or "doc1")

    return text_splitter.call(documents=[doc])


def split_by_tokens(
    text: str, chunk_size: int = 5, chunk_overlap: int = 0, doc_id: Optional[str] = None
) -> list:

    text_splitter = TextSplitter(
        split_by="token", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    doc = Document(text=text, id=doc_id or "doc1")

    return text_splitter.call(documents=[doc])


def split_by_custom(
    text: str,
    split_by: str,
    separators: Dict[str, str],
    chunk_size: int = 1,
    chunk_overlap: int = 0,
    doc_id: Optional[str] = None,
) -> list:

    text_splitter = TextSplitter(
        split_by=split_by,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    doc = Document(text=text, id=doc_id or "doc1")

    return text_splitter.call(documents=[doc])


def example_usage():
    text = "Example text. More example text. Even more text to illustrate."
    word_splits = split_by_words(text, chunk_size=5, chunk_overlap=1)
    print("\nWord Split Example:")
    for doc in word_splits:
        print(doc)

    token_splits = split_by_tokens(text, chunk_size=5, chunk_overlap=0)
    print("\nToken Split Example:")
    for doc in token_splits:
        print(doc)

    question_text = "What is your name? How old are you? Where do you live?"
    custom_splits = split_by_custom(
        text=question_text,
        split_by="question",
        chunk_size=1,
        separators={"question": "?"},
    )
    print("\nCustom Separator Example:")
    for doc in custom_splits:
        print(doc)


if __name__ == "__main__":
    example_usage()
