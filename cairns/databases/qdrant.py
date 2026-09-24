from typing import Any

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Qdrant



class CustomQdrant(Qdrant):
    """
    Class performs lookup on the Qdrant in chains.
    """
    @classmethod
    def _document_from_scored_point(
            cls,
            scored_point: Any,
            collection_name: str,
            content_payload_key: str,
            metadata_payload_key: str,
    ) -> Document:
        """
        This method is overriden to get the documents form a local file to provide for the context.
        :param scored_point:
        :param collection_name:
        :param content_payload_key:
        :param metadata_payload_key:
        :return:
        """
        payload = scored_point.payload or {}
        metadata = payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        metadata["score"] = scored_point.score

        for key in (
            "tool_id",
            "tool_name",
            "question_id",
            "study_id",
            "fields_used",
            "canonical_text",
        ):
            if key in payload:
                metadata[key] = payload[key]

        page_content = (
            payload.get(content_payload_key)
            or payload.get("canonical_text")
            or payload.get("question")
            or payload.get("description")
            or ""
        )

        return Document(
            page_content=page_content,
            metadata=metadata,
        )
