from typing import Any, List, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr, Field

# Import from core SDK
from kawn.client import KawnClient, AsyncKawnClient
from kawn.services import EmbeddingService, AsyncEmbeddingService


class KawnEmbedding(BaseEmbedding):
    """
    Kawn Embedding Model for Arabic and Islamic texts (Tbyaan).
    """

    model_name: str = Field(
        default="tbyaan/islamic-embedding-tbyaan-v1",
        description="The Kawn embedding model to use."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have."
    )
    normalize: Optional[bool] = Field(
        default=None,
        description="Whether to normalize the output embeddings."
    )
    prompt_name: Optional[str] = Field(
        default=None,
        description="Prompt name to use for the embedding."
    )
    truncate: Optional[bool] = Field(
        default=None,
        description="Whether to truncate the input to fit the model's max context length."
    )
    truncation_direction: Optional[str] = Field(
        default=None,
        description="Direction to truncate ('Left' or 'Right')."
    )

    _api_key: Optional[str] = PrivateAttr()

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        if model_name:
            self.model_name = model_name

    def _get_api_kwargs(self) -> dict:
        """Helper to build the extra arguments for the Kawn SDK request."""
        kwargs = {}
        if self.dimensions is not None: kwargs["dimensions"] = self.dimensions
        if self.normalize is not None: kwargs["normalize"] = self.normalize
        if self.prompt_name is not None: kwargs["promptName"] = self.prompt_name
        if self.truncate is not None: kwargs["truncate"] = self.truncate
        if self.truncation_direction is not None: kwargs["truncationDirection"] = self.truncation_direction
        return kwargs

    def _extract_embedding(self, data: Any) -> List[float]:
        """Safely extract a single embedding from Union[EmbeddingData, List[EmbeddingData]]."""
        if isinstance(data, list):
            return data[0].embedding
        return data.embedding

    def _extract_embeddings_list(self, data: Any) -> List[List[float]]:
        """Safely extract a batch of embeddings from Union[EmbeddingData, List[EmbeddingData]]."""
        if isinstance(data, list):
            return [item.embedding for item in data]
        return [data.embedding]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Synchronous query embedding."""
        with KawnClient(api_key=self._api_key) as client:
            service = EmbeddingService(client)
            response = service.create(
                input=query,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embedding(response.data)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Synchronous text embedding."""
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch text embeddings."""
        with KawnClient(api_key=self._api_key) as client:
            service = EmbeddingService(client)
            response = service.create(
                input=texts,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embeddings_list(response.data)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding."""
        async with AsyncKawnClient(api_key=self._api_key) as client:
            service = AsyncEmbeddingService(client)
            response = await service.create(
                input=query,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embedding(response.data)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous text embedding."""
        return await self._aget_query_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch text embeddings."""
        async with AsyncKawnClient(api_key=self._api_key) as client:
            service = AsyncEmbeddingService(client)
            response = await service.create(
                input=texts,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embeddings_list(response.data)
