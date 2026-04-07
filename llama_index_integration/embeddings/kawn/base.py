from typing import Any, List, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr, Field

# Import from core SDK
from kawn.client import KawnClient, AsyncKawnClient
from kawn.services import EmbeddingService, AsyncEmbeddingService


class KawnEmbedding(BaseEmbedding):
    """
    Kawn Embedding Model for Arabic and Islamic texts (Tbyaan).

    This class integrates Kawn's Tbyaan embedding models with LlamaIndex. 
    It supports both synchronous and asynchronous embedding generation for queries and texts.

    Attributes:
        model_name (str): The Kawn embedding model to use. Defaults to "tbyaan/islamic-embedding-tbyaan-v1".
        dimensions (Optional[int]): The number of dimensions the resulting output embeddings should have.
        normalize (Optional[bool]): Whether to normalize the output embeddings.
        prompt_name (Optional[str]): Prompt name to use for the embedding.
        truncate (Optional[bool]): Whether to truncate the input to fit the model's max context length.
        truncation_direction (Optional[str]): Direction to truncate ('Left' or 'Right').
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
        """
        Initialize the KawnEmbedding class.

        Args:
            api_key (Optional[str]): Kawn API key. If not provided, the underlying client will attempt to find it in the environment variables (KAWN_API_KEY).
            model_name (Optional[str]): The model to use for generating embeddings.
            **kwargs: Additional keyword arguments passed to the BaseEmbedding class.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        if model_name:
            self.model_name = model_name

    def _get_api_kwargs(self) -> dict:
        """
        Helper method to build the extra arguments for the Kawn SDK request.

        Returns:
            dict: A dictionary containing the embedding configuration parameters.
        """
        kwargs = {}
        if self.dimensions is not None: kwargs["dimensions"] = self.dimensions
        if self.normalize is not None: kwargs["normalize"] = self.normalize
        if self.prompt_name is not None: kwargs["promptName"] = self.prompt_name
        if self.truncate is not None: kwargs["truncate"] = self.truncate
        if self.truncation_direction is not None: kwargs["truncationDirection"] = self.truncation_direction
        return kwargs

    def _extract_embedding(self, data: Any) -> List[float]:
        """
        Safely extract a single embedding from the API response data.

        Args:
            data (Any): The embedding data returned from the API. Can be a single item or a list.

        Returns:
            List[float]: The extracted embedding vector.
        """
        if isinstance(data, list):
            return data[0].embedding
        return data.embedding

    def _extract_embeddings_list(self, data: Any) -> List[List[float]]:
        """
        Safely extract a batch of embeddings from the API response data.

        Args:
            data (Any): The embedding data returned from the API, expected to be a list of embedding objects.

        Returns:
            List[List[float]]: A list of extracted embedding vectors.
        """
        if isinstance(data, list):
            return [item.embedding for item in data]
        return [data.embedding]

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Synchronously compute the embedding for a single query string.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The embedding vector for the query.
        """
        with KawnClient(api_key=self._api_key) as client:
            service = EmbeddingService(client)
            response = service.create(
                input=query,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embedding(response.data)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Synchronously compute the embedding for a single text document.

        Args:
            text (str): The text document to embed.

        Returns:
            List[float]: The embedding vector for the text.
        """
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronously compute embeddings for a batch of text documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embedding vectors corresponding to the input texts.
        """
        with KawnClient(api_key=self._api_key) as client:
            service = EmbeddingService(client)
            response = service.create(
                input=texts,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embeddings_list(response.data)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronously compute the embedding for a single query string.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The embedding vector for the query.
        """
        async with AsyncKawnClient(api_key=self._api_key) as client:
            service = AsyncEmbeddingService(client)
            response = await service.create(
                input=query,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embedding(response.data)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously compute the embedding for a single text document.

        Args:
            text (str): The text document to embed.

        Returns:
            List[float]: The embedding vector for the text.
        """
        return await self._aget_query_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously compute embeddings for a batch of text documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embedding vectors corresponding to the input texts.
        """
        async with AsyncKawnClient(api_key=self._api_key) as client:
            service = AsyncEmbeddingService(client)
            response = await service.create(
                input=texts,
                model=self.model_name,
                **self._get_api_kwargs()
            )
            return self._extract_embeddings_list(response.data)
