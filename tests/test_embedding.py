import pytest
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from llama_index_integration.embeddings.kawn.base import KawnEmbedding

@pytest.fixture
def mock_embedding_response():
    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = mock_data
    return mock_response

@pytest.fixture
def mock_embeddings_list_response():
    mock_data1 = MagicMock()
    mock_data1.embedding = [0.1, 0.2, 0.3]
    mock_data2 = MagicMock()
    mock_data2.embedding = [0.4, 0.5, 0.6]
    mock_response = MagicMock()
    mock_response.data = [mock_data1, mock_data2]
    return mock_response

def test_kawn_embedding_initialization():
    embed_model = KawnEmbedding(api_key="test_key", model_name="test_model", dimensions=128)
    assert embed_model._api_key == "test_key"
    assert embed_model.model_name == "test_model"
    assert embed_model.dimensions == 128

@patch("llama_index_integration.embeddings.kawn.base.KawnClient")
@patch("llama_index_integration.embeddings.kawn.base.EmbeddingService")
def test_get_query_embedding(mock_embedding_service, mock_kawn_client, mock_embedding_response):
    mock_service_instance = mock_embedding_service.return_value
    mock_service_instance.create.return_value = mock_embedding_response

    embed_model = KawnEmbedding(api_key="test_key")
    result = embed_model._get_query_embedding("test query")
    
    assert result == [0.1, 0.2, 0.3]
    mock_service_instance.create.assert_called_once_with(
        input="test query", model="tbyaan/islamic-embedding-tbyaan-v1"
    )

@patch("llama_index_integration.embeddings.kawn.base.KawnClient")
@patch("llama_index_integration.embeddings.kawn.base.EmbeddingService")
def test_get_text_embedding(mock_embedding_service, mock_kawn_client, mock_embedding_response):
    mock_service_instance = mock_embedding_service.return_value
    mock_service_instance.create.return_value = mock_embedding_response

    embed_model = KawnEmbedding(api_key="test_key")
    result = embed_model._get_text_embedding("test text")
    
    assert result == [0.1, 0.2, 0.3]

@patch("llama_index_integration.embeddings.kawn.base.KawnClient")
@patch("llama_index_integration.embeddings.kawn.base.EmbeddingService")
def test_get_text_embeddings(mock_embedding_service, mock_kawn_client, mock_embeddings_list_response):
    mock_service_instance = mock_embedding_service.return_value
    mock_service_instance.create.return_value = mock_embeddings_list_response

    embed_model = KawnEmbedding(api_key="test_key")
    result = embed_model._get_text_embeddings(["text 1", "text 2"])
    
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

@pytest.mark.asyncio
@patch("llama_index_integration.embeddings.kawn.base.AsyncKawnClient")
@patch("llama_index_integration.embeddings.kawn.base.AsyncEmbeddingService")
async def test_aget_query_embedding(mock_async_embedding_service, mock_async_kawn_client, mock_embedding_response):
    mock_service_instance = mock_async_embedding_service.return_value
    mock_service_instance.create = AsyncMock(return_value=mock_embedding_response)

    embed_model = KawnEmbedding(api_key="test_key")
    result = await embed_model._aget_query_embedding("test query")
    
    assert result == [0.1, 0.2, 0.3]
    mock_service_instance.create.assert_called_once_with(
        input="test query", model="tbyaan/islamic-embedding-tbyaan-v1"
    )

@pytest.mark.asyncio
@patch("llama_index_integration.embeddings.kawn.base.AsyncKawnClient")
@patch("llama_index_integration.embeddings.kawn.base.AsyncEmbeddingService")
async def test_aget_text_embedding(mock_async_embedding_service, mock_async_kawn_client, mock_embedding_response):
    mock_service_instance = mock_async_embedding_service.return_value
    mock_service_instance.create = AsyncMock(return_value=mock_embedding_response)

    embed_model = KawnEmbedding(api_key="test_key")
    result = await embed_model._aget_text_embedding("test text")
    
    assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
@patch("llama_index_integration.embeddings.kawn.base.AsyncKawnClient")
@patch("llama_index_integration.embeddings.kawn.base.AsyncEmbeddingService")
async def test_aget_text_embeddings(mock_async_embedding_service, mock_async_kawn_client, mock_embeddings_list_response):
    mock_service_instance = mock_async_embedding_service.return_value
    mock_service_instance.create = AsyncMock(return_value=mock_embeddings_list_response)

    embed_model = KawnEmbedding(api_key="test_key")
    result = await embed_model._aget_text_embeddings(["text 1", "text 2"])
    
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
