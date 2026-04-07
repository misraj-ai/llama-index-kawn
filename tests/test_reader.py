import unittest

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from llama_index.core.schema import Document
from llama_index_integration.readers.kawn.base import BaseerReader, _format_result_to_documents

@pytest.fixture
def mock_ocr_result():
    mock_page1 = MagicMock()
    mock_page1.content = "Page 1 content"
    mock_page2 = MagicMock()
    mock_page2.content = "Page 2 content"
    
    mock_result = MagicMock()
    mock_result.fileId = "test_id_123"
    mock_result.model = "test_model"
    mock_result.creditsConsumed = 2
    mock_result.pages = [mock_page1, mock_page2]
    return mock_result

def test_format_result_to_documents_multiple(mock_ocr_result):
    docs = _format_result_to_documents(
        result=mock_ocr_result,
        file_path="test.pdf",
        one_text_result=False,
        extra_info={"custom": "info"}
    )
    
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].text == "Page 1 content"
    assert docs[0].metadata["file_name"] == "test.pdf"
    assert docs[0].metadata["kawn_file_id"] == "test_id_123"
    assert docs[0].metadata["kawn_model"] == "test_model"
    assert docs[0].metadata["credits_consumed"] == 2
    assert docs[0].metadata["page_index"] == 0
    assert docs[0].metadata["custom"] == "info"
    
    assert docs[1].text == "Page 2 content"
    assert docs[1].metadata["page_index"] == 1

def test_format_result_to_documents_single(mock_ocr_result):
    docs = _format_result_to_documents(
        result=mock_ocr_result,
        file_path="test.pdf",
        one_text_result=True,
    )
    
    assert len(docs) == 1
    assert docs[0].text == "Page 1 content\n\nPage 2 content"
    assert docs[0].metadata["file_name"] == "test.pdf"

def test_baseer_reader_initialization():
    reader = BaseerReader(api_key="test_key", model="test_model", options={"opt1": "val1"})
    assert reader.api_key == "test_key"
    assert reader.model == "test_model"
    assert reader.options == {"opt1": "val1"}

@patch("llama_index_integration.readers.kawn.base.KawnClient")
@patch("llama_index_integration.readers.kawn.base.OCRService")
def test_load_data(mock_ocr_service, mock_kawn_client, mock_ocr_result):
    mock_service_instance = mock_ocr_service.return_value
    mock_service_instance.process_file.return_value = mock_ocr_result

    reader = BaseerReader(api_key="test_key")
    docs = reader.load_data("dummy_path.pdf")
    
    assert len(docs) == 2
    mock_service_instance.process_file.assert_called_once_with(
        file_path="dummy_path.pdf",
        model=None,
        options=None,
        return_result=True
    )

@pytest.mark.asyncio
@patch("llama_index_integration.readers.kawn.base.AsyncKawnClient")
@patch("llama_index_integration.readers.kawn.base.AsyncOCRService")
async def test_aload_data(mock_async_ocr_service, mock_async_kawn_client, mock_ocr_result):
    mock_service_instance = mock_async_ocr_service.return_value
    mock_service_instance.process_file = AsyncMock(return_value=mock_ocr_result)

    reader = BaseerReader(api_key="test_key")
    docs = await reader.aload_data("dummy_path.pdf", one_text_result=True)
    
    assert len(docs) == 1
    mock_service_instance.process_file.assert_called_once_with(
        file_path="dummy_path.pdf",
        model=None,
        options=None,
        return_result=True
    )
