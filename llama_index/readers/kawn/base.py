from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

# Import from your core SDK
from kawn.client import KawnClient
from kawn.services import OCRService


class BasserReader(BaseReader):
    """
    Basser OCR Reader by Kawn AI.
    Extracts state-of-the-art Arabic text and structural markdown from documents.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, options: Optional[dict] = None):
        """
        Initialize the Basser Reader.

        Args:
            api_key: Kawn API key. If None, it will look for KAWN_API_KEY environment variable.
            model: The specific model string to use (defaults to 'baseer/baseer-v2').
            options: Optional dictionary of OCR configuration parameters.
        """
        self.api_key = api_key
        self.model = model
        self.options = options

    def load_data(
            self, file_path: Union[str, Path], extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        Process the document and return it as a LlamaIndex Document.
        """
        # Initialize client context manager to ensure safe HTTP connection closure
        with KawnClient(api_key=self.api_key) as client:
            ocr_service = OCRService(client)

            # This triggers the upload, polls for status, and retrieves the result
            result = ocr_service.process_file(
                file_path=file_path,
                model=self.model,
                options=self.options,
                return_result=True
            )

        # NOTE: Adjust the attribute below depending on how OCRResult is structured
        # in your types/ocr.py (e.g., result.text, result.markdown, or result.content)
        extracted_text = result.text if hasattr(result, 'text') else str(result)

        # Package the result into a LlamaIndex Document
        metadata = extra_info or {}
        metadata["file_name"] = str(file_path)

        return [Document(text=extracted_text, metadata=metadata)]
