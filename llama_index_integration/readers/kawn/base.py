from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

# Import from your core SDK
from kawn.client import KawnClient, AsyncKawnClient
from kawn.services import OCRService, AsyncOCRService
from kawn.utils.logging import get_logger


logger = get_logger(['BaseerReader'])


def _format_result_to_documents(result: Any,
                                file_path: Union[str, Path],
                                one_text_result: Optional[bool] = False,
                                extra_info: Optional[Dict] = None) -> List[Document]:
    """
    Helper method to map the Kawn OCRResult into LlamaIndex Document objects.

    This function extracts text pages from the OCRResult and wraps them with metadata 
    to create LlamaIndex Document objects compatible with downstream processes.

    Args:
        result (Any): The OCR result object returned from the Kawn OCR Service. It is expected to have 'fileId', 'model', 'creditsConsumed', and a list of 'pages' with 'content'.
        file_path (Union[str, Path]): The path of the file that was processed by the OCR service. Use for metadata tracking.
        one_text_result (Optional[bool]): If True, concatenates all pages into a single LlamaIndex Document. If False, creates a LlamaIndex Document for each page. Defaults to False.
        extra_info (Optional[Dict]): Additional metadata to be injected into the resulting Document(s). Defaults to None.

    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    # Combine the content of all pages from the OCRResult
    metadata = extra_info or {}
    metadata["file_name"] = str(file_path)
    metadata["kawn_file_id"] = result.fileId
    metadata["kawn_model"] = result.model
    metadata["credits_consumed"] = result.creditsConsumed

    if one_text_result:
        full_text = "\n\n".join([page.content for page in result.pages])
        # We return a single Document containing the entire file's text.
        # Alternatively, you could yield one Document per page if preferred.
        return [Document(text=full_text, metadata=metadata)]

    else:
        formated_result = []
        for index, page in enumerate(result.pages):
            # Create a shallow copy for each page to avoid modifying the same dictionary reference
            metadata_copy = metadata.copy()
            metadata_copy["page_index"] = index
            formated_result.append(Document(text=page.content, metadata=metadata_copy))

        return formated_result


class BaseerReader(BaseReader):
    """
    Baseer OCR Reader by Kawn AI.
    
    This reader seamlessly processes files (PDF, images) using Kawn's Baseer OCR API, 
    and extracts state-of-the-art Arabic text and structural markdown from documents.
    It returns the resulting information as LlamaIndex Documents.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            options: Optional[dict] = None
    ):
        """
        Initialize the Baseer Reader.

        Args:
            api_key (Optional[str]): Kawn API key. If None, the client looks for the `KAWN_API_KEY` environment variable.
            model (Optional[str]): The specific model string to use for OCR (defaults to the backend's default, usually 'baseer/baseer-v2').
            options (Optional[dict]): Optional dictionary of OCR configuration parameters (e.g., webhook configuration, specific language options).
        """
        self.api_key = api_key
        self.model = model
        self.options = options

    def load_data(
            self, file_path: Union[str, Path],
            one_text_result: Optional[bool] = False,
            extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        Process a single document synchronously and return it as a list of LlamaIndex Documents.

        Args:
            file_path (Union[str, Path]): Path to the local file (e.g. PDF or image) to be processed.
            one_text_result (Optional[bool]): If True, groups all extracted text across all pages into a single LlamaIndex Document. Defaults to False.
            extra_info (Optional[Dict]): Any custom metadata dictionary to append to the parsed document. Defaults to None.

        Returns:
            List[Document]: A list of LlamaIndex Documents containing text extracted via the Baseer OCR model.
        """
        logger.info("Build the client to connect with Kawn AI, make sure to set the API key")
        with KawnClient(api_key=self.api_key) as client:
            ocr_service = OCRService(client)

            # Assuming process_file wraps the upload -> poll -> fetch logic
            # and returns the OCRResult Pydantic model
            logger.info("Start processing document...")
            result = ocr_service.process_file(
                file_path=file_path,
                model=self.model,
                options=self.options,
                return_result=True
            )
        logger.info("Finish processing document...")
        return _format_result_to_documents(result, file_path, one_text_result, extra_info)

    async def aload_data(self,
                         file_path: Union[str, Path],
                         one_text_result: Optional[bool] = False,
                         extra_info: Optional[Dict] = None
                         ) -> List[Document]:
        """
        Process a single document asynchronously and return it as a list of LlamaIndex Documents.

        Args:
            file_path (Union[str, Path]): Path to the local file (e.g. PDF or image) to be processed.
            one_text_result (Optional[bool]): If True, groups all extracted text across all pages into a single LlamaIndex Document. Defaults to False.
            extra_info (Optional[Dict]): Any custom metadata dictionary to append to the parsed document. Defaults to None.

        Returns:
            List[Document]: A list of LlamaIndex Documents containing text extracted via the Baseer OCR model.
        """
        logger.info("Build the client to connect with Kawn AI, make sure to set the API key")
        async with AsyncKawnClient(api_key=self.api_key) as client:
            ocr_service = AsyncOCRService(client)
            logger.info("Start processing document...")
            result = await ocr_service.process_file(
                file_path=file_path,
                model=self.model,
                options=self.options,
                return_result=True
            )
        logger.info("Finish processing document...")
        return _format_result_to_documents(result, file_path, one_text_result, extra_info)
