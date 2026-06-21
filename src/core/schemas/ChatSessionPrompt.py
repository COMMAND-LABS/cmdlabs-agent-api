from pydantic import BaseModel
from typing import Optional


class ChatSessionPrompt(BaseModel):
    prompt: str
    sessionId: str
    # Optional PDF attachment (base64 encoded)
    pdf: Optional[str] = None
    pdfFilename: Optional[str] = None
    # PDF processing mode:
    # - True: Use vision (images) - for scanned PDFs, charts, visual layout
    # - False: Use text extraction - for data extraction, cheaper with gpt-4o-mini
    pdfUseVision: Optional[bool] = False

    # Optional image attachment (base64 encoded) for vision models.
    image: Optional[str] = None
    # Optional inline text content for txt/csv/md attachments.
    documentText: Optional[str] = None

    # GCS reference for the persisted attachment (returned by ai-api
    # POST /api/files/upload). Stored on the chat message so the original file
    # can be resolved later. The model-facing content still rides inline above.
    gcsBucket: Optional[str] = None
    gcsFilePath: Optional[str] = None
    attachmentFilename: Optional[str] = None
    attachmentContentType: Optional[str] = None