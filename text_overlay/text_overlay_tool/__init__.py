"""
Text Overlay Tool - Google Cloud Vision OCR Version
텍스트 오버레이 툴 - 구글 클라우드 비전 OCR 버전

A modular package for overlaying OCR text onto target images.
OCR 텍스트를 타겟 이미지에 오버레이하는 모듈화된 패키지입니다.
"""

__version__ = "2.0.0"

from .utils import logger, resource_path
from .models import TextRegion, DraggableTableWidgetItem
from .ocr import CloudVisionOCR, CLOUD_VISION_AVAILABLE
from .ui import ImageCanvas

__all__ = [
    'logger',
    'resource_path',
    'TextRegion',
    'DraggableTableWidgetItem',
    'CloudVisionOCR',
    'CLOUD_VISION_AVAILABLE',
    'ImageCanvas',
]

