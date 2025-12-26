"""
Google Cloud Vision API OCR module
구글 클라우드 비전 API OCR 모듈
"""

import io
import cv2
import numpy as np
from PIL import Image

# Import logger from utils / 유틸리티에서 로거 가져오기
from ..utils import logger

# Google Cloud Vision API (optional)
# 구글 클라우드 비전 API (선택적)
# Note: google-cloud-vision package must be installed
# 참고: google-cloud-vision 패키지가 설치되어야 합니다
# Installation: pip install google-cloud-vision
try:
    from google.cloud import vision  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    CLOUD_VISION_AVAILABLE = True
except ImportError:
    CLOUD_VISION_AVAILABLE = False
    vision = None  # type: ignore
    service_account = None  # type: ignore


class CloudVisionOCR:
    """
    Text extraction class using Google Cloud Vision API
    구글 클라우드 비전 API를 사용한 텍스트 추출 클래스
    
    This class handles OCR processing using Google Cloud Vision API.
    이 클래스는 Google Cloud Vision API를 사용하여 OCR 처리를 수행합니다.
    """
    
    def __init__(self):
        """Initialize Cloud Vision OCR client / 클라우드 비전 OCR 클라이언트 초기화"""
        self.credentials_path = None  # Service account key file path / 서비스 계정 키 파일 경로
        self.vision_client = None  # Cloud Vision client instance / 클라우드 비전 클라이언트 인스턴스
    
    def set_credentials_path(self, credentials_path):
        """
        Set Google Cloud Vision API service account key file path
        구글 클라우드 비전 API 서비스 계정 키 파일 경로 설정
        
        Args / 인자:
            credentials_path (str): Path to service account JSON key file
                                  / 서비스 계정 JSON 키 파일 경로
                                  
        Returns / 반환값:
            bool: True if credentials are set successfully, False otherwise
                 / 인증 정보가 성공적으로 설정되면 True, 그렇지 않으면 False
        """
        self.credentials_path = credentials_path
        if credentials_path and CLOUD_VISION_AVAILABLE and vision is not None:
            try:
                # 서비스 계정 키 파일로 인증 / Authenticate with service account key file
                credentials = service_account.Credentials.from_service_account_file(credentials_path)  # type: ignore
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)  # type: ignore
                return True
            except Exception as e:
                logger.error(f"구글 클라우드 비전 API 설정 실패: {e}")
                self.vision_client = None
                return False
        else:
            self.vision_client = None
            return False
    
    def extract_text_full_image_vision(self, image_path):
        """
        Perform OCR on entire image using Google Cloud Vision API
        구글 클라우드 비전 API로 전체 이미지 OCR 수행
        
        Args / 인자:
            image_path (str or np.ndarray): Path to image file or image array
                                          / 이미지 파일 경로 또는 이미지 배열
                                          
        Returns / 반환값:
            list[str]: List of extracted text lines / 추출된 텍스트 라인 목록
            
        Raises / 예외:
            Exception: If OCR processing fails / OCR 처리 실패 시
        """
        if not CLOUD_VISION_AVAILABLE:
            logger.error("google-cloud-vision 패키지가 설치되지 않았습니다.")
            return []
        
        if not self.vision_client:
            logger.error("구글 클라우드 비전 API 클라이언트가 설정되지 않았습니다.")
            return []
        
        try:
            # 이미지 파일 읽기 / Read image file
            if isinstance(image_path, str):
                # 파일 경로인 경우 / If it's a file path
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')):
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                else:
                    logger.error(f"지원하지 않는 이미지 형식: {image_path}")
                    return []
            else:
                # 이미지 배열인 경우 (OpenCV 이미지) / If it's an image array (OpenCV image)
                # PIL로 변환 후 바이트로 저장 / Convert to PIL and save as bytes
                pil_image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                image_data = img_byte_arr.getvalue()
            
            # Cloud Vision API 호출 / Call Cloud Vision API
            image = vision.Image(content=image_data)  # type: ignore
            
            # 텍스트 감지 수행 (한국어, 일본어, 영어 지원) / Perform text detection (supports Korean, Japanese, English)
            response = self.vision_client.text_detection(image=image)  # type: ignore
            
            # 응답에서 텍스트 추출 / Extract text from response
            texts = []
            if response.text_annotations:
                # 첫 번째 annotation은 전체 텍스트 / First annotation is the full text
                full_text = response.text_annotations[0].description
                if full_text:
                    # 개행 문자로 분리하여 리스트로 변환 / Split by newline and convert to list
                    text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                    texts.extend(text_lines)
            
            return texts
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"구글 클라우드 비전 OCR 오류: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            
            # API 오류 분류 / API error classification
            if "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
                raise Exception(
                    "구글 클라우드 비전 API 권한 오류가 발생했습니다.\n\n"
                    "가능한 원인:\n"
                    "1. 서비스 계정 키 파일이 유효하지 않음\n"
                    "2. Cloud Vision API가 활성화되지 않음\n"
                    "3. 서비스 계정에 필요한 권한이 없음\n\n"
                    "해결 방법:\n"
                    "1. Google Cloud Console에서 Cloud Vision API 활성화 확인\n"
                    "2. 서비스 계정에 'Cloud Vision API 사용자' 역할 부여\n"
                    "3. 새로운 서비스 계정 키 파일 다운로드"
                )
            elif "invalid" in error_msg.lower() or "not found" in error_msg.lower():
                raise Exception(
                    "구글 클라우드 비전 API 인증 오류가 발생했습니다.\n\n"
                    "가능한 원인:\n"
                    "1. 서비스 계정 키 파일 경로가 잘못됨\n"
                    "2. 키 파일이 손상되었거나 유효하지 않음\n\n"
                    "해결 방법:\n"
                    "1. 서비스 계정 키 파일 경로 확인\n"
                    "2. Google Cloud Console에서 새로운 키 파일 다운로드"
                )
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                raise Exception(
                    "구글 클라우드 비전 API 사용 한도에 도달했습니다.\n\n"
                    "해결 방법:\n"
                    "1. Google Cloud Console에서 할당량 확인\n"
                    "2. 결제 계정 설정 확인\n"
                    "3. 잠시 후 다시 시도"
                )
            else:
                # 일반 오류는 그대로 전달 / General error is passed as is
                raise Exception(f"구글 클라우드 비전 OCR 오류: {error_msg}")

