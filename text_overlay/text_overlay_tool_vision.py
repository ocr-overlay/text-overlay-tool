"""
Text Overlay Tool - Google Cloud Vision OCR Version
텍스트 오버레이 툴 - 구글 클라우드 비전 OCR 버전

Description / 설명:
    A tool for overlaying OCR text extracted from source images onto target images.
    This version uses Google Cloud Vision API for OCR processing.
    
    소스 이미지에서 추출한 OCR 텍스트를 타겟 이미지에 오버레이하는 도구입니다.
    이 버전은 OCR 처리를 위해 Google Cloud Vision API를 사용합니다.

Features / 기능:
    - Google Cloud Vision API OCR for text extraction
    - Text overlay on target images with customizable fonts and styles
    - Support for Korean, Japanese, and English text
    - Batch processing for multiple images
    - CSV import/export for text data
    - Custom font support
    
    - 구글 클라우드 비전 API를 통한 텍스트 추출
    - 커스터마이징 가능한 폰트와 스타일로 타겟 이미지에 텍스트 오버레이
    - 한글, 일본어, 영어 텍스트 지원
    - 여러 이미지 일괄 처리 지원
    - 텍스트 데이터 CSV 가져오기/내보내기
    - 사용자 정의 폰트 지원

Requirements / 요구사항:
    - Python 3.7+
    - PyQt5
    - OpenCV (cv2)
    - PIL/Pillow
    - google-cloud-vision (optional, for OCR)
    - NumPy

Author / 작성자: TextOverlayTool Team
Version / 버전: 2.0
License / 라이선스: See LICENSE file
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QFontDatabase, QImage, QPainter, QFont, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QTimer
import threading
import logging
import datetime
import base64
import json
import configparser

# 구글 클라우드 비전 API (필수)
# 참고: google-cloud-vision 패키지가 설치되지 않은 경우 ImportError가 발생합니다.
# 설치 방법: pip install google-cloud-vision
try:
    from google.cloud import vision  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    CLOUD_VISION_AVAILABLE = True
except ImportError:
    CLOUD_VISION_AVAILABLE = False
    vision = None  # type: ignore
    service_account = None  # type: ignore
    # google-cloud-vision 패키지 미설치 경고는 logger를 통해 처리됨

def resource_path(relative_path):
    """
    Get resource path compatible with PyInstaller
    PyInstaller와 호환되는 리소스 경로 반환
    
    Args / 인자:
        relative_path (str): Relative path to resource file / 리소스 파일의 상대 경로
        
    Returns / 반환값:
        str: Absolute path to resource file / 리소스 파일의 절대 경로
    """
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle / PyInstaller 번들로 실행 중
        return os.path.join(sys._MEIPASS, relative_path)
    # Running as script / 스크립트로 실행 중
    return os.path.join(os.path.abspath("."), relative_path)

class Logger:
    """
    Logger class for application logging
    애플리케이션 로깅을 위한 로거 클래스
    
    This class manages logging for warnings and errors to a file.
    이 클래스는 경고 및 오류를 파일에 기록하는 로깅을 관리합니다.
    """
    
    def __init__(self):
        """Initialize logger / 로거 초기화"""
        self.log_file = "text_overlay_tool.log"
        self.setup_logging()
    
    def setup_logging(self):
        """
        Setup logging configuration - only errors and warnings are saved to file
        로깅 설정 - 오류 및 경고만 파일에 저장됩니다
        """
        # 로그 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러 설정 (오류 및 경고만 저장)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.WARNING)  # WARNING 이상만 저장
        
        # 로거 설정
        self.logger = logging.getLogger('TextOverlayTool')
        self.logger.setLevel(logging.WARNING)  # WARNING 이상만 처리
        self.logger.addHandler(file_handler)
        # 콘솔 핸들러 제거 (배포용)
    
    def info(self, message):
        """정보 로그"""
        self.logger.info(message)
    
    def debug(self, message):
        """디버그 로그"""
        self.logger.debug(message)
    
    def warning(self, message):
        """경고 로그"""
        self.logger.warning(message)
    
    def error(self, message):
        """에러 로그"""
        self.logger.error(message)


# 전역 로거 인스턴스
logger = Logger()


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
                # 서비스 계정 키 파일로 인증
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
            import io
            
            # 이미지 파일 읽기
            if isinstance(image_path, str):
                # 파일 경로인 경우
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')):
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                else:
                    logger.error(f"지원하지 않는 이미지 형식: {image_path}")
                    return []
            else:
                # 이미지 배열인 경우 (OpenCV 이미지)
                # PIL로 변환 후 바이트로 저장
                pil_image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                image_data = img_byte_arr.getvalue()
            
            # Cloud Vision API 호출
            image = vision.Image(content=image_data)  # type: ignore
            
            # 텍스트 감지 수행 (한국어, 일본어, 영어 지원)
            response = self.vision_client.text_detection(image=image)  # type: ignore
            
            # 응답에서 텍스트 추출
            texts = []
            if response.text_annotations:
                # 첫 번째 annotation은 전체 텍스트
                full_text = response.text_annotations[0].description
                if full_text:
                    # 개행 문자로 분리하여 리스트로 변환
                    text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                    texts.extend(text_lines)
                
            return texts
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"구글 클라우드 비전 OCR 오류: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            
            # API 오류 분류
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
                # 일반 오류는 그대로 전달
                raise Exception(f"구글 클라우드 비전 OCR 오류: {error_msg}")


class TextRegion:
    """
    Text region information storage class
    텍스트 영역 정보를 저장하는 클래스
    
    This class stores all information about a text region including position,
    styling, and formatting options.
    이 클래스는 위치, 스타일 및 포맷팅 옵션을 포함한 텍스트 영역의 모든 정보를 저장합니다.
    """
    
    def __init__(self, text="", bbox=None, font_size=18, color=(0, 0, 0), 
                 font_family="나눔고딕", margin=2, wrap_mode="word", 
                 line_spacing=1.2, bold=False, text_align="center", bg_color=None):
        self.text = text
        self.bbox = bbox if bbox is not None else (0, 0, 0, 0)  # (x1, y1, x2, y2)
        self.font_size = font_size
        self.font_family = font_family
        self.margin = margin  # 상하좌우 여백
        # 텍스트 색상: (B, G, R) 형식, 기본값 (0, 0, 0) = 검은색
        # Text color: (B, G, R) format, default (0, 0, 0) = Black
        self.color = color
        self.wrap_mode = wrap_mode  # "char" 또는 "word" - 줄바꿈 모드
        self.bold = bold  # 볼드 설정 (상세 팝업에서만 설정됨, 기본: 보통/진하게 구분)
        # 폰트 굵기 레벨: 0=보통, 1=진하게, 2=더 진하게
        self.bold_level = 1 if bold else 0
        self.line_spacing = line_spacing  # 줄간격 배율 (1.0, 1.2, 1.5, 2.0)
        self.text_align = text_align  # 텍스트 정렬: "left", "center", "right"
        # 배경색: (R, G, B, A) 형식, None이면 기본값 흰색 (255, 255, 255, 255) 사용
        # Background color: (R, G, B, A) format, if None then default white (255, 255, 255, 255)
        self.bg_color = bg_color if bg_color is not None else (255, 255, 255, 255)  # 기본값: 흰색 / Default: White
        # 텍스트 테두리: stroke_color는 (R, G, B) 형식, stroke_width는 픽셀 단위 (기본값: 없음)
        self.stroke_color = None  # None이면 테두리 없음
        self.stroke_width = 0  # 0이면 테두리 없음
        self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        self.target_bbox = None  # 타겟 이미지에서의 타겟 위치
        self.is_positioned = False  # 위치가 설정되었는지 여부
        self.image_filename = None  # 해당 텍스트 박스가 속한 이미지 파일명
        self.is_manual = False  # 수동으로 추가된 텍스트인지 여부
        self.visible = True  # 텍스트 박스 표시 여부 (기본값: 표시)


class DraggableTableWidgetItem(QtWidgets.QTableWidgetItem):
    """
    Draggable table widget item for text table
    텍스트 테이블용 드래그 가능한 테이블 위젯 아이템
    
    This class extends QTableWidgetItem to support drag and drop operations.
    이 클래스는 드래그 앤 드롭 작업을 지원하기 위해 QTableWidgetItem을 확장합니다.
    """
    
    def __init__(self, text, text_index):
        """
        Initialize draggable table item / 드래그 가능한 테이블 아이템 초기화
        
        Args / 인자:
            text (str): Text content / 텍스트 내용
            text_index (int): Index of text in regions list / 영역 목록에서의 텍스트 인덱스
        """
        super().__init__(text)
        self.text_index = text_index
    
    def clone(self):
        """
        Create clone for drag operation
        드래그 작업을 위한 클론 생성
        
        Returns / 반환값:
            DraggableTableWidgetItem: Cloned item / 복제된 아이템
        """
        return DraggableTableWidgetItem(self.text(), self.text_index)


class ImageCanvas(QtWidgets.QLabel):
    """
    Canvas for image display and text overlay editing
    이미지 표시 및 텍스트 오버레이 편집을 위한 캔버스
    
    This widget displays images and allows interactive text box positioning,
    resizing, and editing.
    이 위젯은 이미지를 표시하고 대화형 텍스트 박스 위치 지정, 크기 조정 및 편집을 허용합니다.
    """
    
    # Signals / 시그널
    region_selected = QtCore.pyqtSignal(dict)  # Region selection signal / 영역 선택 시그널
    text_dropped = QtCore.pyqtSignal(int, dict)  # Text drop signal (text_index, position) / 텍스트 드롭 시그널 (텍스트 인덱스, 위치)
    
    def __init__(self, canvas_id="", owner=None):
        """
        Initialize image canvas / 이미지 캔버스 초기화
        
        Args / 인자:
            canvas_id (str): Canvas identifier ("kr" for source, "jp" for target)
                           / 캔버스 식별자 ("kr"는 소스, "jp"는 타겟)
            owner: Reference to main window / 메인 윈도우 참조
        """
        super().__init__()
        self.canvas_id = canvas_id
        self.owner = owner  # 메인 윈도우 참조 저장
        self.image = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        
        # 텍스트 박스 편집 상태
        self.selected_text_index = -1
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.show_handles = True  # 핸들 표시 여부 (오른쪽 클릭으로 토글)
        
        # 중앙 정렬 제거 (스크롤바 지원을 위해)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #2196F3;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        self.setMinimumSize(400, 300)
        
        # 드래그 앤 드롭 허용
        self.setAcceptDrops(True)
        
        # 더블클릭 이벤트 연결
        self.mouseDoubleClickEvent = self.on_double_click
    
    def load_image(self, image_path):
        """이미지 로드"""
        try:
            # PIL로 이미지 로드 (유니코드 경로 지원)
            with Image.open(image_path) as pil_img:
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                self.image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 캐시 초기화 (이미지 크기 캐싱)
            if hasattr(self, '_img_size'):
                delattr(self, '_img_size')
            if hasattr(self, '_current_filename'):
                delattr(self, '_current_filename')
            
            self.update_display()
            return True
        except Exception as e:
            logger.error(f"이미지 로드 실패: {e}")
            return False
    
    def update_display(self):
        """이미지 표시 업데이트 (텍스트 미리보기 포함)"""
        if self.image is None:
            return
        
        # 일본어 캔버스인 경우 텍스트 미리보기와 함께 표시
        if self.canvas_id == "jp" and self.owner and hasattr(self.owner, 'text_regions'):
            # 현재 이미지의 텍스트 박스만 직접 표시 (재귀 방지, 성능 최적화)
            if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                current_filename = os.path.basename(self.owner.jp_image_path)
                # 성능 최적화: 한 번의 루프로 필터링
                current_text_regions = []
                for region in self.owner.text_regions:
                    if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                        current_text_regions.append(region)
                self.update_display_with_preview(current_text_regions)
            else:
                self.update_display_basic()
        else:
            # 한국어 캔버스는 기본 표시
            self.update_display_basic()
        
        # 확대/축소 정보 업데이트 (owner를 통해 접근)
        if self.owner:
            if self.canvas_id == "kr" and hasattr(self.owner, 'kr_zoom_label'):
                self.owner.kr_zoom_label.setText(f"🔍 확대율: {self.scale_factor:.1f}x")
            elif self.canvas_id == "jp" and hasattr(self.owner, 'jp_zoom_label'):
                self.owner.jp_zoom_label.setText(f"🔍 확대율: {self.scale_factor:.1f}x")
    
    def update_display_basic(self):
        """기본 이미지 표시 (텍스트 미리보기 없음)"""
        if self.image is None:
            return
        
        # 클라우드 비전 OCR 버전에서는 영역 선택 기능 제거 (전체 이미지 OCR만 지원)
        display_img = self.image.copy()
        
        # Qt 이미지로 변환
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        
        # 스케일링 적용 (스크롤바 지원)
        if self.scale_factor != 1.0:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            pix = pix.scaled(new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        self.setPixmap(pix)
        
        # 크기에 맞춰 라벨 크기 조정 (스크롤바 활성화를 위해)
        self.setFixedSize(pix.size())
        
        # 확대/축소 정보는 update_display()에서 처리
    
    def update_display_with_preview(self, text_regions):
        """텍스트 미리보기가 포함된 이미지 표시 (최적화된 버전)"""
        if self.image is None:
            return
        
        # 클라우드 비전 OCR 버전에서는 영역 선택 기능 제거 (전체 이미지 OCR만 지원)
        display_img = self.image.copy()
        
        # 텍스트가 있는 경우에만 PIL 변환 (성능 최적화)
        has_text_regions = any(region.is_positioned and region.target_bbox for region in text_regions)
        
        if has_text_regions:
            # RGBA → 알파 블렌딩 → RGB 변환 순서로 개선
            try:
                # 기본 이미지를 RGBA로 변환
                base_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
                # 텍스트 레이어를 RGBA로 생성
                text_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(text_layer)
                pil_img = base_img  # 기본 이미지 참조 저장
            except Exception as e:
                pil_img = None
                draw = None
        else:
            pil_img = None
            draw = None
        
        # 텍스트 미리보기 그리기 (최적화된 버전)
        for i, region in enumerate(text_regions):
            # visible 속성 확인 (기본값 True)
            if not getattr(region, 'visible', True):
                continue  # 숨김 처리된 텍스트 박스는 건너뛰기
            
            if region.is_positioned and region.target_bbox:
                x1, y1, x2, y2 = region.target_bbox
                
                # 흰색 배경은 PIL에서 처리하므로 여기서는 제거
                
                # 선택된 텍스트 박스에만 리사이즈 핸들 표시
                is_selected = False
                
                # 전체 text_regions에서의 실제 인덱스 찾기
                actual_index = -1
                if self.owner and hasattr(self.owner, 'text_regions'):
                    for j, orig_region in enumerate(self.owner.text_regions):
                        if orig_region == region:
                            actual_index = j
                            break
                
                # 캔버스에서 선택된 텍스트 인덱스 확인 (현재 이미지의 텍스트 박스만)
                if (hasattr(self, 'selected_text_index') and 
                    self.selected_text_index == actual_index and
                    hasattr(region, 'image_filename') and
                    self.owner and hasattr(self.owner, 'jp_image_path') and
                    self.owner.jp_image_path):
                    current_filename = os.path.basename(self.owner.jp_image_path)
                    if region.image_filename == current_filename:
                        is_selected = True
                
                # 테이블에서 선택된 행 확인 (현재 이미지의 텍스트 박스만)
                current_row = -1
                if (self.owner and hasattr(self.owner, 'text_table') and 
                    hasattr(self.owner.text_table, 'currentRow') and
                    hasattr(region, 'image_filename') and
                    hasattr(self.owner, 'jp_image_path') and
                    self.owner.jp_image_path):
                    current_row = self.owner.text_table.currentRow()
                    current_filename = os.path.basename(self.owner.jp_image_path)
                    if current_row == actual_index and region.image_filename == current_filename:
                        is_selected = True
                
                # 핸들은 PIL에서 처리하므로 여기서는 제거
                
                # 최적화된 한글 텍스트 렌더링 (핸들 정보 포함)
                self.draw_korean_text_optimized(display_img, pil_img, draw, region, x1, y1, x2, y2, is_selected, text_layer)
        
        # PIL 이미지가 사용된 경우 알파 블렌딩 후 최종 변환
        if pil_img is not None and draw is not None:
            try:
                # 🔥 알파 블렌딩 (투명 반올림 보존)
                blended = Image.alpha_composite(pil_img, text_layer)
                # 이제야 RGB로 변환
                display_img[:] = cv2.cvtColor(np.array(blended.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 오류 시 기본 변환
                display_img[:] = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Qt 이미지로 변환
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        
        # 스케일링 적용
        if self.scale_factor != 1.0:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            pix = pix.scaled(new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        self.setPixmap(pix)
        
        # 크기에 맞춰 라벨 크기 조정 (스크롤바 활성화를 위해)
        self.setFixedSize(pix.size())
        
        # 확대/축소 정보는 update_display()에서 처리
    
    def wrap_text(self, text, max_width, font_size):
        """초안전한 자동 줄바꿈 (최소 처리 버전)"""
        try:
            # 기본 검증
            if not text:
                return [""]
            
            # 안전한 값 보장 (최소 폭 20으로 증가)
            max_width = max(20, int(max_width)) if max_width > 0 else 100
            font_size = max(6, int(font_size)) if font_size > 0 else 12
            
            # 매우 간단한 줄바꿈 (문자 수 기반)
            chars_per_line = max(1, max_width // 8)  # 폰트 크기 무관하게 고정
            
            lines = []
            current_line = ""
            
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                elif len(current_line) >= chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            return lines if lines else [text]

        except Exception as e:
            return [text]
    
    
    def wrap_text_for_overlay(self, text, max_width, font_size):
        """오버레이용 초안전한 자동 줄바꿈 (최소 처리 버전)"""
        try:
            # 기본 검증
            if not text:
                return [""]
            
            # 안전한 값 보장
            max_width = max(10, int(max_width)) if max_width > 0 else 100
            font_size = max(6, int(font_size)) if font_size > 0 else 12
            
            # 매우 간단한 줄바꿈 (문자 수 기반)
            chars_per_line = max(1, max_width // 8)  # 폰트 크기 무관하게 고정
            
            lines = []
            current_line = ""
            
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                elif len(current_line) >= chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            return lines if lines else [text]

        except Exception as e:
            return [text]
    
    def wrap_text_for_overlay_safe(self, text, max_width, font_size, font_path="fonts/NanumGothic.ttf"):
        """PIL 충돌 없는 안전한 줄바꿈 (글자 단위, textbbox 미사용, textlength만 사용)"""
        try:
            if not text or not text.strip():
                return [""]

            max_width = max(20, int(max_width))
            font_size = max(6, int(font_size))

            # ⚠️ Dummy Image (항상 새로 생성)
            dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
            draw = ImageDraw.Draw(dummy_img)

            try:
                font = ImageFont.truetype(resource_path(font_path), font_size)
            except Exception:
                font = ImageFont.load_default()

            # 폭 계산 전용 (글자 단위 안전)
            lines = []
            current_line = ""
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                    continue

                test_line = current_line + char
                width = draw.textlength(test_line, font=font)
                if width > max_width and current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            del draw  # ⚠️ Pillow 객체 명시 해제
            return lines

        except Exception as e:
            logger.error(f"wrap_text_for_overlay_safe 오류: {e}")
            return [text]
    
    def wrap_text_for_overlay_safe_word(self, text, max_width, font_size, font):
        """PIL 충돌 없는 안전한 단어 단위 줄바꿈 (띄어쓰기 단위, 줄바꿈 문자 지원)"""
        try:
            if not text or not text.strip():
                return [""]

            max_width = max(20, int(max_width))
            font_size = max(6, int(font_size))

            # ⚠️ Dummy Image (항상 새로 생성)
            dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
            draw = ImageDraw.Draw(dummy_img)

            # 전달받은 폰트 사용
            if font is None:
                font = ImageFont.load_default()

            # 먼저 줄바꿈 문자로 분할 (사용자가 엔터키로 입력한 줄바꿈 보존)
            paragraphs = text.split('\n')
            lines = []
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    # 빈 줄은 빈 문자열로 추가
                    lines.append("")
                    continue
                
                # 각 단락을 띄어쓰기 단위로 단어 분할
                words = paragraph.split()
                current_line = ""
                
                for word in words:
                    # 현재 줄에 단어를 추가했을 때의 너비 계산
                    test_line = current_line + (" " if current_line else "") + word
                    try:
                        width = draw.textlength(test_line, font=font)
                    except Exception:
                        # textlength 실패 시 문자 수 기반 추정
                        width = len(test_line) * font_size * 0.6
                    
                    if width <= max_width:
                        current_line = test_line
                    else:
                        # 현재 줄이 너무 길면 새 줄로 이동
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # 단어 자체가 너무 긴 경우 강제로 줄바꿈
                            lines.append(word)
                            current_line = ""
                
                # 단락의 마지막 줄 추가
                if current_line:
                    lines.append(current_line)

            del draw  # ⚠️ Pillow 객체 명시 해제
            return lines if lines else [text]

        except Exception as e:
            logger.error(f"wrap_text_for_overlay_safe_word 오류: {e}")
            return [text]
    
    def draw_korean_text(self, display_img, region, x1, y1, x2, y2):
        """PIL을 사용하여 한글 텍스트 렌더링 (텍스트 박스 크기 기반)"""
        try:
            # 이미지 크기 가져오기
            img_height, img_width = display_img.shape[:2]
            
            # --- 안전 클램핑 ---
            x1 = max(0, min(int(x1), img_width - 2))
            y1 = max(0, min(int(y1), img_height - 2))
            x2 = max(x1 + 2, min(int(x2), img_width - 1))
            y2 = max(y1 + 2, min(int(y2), img_height - 1))
            
            # 폭·높이 0일 때는 그리지 않음
            if x2 - x1 < 2 or y2 - y1 < 2:
                return
            
            # 원본 이미지 복사 (원본 배열 직접 수정 방지)
            safe_display_img = display_img.copy()
            
            # PIL 이미지로 안전한 변환
            try:
                pil_img = Image.fromarray(cv2.cvtColor(safe_display_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
            except Exception as e:
                return
            
            # 텍스트 박스 크기 계산
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 폰트 크기를 박스 크기에 맞게 계산 (박스 높이의 60%로 제한)
            font_size = max(8, min(int(box_height * 0.6), int(region.font_size)))
            
            # 여백 계산 (사용자 설정 여백 사용, 음수 허용)
            margin = region.margin
            
            # 텍스트 영역 계산 (음수 여백 허용)
            text_x1 = x1 + margin
            text_y1 = y1 + margin
            text_x2 = x2 - margin
            text_y2 = y2 - margin
            
            # 텍스트 영역이 너무 작으면 최소 크기로 조정
            if text_x2 <= text_x1 or text_y2 <= text_y1:
                # 최소 크기 보장 (폰트 크기 기반)
                min_width = max(20, font_size * 2)
                min_height = max(15, font_size)
                text_x1 = x1
                text_y1 = y1
                text_x2 = max(x1 + min_width, x2)
                text_y2 = max(y1 + min_height, y2)
            
            # 사용자 설정 폰트 로드
            font = self.load_font_for_overlay(region.font_family, font_size)
            
            # 줄바꿈 계산용 너비 (음수 여백 고려)
            box_width = max(10, text_x2 - text_x1)  # 최소 너비 보장
            # 음수 여백일 때는 텍스트가 박스를 넘어갈 수 있도록 허용
            if margin < 0:
                wrap_width = box_width - (margin * 2)  # 음수 여백만큼 더 넓게
            else:
                wrap_width = box_width  # 정상 여백일 때는 박스 크기 그대로
            
            # 텍스트 줄바꿈 (줄바꿈 모드에 따라)
            if region.wrap_mode == "word":
                text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
            else:  # "char" 기본값
                text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, font)
            
            # 줄간격 계산 (사용자 설정 적용)
            base_line_height = int(font_size * 1.0)
            line_height = int(base_line_height * region.line_spacing)
            
            # 전체 텍스트 높이 계산
            total_text_height = len(text_lines) * line_height
            
            # 텍스트가 박스를 넘치면 줄간격 조정 및 폰트 크기 축소
            available_height = text_y2 - text_y1
            if total_text_height > available_height:
                # 먼저 줄간격을 최소화
                line_height = max(font_size, available_height // len(text_lines))
                total_text_height = len(text_lines) * line_height
                
                # 여전히 넘치면 폰트 크기 축소
                if total_text_height > available_height:
                    scale_factor = available_height / total_text_height
                    font_size = max(8, int(font_size * scale_factor))
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
                    
                    # 폰트 크기 변경 후 폰트 다시 로드
                    font = self.load_font_for_overlay(region.font_family, font_size)
                    
                    # 줄바꿈 다시 계산 (새로운 폰트 크기로)
                    if region.wrap_mode == "word":
                        text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                    else:  # "char" 기본값
                        text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, font)
                    
                    # 줄 수가 변경되었으므로 높이 재계산
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
            
            # 텍스트 시작 위치 계산 (정확한 중앙 정렬) - 상단 여백 제거
            start_y = text_y1 + (available_height - total_text_height) // 2
            
            # 텍스트 색상 설정 (BGR → RGB)
            text_color = (region.color[2], region.color[1], region.color[0])
            
            # 각 줄의 텍스트 그리기
            for line_idx, line_text in enumerate(text_lines):
                if line_text.strip():
                    # 텍스트 너비 계산
                    try:
                        text_width = draw.textlength(line_text, font=font)
                    except Exception:
                        text_width = len(line_text) * font_size * 0.6
                    
                    # 텍스트 위치 계산 (정렬 적용)
                    text_align = getattr(region, 'text_align', 'center')
                    if text_align == "left":
                        text_x = text_x1
                    elif text_align == "right":
                        text_x = text_x2 - text_width
                    else:  # "center"
                        text_x = text_x1 + (text_x2 - text_x1 - text_width) // 2
                    text_y = start_y + line_idx * line_height
                    
                    # 텍스트가 박스를 넘치지 않도록 확인 (하단 잘림 방지, 20px 허용)
                    tolerance = 20
                    if text_x >= text_x1 - tolerance and text_x + text_width <= text_x2 + tolerance and text_y <= text_y2 + tolerance:
                        # 텍스트가 박스 내에 완전히 들어가는지 확인 (5px 허용)
                        if text_y + font_size <= text_y2 + tolerance:
                            # 테두리 적용
                            stroke_color = getattr(region, 'stroke_color', None)
                            stroke_width = getattr(region, 'stroke_width', 0)
                            if stroke_color is not None and stroke_width > 0:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color, 
                                         stroke_width=stroke_width, stroke_fill=stroke_color)
                            else:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color)
                        else:
                            # 텍스트가 박스를 넘치면 잘린 부분 표시
                            truncated_text = line_text
                            while truncated_text and text_y + font_size > text_y2 + tolerance:
                                truncated_text = truncated_text[:-1]
                                if truncated_text:
                                    try:
                                        truncated_width = draw.textlength(truncated_text + "...", font=font)
                                    except Exception:
                                        truncated_width = len(truncated_text + "...") * font_size * 0.6
                                    text_x = text_x1 + (text_x2 - text_x1 - truncated_width) // 2
                            
                            if truncated_text:
                                # 테두리 적용
                                stroke_color = getattr(region, 'stroke_color', None)
                                stroke_width = getattr(region, 'stroke_width', 0)
                                if stroke_color is not None and stroke_width > 0:
                                    draw.text((text_x, text_y), truncated_text + "...", font=font, fill=text_color,
                                             stroke_width=stroke_width, stroke_fill=stroke_color)
                                else:
                                    draw.text((text_x, text_y), truncated_text + "...", font=font, fill=text_color)
            
            # PIL 이미지를 OpenCV 형식으로 변환
            display_img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"한글 텍스트 렌더링 오류: {e}")
            # 오류 시 기본 텍스트 표시
            cv2.putText(display_img, region.text, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def draw_korean_text_optimized(self, display_img, pil_img, draw, region, x1, y1, x2, y2, is_selected=False, text_layer=None):
        """최적화된 한글 텍스트 렌더링 (텍스트 박스 크기 기반)"""
        try:
            # 이미지 크기 가져오기
            img_height, img_width = display_img.shape[:2]
            
            # --- 안전 클램핑 ---
            x1 = max(0, min(int(x1), img_width - 2))
            y1 = max(0, min(int(y1), img_height - 2))
            x2 = max(x1 + 2, min(int(x2), img_width - 1))
            y2 = max(y1 + 2, min(int(y2), img_height - 1))
            
            # 폭·높이 0일 때는 그리지 않음
            if x2 - x1 < 2 or y2 - y1 < 2:
                return
            
            # PIL 이미지가 없는 경우 기본 렌더링
            if pil_img is None or draw is None:
                cv2.putText(display_img, region.text, (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                return
            
            # 텍스트 박스 크기 계산
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 폰트 크기를 박스 크기에 맞게 계산 (박스 높이의 60%로 제한)
            font_size = max(8, min(int(box_height * 0.6), int(region.font_size)))
            
            # 여백 계산 (사용자 설정 여백 사용, 음수 허용)
            margin = region.margin
            
            # 텍스트 영역 계산 (음수 여백 허용)
            text_x1 = x1 + margin
            text_y1 = y1 + margin
            text_x2 = x2 - margin
            text_y2 = y2 - margin
            
            # 텍스트 영역이 너무 작으면 최소 크기로 조정
            if text_x2 <= text_x1 or text_y2 <= text_y1:
                # 최소 크기 보장 (폰트 크기 기반)
                min_width = max(20, font_size * 2)
                min_height = max(15, font_size)
                text_x1 = x1
                text_y1 = y1
                text_x2 = max(x1 + min_width, x2)
                text_y2 = max(y1 + min_height, y2)
            
            # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
            bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
            if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            
            # 사용자 설정 폰트 로드
            font = self.load_font_for_overlay(region.font_family, font_size)
            
            # 화면 표시에서도 bold 설정 적용
            if hasattr(region, 'bold') and region.bold:
                # PIL 폰트는 bold 속성을 직접 지원하지 않으므로 폰트 크기를 약간 키워서 진하게 표시
                bold_font_size = int(font_size * 1.1)  # 10% 크게
                try:
                    font = self.load_font_for_overlay(region.font_family, bold_font_size)
                except:
                    pass  # 폰트 로딩 실패 시 원본 폰트 사용
            
            # 줄바꿈 계산용 너비 (음수 여백 고려)
            box_width = max(10, text_x2 - text_x1)  # 최소 너비 보장
            # 음수 여백일 때는 텍스트가 박스를 넘어갈 수 있도록 허용
            if margin < 0:
                wrap_width = box_width - (margin * 2)  # 음수 여백만큼 더 넓게
            else:
                wrap_width = box_width  # 정상 여백일 때는 박스 크기 그대로
            
            # 텍스트 줄바꿈 (줄바꿈 모드에 따라)
            if region.wrap_mode == "word":
                text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
            else:  # "char" 기본값
                text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, font)
            
            # 줄간격 계산 (사용자 설정 적용, 폰트가 안 잘리도록 20% 여유 증가)
            base_line_height = int(font_size * 1.0)
            line_height = int(base_line_height * region.line_spacing)
            
            # 전체 텍스트 높이 계산
            total_text_height = len(text_lines) * line_height
            
            # 텍스트가 박스를 넘치면 줄간격 조정 및 폰트 크기 축소
            available_height = text_y2 - text_y1
            if total_text_height > available_height:
                # 먼저 줄간격을 최소화
                line_height = max(font_size, available_height // len(text_lines))
                total_text_height = len(text_lines) * line_height
                
                # 여전히 넘치면 폰트 크기 축소
                if total_text_height > available_height:
                    scale_factor = available_height / total_text_height
                    font_size = max(8, int(font_size * scale_factor))
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
                    
                    # 폰트 크기 변경 후 폰트 다시 로드
                    font = self.load_font_for_overlay(region.font_family, font_size)
                    
                    # 줄바꿈 다시 계산 (새로운 폰트 크기로)
                    if region.wrap_mode == "word":
                        text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                    else:  # "char" 기본값
                        text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, font)
                    
                    # 줄 수가 변경되었으므로 높이 재계산
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
            
            # 텍스트 시작 위치 계산 (정확한 중앙 정렬) - 상단 여백 제거
            start_y = text_y1 + (available_height - total_text_height) // 2
            
            # 텍스트 색상 설정 (BGR → RGB)
            text_color = (region.color[2], region.color[1], region.color[0])
            
            # 각 줄의 텍스트 그리기
            for line_idx, line_text in enumerate(text_lines):
                if line_text.strip():
                    # 텍스트 너비 계산
                    try:
                        text_width = draw.textlength(line_text, font=font)
                    except Exception:
                        text_width = len(line_text) * font_size * 0.6
                    
                    # 텍스트 위치 계산 (정렬 적용)
                    text_align = getattr(region, 'text_align', 'center')
                    if text_align == "left":
                        text_x = text_x1
                    elif text_align == "right":
                        text_x = text_x2 - text_width
                    else:  # "center"
                        text_x = text_x1 + (text_x2 - text_x1 - text_width) // 2
                    text_y = start_y + line_idx * line_height
                    
                    # 텍스트가 박스를 넘치지 않도록 확인 (하단 잘림 방지, 20px 허용)
                    tolerance = 20
                    if text_x >= text_x1 - tolerance and text_x + text_width <= text_x2 + tolerance and text_y <= text_y2 + tolerance:
                        # 텍스트가 박스 내에 완전히 들어가는지 확인 (5px 허용)
                        if text_y + font_size <= text_y2 + tolerance:
                            # 테두리 적용
                            stroke_color = getattr(region, 'stroke_color', None)
                            stroke_width = getattr(region, 'stroke_width', 0)
                            if stroke_color is not None and stroke_width > 0:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color, 
                                         stroke_width=stroke_width, stroke_fill=stroke_color)
                            else:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color)
                        else:
                            # 텍스트가 박스를 넘치면 잘린 부분 표시
                            truncated_text = line_text
                            while truncated_text and text_y + font_size > text_y2 + tolerance:
                                truncated_text = truncated_text[:-1]
                                if truncated_text:
                                    try:
                                        truncated_width = draw.textlength(truncated_text + "...", font=font)
                                    except Exception:
                                        truncated_width = len(truncated_text + "...") * font_size * 0.6
                                    text_x = text_x1 + (text_x2 - text_x1 - truncated_width) // 2
                            
                            if truncated_text:
                                # 테두리 적용
                                stroke_color = getattr(region, 'stroke_color', None)
                                stroke_width = getattr(region, 'stroke_width', 0)
                                if stroke_color is not None and stroke_width > 0:
                                    draw.text((text_x, text_y), truncated_text + "...", font=font, fill=text_color,
                                             stroke_width=stroke_width, stroke_fill=stroke_color)
                                else:
                                    draw.text((text_x, text_y), truncated_text + "...", font=font, fill=text_color)
            
            # 선택된 텍스트 박스에 핸들 그리기 (show_handles가 True일 때만)
            if is_selected and hasattr(self, 'show_handles') and self.show_handles:
                handle_size = min(15, min(box_width, box_height) // 4)
                handle_color = (0, 0, 0, 255)
                
                # 네 모서리 핸들 그리기
                draw.rectangle([x2 - handle_size, y2 - handle_size, x2, y2], fill=handle_color)
                draw.rectangle([x2 - handle_size, y1, x2, y1 + handle_size], fill=handle_color)
                draw.rectangle([x1, y2 - handle_size, x1 + handle_size, y2], fill=handle_color)
                draw.rectangle([x1, y1, x1 + handle_size, y1 + handle_size], fill=handle_color)
            
        except Exception as e:
            logger.error(f"최적화된 한글 텍스트 렌더링 오류: {e}")
            # 오류 시 기본 텍스트 표시
            cv2.putText(display_img, region.text, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    
    def wrap_text_for_box(self, text, max_width, font_size, font):
        """텍스트 박스에 맞는 줄바꿈 (한글 지원)"""
        try:
            if not text or not text.strip():
                return [""]
            
            # 한글과 영문을 구분하여 처리
            lines = []
            current_line = ""
            
            i = 0
            while i < len(text):
                char = text[i]
                
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                    i += 1
                    continue
                
                # 한글, 영문, 숫자, 특수문자 구분
                if self._is_korean(char):
                    # 한글은 문자 단위로 처리
                    test_line = current_line + char
                elif char.isspace():
                    # 공백은 단어 경계로 처리
                    test_line = current_line + char
                else:
                    # 영문/숫자는 단어 단위로 처리
                    word = ""
                    while i < len(text) and not text[i].isspace() and not self._is_korean(text[i]):
                        word += text[i]
                        i += 1
                    i -= 1  # 다음 루프에서 올바른 위치에서 시작
                    test_line = current_line + word
                
                # 텍스트 너비 측정
                try:
                    temp_img = Image.new("L", (max_width * 2, font_size * 2), color=0)
                    temp_draw = ImageDraw.Draw(temp_img)
                    width = temp_draw.textlength(test_line, font=font)
                    
                    if width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = char if self._is_korean(char) else word
                        else:
                            # 단어/문자가 너무 긴 경우 강제로 줄바꿈
                            lines.append(char if self._is_korean(char) else word)
                            current_line = ""
                except Exception:
                    # textlength 실패 시 문자 수 기반 추정
                    estimated_width = len(test_line) * font_size * 0.6
                    if estimated_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = char if self._is_korean(char) else word
                        else:
                            lines.append(char if self._is_korean(char) else word)
                            current_line = ""
                
                i += 1
            
            if current_line:
                lines.append(current_line)
            
            return lines if lines else [text]
            
        except Exception as e:
            logger.error(f"줄바꿈 처리 오류: {e}")
            return [text]
    
    def _is_korean(self, char):
        """한글 문자인지 확인"""
        return '\uAC00' <= char <= '\uD7AF' or '\u1100' <= char <= '\u11FF' or '\u3130' <= char <= '\u318F'
    
    def wrap_text_for_overlay_safe(self, text, max_width, font_size, font_path="fonts/NanumGothic.ttf"):
        """PIL 충돌 없는 안전한 줄바꿈 (글자 단위, textbbox 미사용, textlength만 사용)"""
        try:
            if not text or not text.strip():
                return [""]

            max_width = max(20, int(max_width))
            font_size = max(6, int(font_size))

            # ⚠️ Dummy Image (항상 새로 생성)
            dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
            draw = ImageDraw.Draw(dummy_img)

            try:
                font = ImageFont.truetype(resource_path(font_path), font_size)
            except Exception:
                font = ImageFont.load_default()

            # 폭 계산 전용 (글자 단위 안전)
            lines = []
            current_line = ""
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                    continue

                test_line = current_line + char
                try:
                    width = draw.textlength(test_line, font=font)
                except Exception:
                    # textlength 실패 시 문자 수 기반 추정
                    width = len(test_line) * font_size * 0.6
                
                if width > max_width and current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            del draw  # ⚠️ Pillow 객체 명시 해제
            return lines if lines else [text]

        except Exception as e:
            logger.error(f"wrap_text_for_overlay_safe 오류: {e}")
            return [text]
    
    def wheelEvent(self, event):
        """마우스 휠로 확대/축소"""
        if self.image is None:
            return
        
        delta = event.angleDelta().y()
        old_scale = self.scale_factor
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor *= 0.9
        
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))
        
        self.update_display()
        event.accept()
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 (간소화된 버전)"""
        if self.image is None:
            return
        
        # 오른쪽 마우스 클릭: 핸들 표시 토글
        if event.button() == QtCore.Qt.RightButton:
            if self.canvas_id == "jp":
                self.show_handles = not self.show_handles
                # 빠른 업데이트: 테이블 업데이트 없이 캔버스만 업데이트
                if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                    current_filename = os.path.basename(self.owner.jp_image_path)
                    current_text_regions = []
                    for region in self.owner.text_regions:
                        if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                            current_text_regions.append(region)
                    if hasattr(self, 'update_display_with_preview'):
                        self.update_display_with_preview(current_text_regions)
                return
        
        img_pos = self._get_image_position(event.pos())
        if img_pos is None:
            return
        
        # 일본어 캔버스에서 텍스트 박스 편집
        if self.canvas_id == "jp" and self.owner and hasattr(self.owner, 'text_regions'):
            clicked_text_index = self.get_text_at_position(img_pos)
            if clicked_text_index >= 0:
                self.selected_text_index = clicked_text_index
                
                # 리사이즈 핸들 확인 (우선순위)
                handle = self.get_resize_handle(img_pos, clicked_text_index)
                if handle:
                    self.resizing = True
                    self.resize_handle = handle
                else:
                    self.moving = True
                
                # 텍스트 테이블에서 해당 행 선택
                if hasattr(self.owner, 'text_table'):
                    self.owner.text_table.selectRow(clicked_text_index)
                
                # 빠른 업데이트: 테이블 업데이트 없이 캔버스만 업데이트
                if hasattr(self.owner, 'text_regions') and hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                    current_filename = os.path.basename(self.owner.jp_image_path)
                    current_text_regions = []
                    for region in self.owner.text_regions:
                        if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                            current_text_regions.append(region)
                    if hasattr(self, 'update_display_with_preview'):
                        self.update_display_with_preview(current_text_regions)
                return
        
        # 클라우드 비전 OCR 버전에서는 영역 선택 기능 제거 (전체 이미지 OCR만 지원)
        # 한국어 캔버스에서는 영역 선택 비활성화
        pass
    
    def mouseMoveEvent(self, event):
        """마우스 드래그 이벤트 (간소화된 버전)"""
        if self.image is None:
            return
        
        img_pos = self._get_image_position(event.pos())
        if img_pos is None:
            return
        
        # 텍스트 박스 편집 (최적화된 throttle 적용)
        if self.selected_text_index >= 0:
            try:
                if self.resizing and self.resize_handle:
                    # 5ms 단위로만 업데이트 (더 빠른 반응성)
                    current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
                    if not hasattr(self, '_last_resize_update') or (current_time - self._last_resize_update >= 5):
                        self._last_resize_update = current_time
                        self.resize_text_box(img_pos)
                elif self.moving:
                    # 5ms 단위로만 업데이트 (더 빠른 반응성)
                    current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
                    if not hasattr(self, '_last_move_update') or (current_time - self._last_move_update >= 5):
                        self._last_move_update = current_time
                        self.move_text_box(img_pos)
            except Exception as e:
                # 오류 발생 시 편집 모드 종료
                self.resizing = False
                self.moving = False
                self.resize_handle = None
            return
        
        # 제미나이 OCR 버전에서는 영역 선택 기능 제거
        pass
    
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트 (간소화된 버전)"""
        if self.image is None:
            return
        
        # 텍스트 박스 편집에서 드래그 종료
        if self.selected_text_index >= 0:
            self.resizing = False
            self.moving = False
            self.resize_handle = None
            # 드래그 시작 위치 초기화
            if hasattr(self, 'drag_start_pos'):
                delattr(self, 'drag_start_pos')
            if hasattr(self, 'drag_start_bbox'):
                delattr(self, 'drag_start_bbox')
            # 리사이즈 시작 위치 초기화
            if hasattr(self, 'resize_start_pos'):
                delattr(self, 'resize_start_pos')
            if hasattr(self, 'resize_start_bbox'):
                delattr(self, 'resize_start_bbox')
            return
        
        # 일반 영역 선택 모드
        if not self.drawing:
            return
        
        self.drawing = False
        
        img_pos = self._get_image_position(event.pos())
        if img_pos is None:
            return
        
        self.end_point = img_pos
        
        # 클라우드 비전 OCR 버전에서는 영역 선택 기능 제거 (전체 이미지 OCR만 지원)
        pass
        
        # 드래그 상태 초기화
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.update_display()
    
    def _get_image_position(self, widget_pos):
        """위젯 좌표를 이미지 좌표로 변환 (개선된 버전)"""
        if self.image is None:
            return None
        
        try:
            # 이미지의 원본 크기
            img_height, img_width = self.image.shape[:2]
            
            # QLabel에 표시된 픽스맵의 크기 (스케일링 적용됨)
            pixmap = self.pixmap()
            if pixmap is None:
                return None
            
            pixmap_size = pixmap.size()
            
            # QLabel의 실제 크기
            label_size = self.size()
            
            # QLabel 내에서 픽스맵이 중앙 정렬되어 있으므로 오프셋 계산
            offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
            offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
            
            # 위젯 좌표에서 픽스맵 좌표로 변환
            pixmap_x = widget_pos.x() - offset_x
            pixmap_y = widget_pos.y() - offset_y
            
            # 픽스맵 좌표가 유효한 범위 내에 있는지 확인
            if 0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height():
                # 픽스맵 좌표를 원본 이미지 좌표로 변환
                img_x = int(pixmap_x * img_width / pixmap_size.width())
                img_y = int(pixmap_y * img_height / pixmap_size.height())
                
                # 이미지 범위 내에 있는지 확인
                if 0 <= img_x < img_width and 0 <= img_y < img_height:
                    return (img_x, img_y)
            
            return None
            
        except Exception as e:
            logger.error(f"좌표 변환 오류: {e}")
            return None
    
    def dragEnterEvent(self, event):
        """드래그 진입 이벤트"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """드래그 이동 이벤트"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """드롭 이벤트"""
        if event.mimeData().hasText():
            # 드롭된 텍스트에서 인덱스 추출
            try:
                text_data = event.mimeData().text()
                if text_data.startswith("text_index:"):
                    text_index = int(text_data.split(":")[1])
                    
                    # 드롭 위치를 이미지 좌표로 변환
                    img_pos = self._get_image_position(event.pos())
                    if img_pos is not None:
                        # 텍스트 크기 설정 (기본값) - 좌우 폭만 줄임
                        text_width = 120  # 200에서 150으로 줄임 (25% 감소)
                        text_height = 50
                        
                        x, y = img_pos
                        target_bbox = (x - text_width//2, y - text_height//2, 
                                     x + text_width//2, y + text_height//2)
                        
                        # 이미지 범위 내로 제한
                        if self.image is not None:
                            img_h, img_w = self.image.shape[:2]
                            target_bbox = (
                                max(0, min(target_bbox[0], img_w - text_width)),
                                max(0, min(target_bbox[1], img_h - text_height)),
                                max(text_width, min(target_bbox[2], img_w)),
                                max(text_height, min(target_bbox[3], img_h))
                            )
                        
                        self.text_dropped.emit(text_index, {'bbox': target_bbox})
                        event.acceptProposedAction()
                        return
            except (ValueError, IndexError):
                pass
        
        event.ignore()
    
    def on_double_click(self, event):
        """더블클릭 이벤트 - 텍스트 박스 편집"""
        if self.canvas_id == "jp" and self.owner and hasattr(self.owner, 'text_regions'):
            img_pos = self._get_image_position(event.pos())
            if img_pos is not None:
                clicked_text_index = self.get_text_at_position(img_pos)
                if clicked_text_index >= 0:
                    # 핸들 영역인지 확인 - 핸들 근처를 클릭한 경우 편집 다이얼로그를 열지 않음
                    handle = self.get_resize_handle(img_pos, clicked_text_index)
                    if handle:
                        # 핸들 영역을 더블클릭한 경우 편집하지 않음
                        return
                    
                    # 텍스트 테이블에서 해당 행 선택
                    if hasattr(self.owner, 'text_table'):
                        self.owner.text_table.selectRow(clicked_text_index)
                        if hasattr(self.owner, 'text_regions'):
                            # 현재 이미지의 텍스트 박스만 표시
                            if hasattr(self.owner, 'update_display_for_current_image'):
                                self.owner.update_display_for_current_image()
                        
                        # 텍스트 편집 대화상자 열기
                        self.edit_text_dialog(clicked_text_index)
    
    def edit_text_dialog(self, text_index):
        """텍스트 편집 대화상자"""
        if not self.owner or not hasattr(self.owner, 'text_regions') or text_index < 0 or text_index >= len(self.owner.text_regions):
            return
        
        region = self.owner.text_regions[text_index]
        
        dialog = QtWidgets.QDialog(None)
        dialog.setWindowTitle("텍스트 박스 설정")
        dialog.setModal(True)
        dialog.setMinimumWidth(500)  # 다이얼로그 최소 너비 설정
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 텍스트 입력
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.addWidget(QtWidgets.QLabel("텍스트:"))
        text_edit = QtWidgets.QTextEdit(region.text)
        text_edit.setMinimumHeight(150)  # 에디터처럼 보이도록 높이 증가
        text_edit.setMaximumHeight(400)  # 최대 높이 제한
        text_edit.setAcceptRichText(False)  # 일반 텍스트만 허용
        text_edit.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)  # 줄바꿈 모드
        # 에디터 스타일 적용
        text_edit.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                font-family: '맑은 고딕', 'Malgun Gothic', monospace;
                font-size: 11pt;
                background-color: #fafafa;
            }
            QTextEdit:focus {
                border: 2px solid #2196F3;
                background-color: white;
            }
        """)
        text_layout.addWidget(text_edit)
        layout.addLayout(text_layout)
        
        # 엔터키가 다이얼로그를 닫지 않도록 키 이벤트 오버라이드
        original_keyPressEvent = dialog.keyPressEvent
        def keyPressEvent(event):
            # QTextEdit에 포커스가 있을 때는 엔터키를 다이얼로그가 처리하지 않음
            if text_edit.hasFocus():
                if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    # Ctrl+Enter 또는 Cmd+Enter는 다이얼로그 닫기
                    if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                        dialog.accept()
                        return
                    # 일반 Enter는 줄바꿈 (QTextEdit에 직접 이벤트 전달)
                    QtWidgets.QTextEdit.keyPressEvent(text_edit, event)
                    return
            # 다른 위젯에 포커스가 있을 때는 기본 동작
            original_keyPressEvent(event)
        
        dialog.keyPressEvent = keyPressEvent
        
        # 폰트 크기
        font_size_layout = QtWidgets.QHBoxLayout()
        font_size_layout.addWidget(QtWidgets.QLabel("폰트 크기:"))
        font_size_spin = QtWidgets.QSpinBox()
        font_size_spin.setRange(6, 200)
        font_size_spin.setValue(region.font_size)
        font_size_layout.addWidget(font_size_spin)
        layout.addLayout(font_size_layout)
        
        # 폰트 패밀리
        font_family_layout = QtWidgets.QHBoxLayout()
        font_family_layout.addWidget(QtWidgets.QLabel("폰트:"))
        font_combo = QtWidgets.QComboBox()
        
        # 기본 폰트 목록
        default_fonts = ["Arial", "Times New Roman", "Courier New", "굴림", "맑은 고딕", "나눔고딕"]
        font_combo.addItems(default_fonts)
        
        # 사용자 추가 폰트 추가
        if self.owner and hasattr(self.owner, 'custom_fonts'):
            for custom_font_name in sorted(self.owner.custom_fonts.keys()):
                if custom_font_name not in default_fonts:
                    font_combo.addItem(f"⭐ {custom_font_name}")  # 사용자 추가 폰트 표시
        
        # 현재 폰트 설정
        current_font = region.font_family if region.font_family else "나눔고딕"
        # 사용자 추가 폰트인 경우 "⭐ " 접두사 확인
        if current_font in (self.owner.custom_fonts.keys() if self.owner and hasattr(self.owner, 'custom_fonts') else []):
            if current_font not in default_fonts:
                current_font = f"⭐ {current_font}"
        font_combo.setCurrentText(current_font if current_font in [font_combo.itemText(i) for i in range(font_combo.count())] else "나눔고딕")
        font_family_layout.addWidget(font_combo)
        layout.addLayout(font_family_layout)
        
        # 여백 설정
        margin_layout = QtWidgets.QHBoxLayout()
        margin_layout.addWidget(QtWidgets.QLabel("여백:"))
        margin_spin = QtWidgets.QSpinBox()
        margin_spin.setRange(-50, 50)  # 음수 여백 허용
        margin_spin.setValue(region.margin)
        margin_spin.setSuffix("px")
        margin_layout.addWidget(margin_spin)
        layout.addLayout(margin_layout)
        
        # 색상 설정
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("색상:"))
        color_btn = QtWidgets.QPushButton("색상 선택")
        color_btn.clicked.connect(lambda: self.choose_color_for_region(color_btn, region))
        color_layout.addWidget(color_btn)
        layout.addLayout(color_layout)
        
        # 줄바꿈 모드 설정
        wrap_layout = QtWidgets.QHBoxLayout()
        wrap_layout.addWidget(QtWidgets.QLabel("줄바꿈 모드:"))
        wrap_combo = QtWidgets.QComboBox()
        wrap_combo.addItems(["글자 단위", "단어 단위"])
        wrap_combo.setCurrentText("글자 단위" if region.wrap_mode == "char" else "단어 단위")
        wrap_layout.addWidget(wrap_combo)
        layout.addLayout(wrap_layout)
        
        # 줄간격 설정
        line_spacing_layout = QtWidgets.QHBoxLayout()
        line_spacing_layout.addWidget(QtWidgets.QLabel("줄간격:"))
        line_spacing_combo = QtWidgets.QComboBox()
        line_spacing_combo.addItems(["1.0", "1.2", "1.5", "2.0"])
        line_spacing_combo.setCurrentText(str(region.line_spacing))
        line_spacing_layout.addWidget(line_spacing_combo)
        layout.addLayout(line_spacing_layout)
        
        # 폰트 굵기 설정
        bold_layout = QtWidgets.QHBoxLayout()
        bold_layout.addWidget(QtWidgets.QLabel("폰트 굵기:"))
        bold_combo = QtWidgets.QComboBox()
        bold_combo.addItems(["보통", "진하게", "더 진하게"])
        # region에 bold 속성이 없으면 기본값으로 False 설정
        if not hasattr(region, 'bold'):
            region.bold = False
        # bold_level이 있으면 우선 사용 (0=보통, 1=진하게, 2=더 진하게)
        if not hasattr(region, 'bold_level'):
            region.bold_level = 1 if region.bold else 0
        bold_map = {0: "보통", 1: "진하게", 2: "더 진하게"}
        bold_combo.setCurrentText(bold_map.get(region.bold_level, "보통"))
        bold_layout.addWidget(bold_combo)
        layout.addLayout(bold_layout)
        
        # 텍스트 정렬 설정
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.addWidget(QtWidgets.QLabel("텍스트 정렬:"))
        align_combo = QtWidgets.QComboBox()
        align_combo.addItems(["왼쪽 정렬", "가운데 정렬", "오른쪽 정렬"])
        # region에 text_align 속성이 없으면 기본값으로 "center" 설정
        if not hasattr(region, 'text_align'):
            region.text_align = "center"
        align_map = {"left": "왼쪽 정렬", "center": "가운데 정렬", "right": "오른쪽 정렬"}
        align_combo.setCurrentText(align_map.get(region.text_align, "가운데 정렬"))
        align_layout.addWidget(align_combo)
        layout.addLayout(align_layout)
        
        # 배경색 설정
        bg_color_layout = QtWidgets.QVBoxLayout()
        bg_color_layout.addWidget(QtWidgets.QLabel("배경색:"))
        
        bg_color_h_layout = QtWidgets.QHBoxLayout()
        
        # 배경색 선택 버튼
        bg_color_btn = QtWidgets.QPushButton("배경색 선택")
        
        # 배경색 초기화 (region에 bg_color가 없으면 기본값 흰색)
        if not hasattr(region, 'bg_color') or region.bg_color is None:
            region.bg_color = (255, 255, 255, 255)
        
        # 현재 배경색으로 버튼 스타일 설정
        bg_r, bg_g, bg_b, bg_a = region.bg_color
        bg_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({bg_r}, {bg_g}, {bg_b});
                color: {'white' if (bg_r + bg_g + bg_b) < 384 else 'black'};
                border: 2px solid #ccc;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border: 2px solid #2196F3;
            }}
        """)
        
        # 투명 체크박스
        transparent_checkbox = QtWidgets.QCheckBox("투명")
        transparent_checkbox.setChecked(bg_a == 0)
        
        def choose_bg_color():
            """배경색 선택 다이얼로그"""
            current_bg = region.bg_color if hasattr(region, 'bg_color') and region.bg_color else (255, 255, 255, 255)
            # QColorDialog는 RGB만 지원하므로 RGBA에서 RGB 추출
            qcolor = QColor(current_bg[0], current_bg[1], current_bg[2])
            color = QtWidgets.QColorDialog.getColor(qcolor, None, "배경색 선택")
            if color.isValid():
                # 투명 체크박스가 체크되어 있으면 알파를 0으로, 아니면 255로
                alpha = 0 if transparent_checkbox.isChecked() else 255
                region.bg_color = (color.red(), color.green(), color.blue(), alpha)
                # 버튼 스타일 업데이트
                bg_color_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                        color: {'white' if color.lightness() < 128 else 'black'};
                        border: 2px solid #ccc;
                        border-radius: 4px;
                        padding: 5px 10px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        border: 2px solid #2196F3;
                    }}
                """)
        
        def on_transparent_changed(checked):
            """투명 체크박스 변경 시"""
            if checked:
                # 투명으로 설정 (알파를 0으로)
                if hasattr(region, 'bg_color') and region.bg_color:
                    r, g, b, _ = region.bg_color
                    region.bg_color = (r, g, b, 0)
            else:
                # 불투명으로 설정 (알파를 255로)
                if hasattr(region, 'bg_color') and region.bg_color:
                    r, g, b, _ = region.bg_color
                    region.bg_color = (r, g, b, 255)
                    # 버튼 스타일 업데이트
                    bg_color_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: rgb({r}, {g}, {b});
                            color: {'white' if (r + g + b) < 384 else 'black'};
                            border: 2px solid #ccc;
                            border-radius: 4px;
                            padding: 5px 10px;
                            font-weight: bold;
                        }}
                        QPushButton:hover {{
                            border: 2px solid #2196F3;
                        }}
                    """)
        
        bg_color_btn.clicked.connect(choose_bg_color)
        transparent_checkbox.stateChanged.connect(on_transparent_changed)
        
        bg_color_h_layout.addWidget(bg_color_btn)
        bg_color_h_layout.addWidget(transparent_checkbox)
        bg_color_h_layout.addStretch()
        
        bg_color_layout.addLayout(bg_color_h_layout)
        layout.addLayout(bg_color_layout)
        
        # 텍스트 테두리 설정
        stroke_layout = QtWidgets.QVBoxLayout()
        stroke_layout.addWidget(QtWidgets.QLabel("텍스트 테두리:"))
        
        stroke_h_layout = QtWidgets.QHBoxLayout()
        
        # 테두리 색상 선택 버튼
        stroke_color_btn = QtWidgets.QPushButton("테두리 색상 선택")
        
        # 테두리 초기화 (region에 stroke_color가 없으면 기본값 없음)
        if not hasattr(region, 'stroke_color') or region.stroke_color is None:
            region.stroke_color = None
        if not hasattr(region, 'stroke_width'):
            region.stroke_width = 0
        
        # 현재 테두리 색상으로 버튼 스타일 설정
        if region.stroke_color is not None and region.stroke_width > 0:
            stroke_r, stroke_g, stroke_b = region.stroke_color
            stroke_color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({stroke_r}, {stroke_g}, {stroke_b});
                    color: {'white' if (stroke_r + stroke_g + stroke_b) < 384 else 'black'};
                    border: 2px solid #ccc;
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    border: 2px solid #2196F3;
                }}
            """)
        else:
            stroke_color_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    color: #666;
                    border: 2px solid #ccc;
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border: 2px solid #2196F3;
                }
            """)
        
        # 테두리 두께 설정
        stroke_width_label = QtWidgets.QLabel("두께:")
        stroke_width_spin = QtWidgets.QSpinBox()
        stroke_width_spin.setRange(0, 20)
        stroke_width_spin.setValue(region.stroke_width if hasattr(region, 'stroke_width') else 0)
        stroke_width_spin.setSuffix("px")
        
        def choose_stroke_color():
            """테두리 색상 선택 다이얼로그"""
            current_stroke = region.stroke_color if hasattr(region, 'stroke_color') and region.stroke_color else (0, 0, 0)
            qcolor = QColor(current_stroke[0], current_stroke[1], current_stroke[2])
            color = QtWidgets.QColorDialog.getColor(qcolor, None, "테두리 색상 선택")
            if color.isValid():
                region.stroke_color = (color.red(), color.green(), color.blue())
                # 두께가 0이면 1로 설정
                if region.stroke_width == 0:
                    region.stroke_width = 1
                    stroke_width_spin.setValue(1)
                # 버튼 스타일 업데이트
                stroke_color_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                        color: {'white' if color.lightness() < 128 else 'black'};
                        border: 2px solid #ccc;
                        border-radius: 4px;
                        padding: 5px 10px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        border: 2px solid #2196F3;
                    }}
                """)
        
        def on_stroke_width_changed(value):
            """테두리 두께 변경 시"""
            region.stroke_width = value
            if value == 0:
                # 두께가 0이면 테두리 색상도 None으로
                region.stroke_color = None
                stroke_color_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0f0f0;
                        color: #666;
                        border: 2px solid #ccc;
                        border-radius: 4px;
                        padding: 5px 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        border: 2px solid #2196F3;
                    }
                """)
            elif region.stroke_color is None:
                # 두께가 설정되었는데 색상이 없으면 검은색으로 기본 설정
                region.stroke_color = (0, 0, 0)
                stroke_color_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(0, 0, 0);
                        color: white;
                        border: 2px solid #ccc;
                        border-radius: 4px;
                        padding: 5px 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        border: 2px solid #2196F3;
                    }
                """)
        
        stroke_color_btn.clicked.connect(choose_stroke_color)
        stroke_width_spin.valueChanged.connect(on_stroke_width_changed)
        
        stroke_h_layout.addWidget(stroke_color_btn)
        stroke_h_layout.addWidget(stroke_width_label)
        stroke_h_layout.addWidget(stroke_width_spin)
        stroke_h_layout.addStretch()
        
        stroke_layout.addLayout(stroke_h_layout)
        layout.addLayout(stroke_layout)
        
        # 이미지명 표시
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.addWidget(QtWidgets.QLabel("이미지명:"))
        image_label = QtWidgets.QLabel(region.image_filename if region.image_filename else "미설정")
        image_label.setStyleSheet("color: blue; font-weight: bold;")
        image_layout.addWidget(image_label)
        layout.addLayout(image_layout)
        
        # 레이어 순서 설정
        layer_layout = QtWidgets.QHBoxLayout()
        layer_layout.addWidget(QtWidgets.QLabel("레이어 순서:"))
        
        def move_to_front():
            """텍스트 박스를 제일 앞으로 이동 (리스트의 맨 뒤로)"""
            # region 객체를 직접 찾아서 이동 (인덱스 변경에 안전)
            try:
                current_index = self.owner.text_regions.index(region)
                # 현재 텍스트 박스를 리스트에서 제거
                region_to_move = self.owner.text_regions.pop(current_index)
                # 리스트의 맨 뒤에 추가 (가장 위에 표시됨)
                self.owner.text_regions.append(region_to_move)
                # UI 업데이트
                if hasattr(self.owner, 'text_table'):
                    self.owner.update_text_table()
                if hasattr(self.owner, 'update_display_for_current_image'):
                    self.owner.update_display_for_current_image()
                # 새로운 인덱스로 테이블 선택 업데이트
                new_index = len(self.owner.text_regions) - 1
                if hasattr(self.owner, 'text_table'):
                    self.owner.text_table.selectRow(new_index)
                self.owner.update_status(f"텍스트 박스를 제일 앞으로 이동 (레이어 {new_index + 1})", "green")
            except (ValueError, IndexError):
                self.owner.update_status("레이어 순서 변경 실패", "red")
        
        def move_to_back():
            """텍스트 박스를 제일 뒤로 이동 (리스트의 맨 앞으로)"""
            # region 객체를 직접 찾아서 이동 (인덱스 변경에 안전)
            try:
                current_index = self.owner.text_regions.index(region)
                # 현재 텍스트 박스를 리스트에서 제거
                region_to_move = self.owner.text_regions.pop(current_index)
                # 리스트의 맨 앞에 추가 (가장 아래에 표시됨)
                self.owner.text_regions.insert(0, region_to_move)
                # UI 업데이트
                if hasattr(self.owner, 'text_table'):
                    self.owner.update_text_table()
                if hasattr(self.owner, 'update_display_for_current_image'):
                    self.owner.update_display_for_current_image()
                # 새로운 인덱스로 테이블 선택 업데이트
                if hasattr(self.owner, 'text_table'):
                    self.owner.text_table.selectRow(0)
                self.owner.update_status(f"텍스트 박스를 제일 뒤로 이동 (레이어 1)", "green")
            except (ValueError, IndexError):
                self.owner.update_status("레이어 순서 변경 실패", "red")
        
        front_btn = QtWidgets.QPushButton("⬆️ 제일 앞으로")
        front_btn.clicked.connect(move_to_front)
        front_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        back_btn = QtWidgets.QPushButton("⬇️ 제일 뒤로")
        back_btn.clicked.connect(move_to_back)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        
        layer_layout.addWidget(front_btn)
        layer_layout.addWidget(back_btn)
        layer_layout.addStretch()
        layout.addLayout(layer_layout)
        
        # 버튼
        button_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("확인")
        cancel_button = QtWidgets.QPushButton("취소")
        # OK 버튼이 기본 엔터키를 받지 않도록 설정 (텍스트 에디터에서 엔터키 사용 가능하도록)
        ok_button.setAutoDefault(False)
        ok_button.setDefault(False)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # 텍스트 에디터에 포커스 설정 (다이얼로그가 열릴 때)
        text_edit.setFocus()
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            region.text = text_edit.toPlainText()
            region.font_size = font_size_spin.value()
            # 폰트 이름에서 "⭐ " 접두사 제거
            selected_font = font_combo.currentText()
            if selected_font.startswith("⭐ "):
                selected_font = selected_font[2:]  # "⭐ " 제거
            region.font_family = selected_font
            region.margin = margin_spin.value()
            region.wrap_mode = "char" if wrap_combo.currentText() == "글자 단위" else "word"
            region.line_spacing = float(line_spacing_combo.currentText())
            # 폰트 굵기 설정 (0=보통, 1=진하게, 2=더 진하게)
            bold_text = bold_combo.currentText()
            if bold_text == "보통":
                region.bold_level = 0
            elif bold_text == "진하게":
                region.bold_level = 1
            else:  # "더 진하게"
                region.bold_level = 2
            # 기존 bool 속성도 유지 (하위호환용): 0이면 False, 나머지는 True
            region.bold = region.bold_level >= 1
            # 텍스트 정렬 설정
            align_text = align_combo.currentText()
            if align_text == "왼쪽 정렬":
                region.text_align = "left"
            elif align_text == "오른쪽 정렬":
                region.text_align = "right"
            else:  # "가운데 정렬"
                region.text_align = "center"
            
            # 테두리 설정 저장 (UI에서 이미 설정되었지만 명시적으로 저장)
            # stroke_width_spin과 stroke_color_btn에서 이미 region을 직접 수정하고 있음
            # 하지만 명시적으로 확인
            if not hasattr(region, 'stroke_width') or region.stroke_width != stroke_width_spin.value():
                region.stroke_width = stroke_width_spin.value()
            if region.stroke_width == 0:
                region.stroke_color = None
            elif region.stroke_color is None and region.stroke_width > 0:
                # 두께가 설정되었는데 색상이 없으면 검은색으로 기본 설정
                region.stroke_color = (0, 0, 0)
            
            # UI 업데이트
            if hasattr(self.owner, 'text_table'):
                self.owner.update_text_table()
            if hasattr(self.owner, 'text_regions'):
                # 현재 이미지의 텍스트 박스만 표시
                if hasattr(self.owner, 'update_display_for_current_image'):
                    self.owner.update_display_for_current_image()
    
    def choose_color_for_region(self, button, region):
        """텍스트 영역의 색상 선택"""
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            region.color = (color.blue(), color.green(), color.red())  # BGR 순서
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    color: {'white' if color.lightness() < 128 else 'black'};
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    padding: 5px;
                }}
            """)
    
    def get_text_at_position(self, pos):
        """특정 위치에 있는 텍스트 박스 인덱스 반환 (제일 위 레이어 우선, 현재 이미지의 텍스트 박스만)"""
        if not self.owner or not hasattr(self.owner, 'text_regions'):
            return -1
        
        # 현재 이미지의 텍스트 박스만 검사 (성능 최적화)
        if not (hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path):
            return -1
            
        current_filename = os.path.basename(self.owner.jp_image_path)
        x, y = pos
        
        # 역순으로 검사하여 제일 위에 있는 레이어 선택 (나중에 추가된 것이 위에 있음)
        for i in range(len(self.owner.text_regions) - 1, -1, -1):
            region = self.owner.text_regions[i]
            if (hasattr(region, 'image_filename') and 
                region.image_filename == current_filename and
                region.is_positioned and region.target_bbox):
                x1, y1, x2, y2 = region.target_bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return i
        return -1
    
    def get_resize_handle(self, pos, text_index):
        """리사이즈 핸들 위치 확인 (현재 이미지의 텍스트 박스만)"""
        try:
            if not self.owner or not hasattr(self.owner, 'text_regions') or text_index < 0 or text_index >= len(self.owner.text_regions):
                return None
            
            region = self.owner.text_regions[text_index]
            if not region.is_positioned or not region.target_bbox:
                return None
            
            # 현재 이미지의 텍스트 박스인지 확인
            if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                current_filename = os.path.basename(self.owner.jp_image_path)
                if region.image_filename != current_filename:
                    return None
            else:
                return None
            
            x, y = pos
            x1, y1, x2, y2 = region.target_bbox
            
            # 핸들 크기 (감지 영역을 넓게 설정하여 핸들 근처 클릭도 감지)
            handle_size = 15  # 실제 핸들 크기
            handle_margin = 5  # 핸들 근처 여유 공간 (핸들 근처 클릭도 감지)
            effective_size = handle_size + handle_margin  # 실제 감지 영역
            
            # 각 핸들 영역을 명확하게 정의 (여유 공간 포함)
            # 우하단 핸들 (southeast)
            se_x1, se_y1 = x2 - effective_size, y2 - effective_size
            se_x2, se_y2 = x2, y2
            if se_x1 <= x <= se_x2 and se_y1 <= y <= se_y2:
                return "se"
            
            # 우상단 핸들 (northeast)  
            ne_x1, ne_y1 = x2 - effective_size, y1
            ne_x2, ne_y2 = x2, y1 + effective_size
            if ne_x1 <= x <= ne_x2 and ne_y1 <= y <= ne_y2:
                return "ne"
        
            # 좌하단 핸들 (southwest)
            sw_x1, sw_y1 = x1, y2 - effective_size
            sw_x2, sw_y2 = x1 + effective_size, y2
            if sw_x1 <= x <= sw_x2 and sw_y1 <= y <= sw_y2:
                return "sw"
            
            # 좌상단 핸들 (northwest)
            nw_x1, nw_y1 = x1, y1
            nw_x2, nw_y2 = x1 + effective_size, y1 + effective_size
            if nw_x1 <= x <= nw_x2 and nw_y1 <= y <= nw_y2:
                return "nw"
            
            return None
            
        except Exception as e:
            return None
    
    def move_text_box(self, new_pos):
        """텍스트 박스 이동 (현재 이미지의 텍스트 박스만) - 최적화된 버전"""
        # 빠른 검증 (최적화)
        if (not self.owner or not hasattr(self.owner, 'text_regions') or 
            self.selected_text_index < 0 or self.selected_text_index >= len(self.owner.text_regions)):
            return
        
        region = self.owner.text_regions[self.selected_text_index]
        if not region.is_positioned or not region.target_bbox:
            return
        
        # 현재 이미지의 텍스트 박스인지 확인 (캐싱된 값 사용)
        if not hasattr(self, '_current_filename'):
            if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                self._current_filename = os.path.basename(self.owner.jp_image_path)
            else:
                return
        if region.image_filename != self._current_filename:
            return
        
        # 처음 이동 시작할 때의 위치를 기억
        if not hasattr(self, 'drag_start_pos'):
            self.drag_start_pos = new_pos
            self.drag_start_bbox = region.target_bbox
            return
        
        # 드래그 거리 계산 (정수 변환으로 성능 향상)
        dx = int(new_pos[0] - self.drag_start_pos[0])
        dy = int(new_pos[1] - self.drag_start_pos[1])
        
        # 이동 거리가 없으면 업데이트 불필요
        if dx == 0 and dy == 0:
            return
        
        # 원래 위치에서 드래그 거리만큼 이동
        x1, y1, x2, y2 = self.drag_start_bbox
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy
        
        # 이미지 범위 내로 제한 (이미지 크기 캐싱)
        if self.image is not None:
            if not hasattr(self, '_img_size'):
                self._img_size = self.image.shape[:2]  # (height, width)
            img_h, img_w = self._img_size
            width = x2 - x1
            height = y2 - y1
            new_x1 = max(0, min(new_x1, img_w - width))
            new_y1 = max(0, min(new_y1, img_h - height))
            new_x2 = new_x1 + width
            new_y2 = new_y1 + height
        
        region.target_bbox = (new_x1, new_y1, new_x2, new_y2)
        
        # --- 안전 클램핑 추가 ---
        if self.image is not None:
            img_h, img_w = self.image.shape[:2]
            x1, y1, x2, y2 = region.target_bbox
            x1 = max(0, min(int(x1), img_w - 2))
            y1 = max(0, min(int(y1), img_h - 2))
            x2 = max(x1 + 1, min(int(x2), img_w - 1))
            y2 = max(y1 + 1, min(int(y2), img_h - 1))
            region.target_bbox = (x1, y1, x2, y2)
        
        # 빠른 업데이트: 테이블 업데이트 없이 캔버스만 업데이트
        if self.owner and hasattr(self.owner, 'text_regions'):
            # 현재 이미지의 텍스트 박스만 필터링하여 직접 업데이트 (성능 최적화)
            if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                current_filename = os.path.basename(self.owner.jp_image_path)
                current_text_regions = []
                for region in self.owner.text_regions:
                    if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                        current_text_regions.append(region)
                # 캔버스만 직접 업데이트 (테이블 업데이트 제외로 성능 향상)
                if hasattr(self, 'update_display_with_preview'):
                    self.update_display_with_preview(current_text_regions)
    
    def resize_text_box(self, new_pos):
        """텍스트 박스 크기 조절 (현재 이미지의 텍스트 박스만) - 최적화된 버전"""
        try:
            # 빠른 검증 (최적화)
            if (not self.owner or not hasattr(self.owner, 'text_regions') or
                self.selected_text_index < 0 or self.selected_text_index >= len(self.owner.text_regions)):
                return
            
            region = self.owner.text_regions[self.selected_text_index]
            if not region or not hasattr(region, 'is_positioned') or not region.is_positioned:
                return
            
            if not hasattr(region, 'target_bbox') or not region.target_bbox:
                return
            
            # 현재 이미지의 텍스트 박스인지 확인 (캐싱된 값 사용)
            if not hasattr(self, '_current_filename'):
                if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                    self._current_filename = os.path.basename(self.owner.jp_image_path)
                else:
                    return
            if region.image_filename != self._current_filename:
                return
            
            # 처음 리사이즈 시작할 때의 위치를 기억
            if not hasattr(self, 'resize_start_pos'):
                self.resize_start_pos = new_pos
                self.resize_start_bbox = region.target_bbox
                return
            
            # 안전한 좌표 검증
            if not isinstance(new_pos, (tuple, list)) or len(new_pos) != 2:
                return
            
            # 드래그 거리 계산 (정수 변환으로 성능 향상)
            try:
                dx = int(new_pos[0]) - int(self.resize_start_pos[0])
                dy = int(new_pos[1]) - int(self.resize_start_pos[1])
            except (ValueError, TypeError) as e:
                return
            
            # 이동 거리가 없으면 업데이트 불필요
            if dx == 0 and dy == 0:
                return
            
            # 원래 위치에서 드래그 거리만큼 조정
            x1, y1, x2, y2 = self.resize_start_bbox
            
            # 이미지 크기 가져오기 (캐싱으로 성능 향상)
            if hasattr(self, 'image') and self.image is not None:
                if not hasattr(self, '_img_size'):
                    self._img_size = self.image.shape[:2]  # (height, width)
                img_height, img_width = self._img_size
            else:
                # 이미지 크기를 알 수 없는 경우 기본값 사용
                img_width, img_height = 1920, 1080
            
            # 최소 크기 제한
            min_size = 30
            
            try:
                if self.resize_handle == "se":  # 우하단
                    new_x2 = max(x1 + min_size, x2 + dx)
                    new_y2 = max(y1 + min_size, y2 + dy)
                    # bbox 경계 클램핑
                    new_x2 = max(x1 + min_size, min(new_x2, img_width))
                    new_y2 = max(y1 + min_size, min(new_y2, img_height))
                    region.target_bbox = (x1, y1, new_x2, new_y2)
                    
                elif self.resize_handle == "ne":  # 우상단
                    new_x2 = max(x1 + min_size, x2 + dx)
                    new_y1 = min(y2 - min_size, y1 + dy)
                    # bbox 경계 클램핑
                    new_x2 = max(x1 + min_size, min(new_x2, img_width))
                    new_y1 = max(0, min(new_y1, y2 - min_size))
                    region.target_bbox = (x1, new_y1, new_x2, y2)
                    
                elif self.resize_handle == "sw":  # 좌하단
                    new_x1 = min(x2 - min_size, x1 + dx)
                    new_y2 = max(y1 + min_size, y2 + dy)
                    # bbox 경계 클램핑
                    new_x1 = max(0, min(new_x1, x2 - min_size))
                    new_y2 = max(y1 + min_size, min(new_y2, img_height))
                    region.target_bbox = (new_x1, y1, x2, new_y2)
                    
                elif self.resize_handle == "nw":  # 좌상단
                    new_x1 = min(x2 - min_size, x1 + dx)
                    new_y1 = min(y2 - min_size, y1 + dy)
                    # bbox 경계 클램핑
                    new_x1 = max(0, min(new_x1, x2 - min_size))
                    new_y1 = max(0, min(new_y1, y2 - min_size))
                    region.target_bbox = (new_x1, new_y1, x2, y2)
                
                # bbox 계산 직후 안전 클램핑 (추가 보안)
                img_h, img_w = self.image.shape[:2]
                x1, y1, x2, y2 = region.target_bbox
                
                # 안전 클램핑
                x1 = max(0, min(x1, img_w - 2))
                x2 = max(x1 + 1, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 2))
                y2 = max(y1 + 1, min(y2, img_h - 1))
                region.target_bbox = (x1, y1, x2, y2)
                
                # 빠른 업데이트: 테이블 업데이트 없이 캔버스만 업데이트
                try:
                    if self.owner and hasattr(self.owner, 'text_regions'):
                        # 현재 이미지의 텍스트 박스만 필터링하여 직접 업데이트 (성능 최적화)
                        if hasattr(self.owner, 'jp_image_path') and self.owner.jp_image_path:
                            current_filename = os.path.basename(self.owner.jp_image_path)
                            current_text_regions = []
                            for region in self.owner.text_regions:
                                if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                                    current_text_regions.append(region)
                            # 캔버스만 직접 업데이트 (테이블 업데이트 제외로 성능 향상)
                            if hasattr(self, 'update_display_with_preview'):
                                self.update_display_with_preview(current_text_regions)
                except Exception as e:
                    pass
                    
            except Exception as e:
                pass
                
        except Exception as e:
            # 오류 발생 시 리사이즈 모드 종료
            self.resizing = False
            self.resize_handle = None

    def load_font_for_overlay(self, font_family, font_size):
        """오버레이용 폰트 로드"""
        # 사용자 추가 폰트 확인 (우선순위)
        if self.owner and hasattr(self.owner, 'custom_fonts') and font_family in self.owner.custom_fonts:
            custom_font_path = self.owner.custom_fonts[font_family]
            if os.path.exists(custom_font_path):
                try:
                    font = ImageFont.truetype(custom_font_path, font_size)
                    return font
                except Exception as e:
                    logger.error(f"사용자 추가 폰트 로딩 실패: {custom_font_path}, 오류: {e}")
                    # 실패 시 기본 폰트로 폴백
        
        # 사용자 설정 폰트가 시스템 폰트 목록에 있는지 확인
        system_fonts = ["Arial", "Times New Roman", "Courier New", "굴림", "맑은 고딕", "나눔고딕"]
        
        if font_family in system_fonts:
            font_paths = {
                "Arial": ["fonts/arial.ttf", "C:/Windows/Fonts/arial.ttf"],
                "Times New Roman": ["fonts/times.ttf", "C:/Windows/Fonts/times.ttf"],
                "Courier New": ["fonts/cour.ttf", "C:/Windows/Fonts/cour.ttf"],
                "굴림": [resource_path("fonts/gulim.ttc"), "C:/Windows/Fonts/gulim.ttc", "C:/Windows/Fonts/NGULIM.TTF"],
                "맑은 고딕": [resource_path("fonts/malgun.ttf"), "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgunsl.ttf"],
                "나눔고딕": [resource_path("fonts/NanumGothic.ttf"), "C:/Windows/Fonts/NanumGothic.ttf"]
            }
            
            if font_family in font_paths:
                for font_path in font_paths[font_family]:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(resource_path(font_path), font_size)
                            return font
                        except Exception as e:
                            logger.error(f"폰트 로딩 실패: {font_path}, 오류: {e}")
                            continue
        
        # 기본 한글 폰트들 시도
        default_font_paths = [
            resource_path("fonts/NanumGothic.ttf"),
            resource_path("fonts/malgun.ttf"),
            resource_path("fonts/gulim.ttc"),
            "C:/Windows/Fonts/NanumGothic.ttf",
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            "C:/Windows/Fonts/batang.ttc",
            "C:/Windows/Fonts/dotum.ttc",
        ]
        
        for font_path in default_font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(resource_path(font_path), font_size)
                    return font
                except Exception as e:
                    logger.error(f"기본 폰트 로딩 실패: {font_path}, 오류: {e}")
                    continue
        
        # 모든 시도가 실패하면 기본 폰트 사용
        return ImageFont.load_default()


class TextOverlayTool(QtWidgets.QMainWindow):
    """
    Main application window for text overlay tool (Cloud Vision OCR version)
    텍스트 오버레이 툴 메인 애플리케이션 윈도우 (클라우드 비전 OCR 버전)
    
    This is the main window class that handles the UI and coordinates
    between OCR processing, text management, and image overlay operations.
    이것은 UI를 처리하고 OCR 처리, 텍스트 관리, 이미지 오버레이 작업을 조정하는 메인 윈도우 클래스입니다.
    """
    
    # Signals for OCR completion (for thread communication)
    # OCR 완료 시그널 정의 (스레드 간 통신용)
    vision_ocr_completed = QtCore.pyqtSignal(list)  # Text lines list / 텍스트 라인 리스트
    vision_ocr_failed = QtCore.pyqtSignal(str)  # Error message / 에러 메시지
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("텍스트 오버레이 툴 (클라우드 비전 OCR) - OCR 소스 이미지 → 타겟 이미지")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 700)
        
        # 설정 파일 경로
        self.config_path = resource_path("text_overlay_tool_gemini.ini")
        
        # 전체 UI 폰트를 나눔고딕으로 설정 (로컬 폰트 등록)
        try:
            # 로컬 fonts 폴더의 나눔고딕 폰트 등록
            font_id = QFontDatabase.addApplicationFont(resource_path("fonts/NanumGothic.ttf"))
            if font_id != -1:
                # 폰트 등록 성공 시 등록된 폰트 이름 사용
                font_name = QFontDatabase.applicationFontFamilies(font_id)[0]
                font = QtGui.QFont(font_name, 9)
            else:
                # 폰트 등록 실패 시 맑은 고딕 사용
                font = QtGui.QFont("맑은 고딕", 9)
                logger.warning("나눔고딕 폰트 등록 실패, 맑은 고딕 사용")
        except Exception as e:
            # 모든 시도가 실패하면 기본 폰트 사용
            font = QtGui.QFont("맑은 고딕", 9)
            logger.error(f"폰트 설정 오류: {e}")
        
        self.setFont(font)
        
        # 변수 초기화 (기본값)
        self.kr_image_path = None
        self.jp_image_path = None
        self.kr_image = None
        self.jp_image = None
        self.text_regions = []
        self.ocr_engine = CloudVisionOCR()
        self.custom_fonts = {}  # 사용자 추가 폰트: {폰트명: 파일경로}
        self.default_font_size = 18  # 기본 폰트 크기
        self.default_font_family = "나눔고딕"  # 기본 폰트
        self.default_color_bgr = (0, 0, 0)  # 기본 색상 (검은색, BGR)
        
        # 한국어/타겟 이미지 폴더 관련 변수들
        self.kr_image_list = []
        self.kr_current_image_index = 0
        self.jp_image_list = []
        self.jp_current_image_index = 0
        self.kr_last_folder = ""
        self.jp_last_folder = ""
        self.result_last_folder = ""
        self.csv_last_folder = ""
        
        # ini 설정 로드 (기본 폰트/색상/폴더 복원)
        self.load_settings()
        
        self.init_ui()
        self.setup_shortcuts()
        
        # 기본 색상 버튼에 적용
        self.apply_default_color_to_button()
        
        # OCR 완료 시그널 연결
        self.vision_ocr_completed.connect(self.on_vision_ocr_completed)
        self.vision_ocr_failed.connect(self.on_vision_ocr_failed)
        
        # 클라우드 비전 OCR 안내 메시지
        if not CLOUD_VISION_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "패키지 미설치",
                "google-cloud-vision 패키지가 설치되지 않았습니다.\n"
                "pip install google-cloud-vision로 설치하세요.\n\n"
                "또한 Google Cloud Console에서:\n"
                "1. Cloud Vision API 활성화\n"
                "2. 서비스 계정 생성 및 키 파일 다운로드\n"
                "가 필요합니다.\n\n"
                "설치 후 서비스 계정 키 파일을 설정하면 클라우드 비전 OCR을 사용할 수 있습니다."
            )
    
    def load_settings(self):
        """INI 설정 파일에서 기본 폰트/색상/폴더 정보를 로드"""
        try:
            config = configparser.ConfigParser()
            if os.path.exists(self.config_path):
                config.read(self.config_path, encoding="utf-8")
                section = config["general"] if "general" in config else None
                if section:
                    # 기본 폰트 크기
                    if section.get("default_font_size"):
                        try:
                            self.default_font_size = int(section.get("default_font_size"))
                        except ValueError:
                            pass
                    # 기본 폰트
                    if section.get("default_font_family"):
                        self.default_font_family = section.get("default_font_family")
                    # 기본 색상 (BGR)
                    if section.get("color_b") and section.get("color_g") and section.get("color_r"):
                        try:
                            b = int(section.get("color_b"))
                            g = int(section.get("color_g"))
                            r = int(section.get("color_r"))
                            self.default_color_bgr = (b, g, r)
                        except ValueError:
                            pass
                    # 마지막 폴더
                    if section.get("kr_last_folder"):
                        self.kr_last_folder = section.get("kr_last_folder")
                    if section.get("jp_last_folder"):
                        self.jp_last_folder = section.get("jp_last_folder")
                    if section.get("result_last_folder"):
                        self.result_last_folder = section.get("result_last_folder")
                    if section.get("csv_last_folder"):
                        self.csv_last_folder = section.get("csv_last_folder")
        except Exception as e:
            logger.error(f"INI 설정 로드 오류: {e}")
    
    def save_settings(self):
        """현재 기본 설정을 INI 파일에 저장"""
        try:
            config = configparser.ConfigParser()
            config["general"] = {}
            general = config["general"]
            
            # 기본 폰트/크기
            general["default_font_size"] = str(getattr(self, "default_font_size", 18))
            general["default_font_family"] = getattr(self, "default_font_family", "나눔고딕")
            
            # 현재 색상 (BGR)
            try:
                # color_btn이 있으면 UI에서 직접 읽기
                if hasattr(self, "color_btn"):
                    b, g, r = self.get_current_color()
                    self.default_color_bgr = (b, g, r)
            except Exception:
                pass
            b, g, r = getattr(self, "default_color_bgr", (0, 0, 0))
            general["color_b"] = str(b)
            general["color_g"] = str(g)
            general["color_r"] = str(r)
            
            # 마지막 폴더
            general["kr_last_folder"] = getattr(self, "kr_last_folder", "") or ""
            general["jp_last_folder"] = getattr(self, "jp_last_folder", "") or ""
            general["result_last_folder"] = getattr(self, "result_last_folder", "") or ""
            general["csv_last_folder"] = getattr(self, "csv_last_folder", "") or ""
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                config.write(f)
        except Exception as e:
            logger.error(f"INI 설정 저장 오류: {e}")
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단 툴바
        self.create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # 메인 콘텐츠 영역
        content_layout = QtWidgets.QHBoxLayout()
        
        # 좌측: 소스 이미지 (OCR 소스)
        left_panel = self.create_image_panel("OCR 소스 이미지", "kr")
        content_layout.addWidget(left_panel, 1)
        
        # 중앙: 텍스트 편집 영역
        center_panel = self.create_text_panel()
        content_layout.addWidget(center_panel, 1)
        
        # 우측: 타겟 이미지 (타겟)
        right_panel = self.create_image_panel("타겟 이미지", "jp")
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout, 1)
        
        # 하단 상태바
        self.create_statusbar()
        main_layout.addWidget(self.statusbar)
    
    def create_toolbar(self):
        """툴바 생성"""
        self.toolbar = QtWidgets.QWidget()
        self.toolbar.setFixedHeight(60)
        self.toolbar.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        layout = QtWidgets.QHBoxLayout(self.toolbar)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 파일 관리 버튼들
        file_group = QtWidgets.QGroupBox("파일 관리")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        file_layout.setSpacing(5)
        file_layout.setContentsMargins(10, 15, 10, 10)  # 상단 여백 추가
        
        kr_btn = QtWidgets.QPushButton("📁 소스 이미지 폴더")
        kr_btn.clicked.connect(self.select_korean_image_folder)
        file_layout.addWidget(kr_btn)
        
        jp_btn = QtWidgets.QPushButton("📁 타겟 이미지 폴더")
        jp_btn.clicked.connect(self.select_japanese_image_folder)
        file_layout.addWidget(jp_btn)
        
        save_btn = QtWidgets.QPushButton("💾 결과 저장")
        save_btn.clicked.connect(self.save_result)
        file_layout.addWidget(save_btn)
        
        # CSV 저장/불러오기 버튼 추가
        csv_save_btn = QtWidgets.QPushButton("📊 CSV 저장")
        csv_save_btn.clicked.connect(self.save_csv)
        file_layout.addWidget(csv_save_btn)
        
        csv_load_btn = QtWidgets.QPushButton("📂 CSV 불러오기")
        csv_load_btn.clicked.connect(self.load_csv)
        file_layout.addWidget(csv_load_btn)
        
        # 폰트 파일 추가 버튼
        font_add_btn = QtWidgets.QPushButton("🔤 폰트 파일 추가")
        font_add_btn.clicked.connect(self.add_font_file)
        font_add_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        file_layout.addWidget(font_add_btn)
        
        layout.addWidget(file_group)
        
        
        # OCR 설정 (클라우드 비전 OCR만)
        ocr_group = QtWidgets.QGroupBox("OCR 설정 (구글 클라우드 비전)")
        ocr_layout = QtWidgets.QHBoxLayout(ocr_group)
        ocr_layout.setSpacing(5)
        ocr_layout.setContentsMargins(10, 15, 10, 5)  # 상단 여백 추가
        
        vision_api_btn = QtWidgets.QPushButton("🔑 인증 파일 설정")
        vision_api_btn.clicked.connect(self.set_vision_credentials_dialog)
        vision_api_btn.setStyleSheet("""
            QPushButton {
                background-color: #4285F4;
                color: white;
                border: none;
                padding: 1px 5px;
                border-radius: 4px;
                font-weight: bold;
                width: 120px;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)
        ocr_layout.addWidget(vision_api_btn)
        
        self.vision_ocr_btn = QtWidgets.QPushButton("👁️ OCR 실행")
        self.vision_ocr_btn.clicked.connect(self.run_vision_ocr)
        self.vision_ocr_btn.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                color: white;
                border: none;
                padding: 1px 5px;
                border-radius: 4px;
                font-weight: bold;
                width: 150px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        self.vision_ocr_btn.setEnabled(False)  # 인증 파일 설정 전까지 비활성화
        ocr_layout.addWidget(self.vision_ocr_btn)
        ocr_layout.addStretch()
        
        layout.addWidget(ocr_group)
        
        # 텍스트 설정
        text_group = QtWidgets.QGroupBox("텍스트 설정")
        text_layout = QtWidgets.QHBoxLayout(text_group)
        text_layout.setSpacing(5)
        text_layout.setContentsMargins(10, 15, 10, 10)  # 상단 여백 추가
        
        text_layout.addWidget(QtWidgets.QLabel("폰트 크기:"))
        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(self.default_font_size)
        self.font_size_spin.valueChanged.connect(self.on_font_size_changed)
        text_layout.addWidget(self.font_size_spin)
        
        # 폰트 크기 슬라이더 추가
        self.font_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.font_size_slider.setRange(8, 72)
        self.font_size_slider.setValue(self.default_font_size)
        self.font_size_slider.valueChanged.connect(self.on_font_size_slider_changed)
        text_layout.addWidget(self.font_size_slider)
        
        # 기본 폰트 크기 변경 버튼
        default_font_size_btn = QtWidgets.QPushButton(f"📏 기본: {self.default_font_size}")
        default_font_size_btn.clicked.connect(self.change_default_font_size)
        default_font_size_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.default_font_size_btn = default_font_size_btn  # 참조 저장
        text_layout.addWidget(default_font_size_btn)
        
        # 기본 폰트 변경 버튼
        default_font_btn = QtWidgets.QPushButton(f"🔤 기본 폰트: {self.default_font_family}")
        default_font_btn.clicked.connect(self.change_default_font)
        default_font_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.default_font_btn = default_font_btn  # 참조 저장
        text_layout.addWidget(default_font_btn)
        
        # 간단한 안내 라벨
        help_label = QtWidgets.QLabel("💡 텍스트 박스를 더블클릭하여 편집")
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        text_layout.addWidget(help_label)
        
        text_layout.addWidget(QtWidgets.QLabel("색상:"))
        self.color_btn = QtWidgets.QPushButton("⚫")
        self.color_btn.clicked.connect(self.choose_color)
        self.color_btn.setStyleSheet("""
            QPushButton {
                background-color: black;
                color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                width: 30px;
                height: 25px;
            }
        """)
        text_layout.addWidget(self.color_btn)
        
        layout.addWidget(text_group)
        
        layout.addStretch()
    
    def create_image_panel(self, title, canvas_id):
        """이미지 패널 생성"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 제목
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        layout.addWidget(title_label)
        
        # 이미지 리스트 (한국어/타겟 이미지별로 추가)
        if canvas_id == "kr":
            # 소스 이미지 리스트 위젯
            list_label = QtWidgets.QLabel("📋 소스 이미지 목록")
            list_label.setStyleSheet("font-weight: bold; color: #333; padding: 3px;")
            layout.addWidget(list_label)
            
            self.kr_image_list_widget = QtWidgets.QListWidget()
            self.kr_image_list_widget.setMaximumHeight(120)
            self.kr_image_list_widget.setStyleSheet("""
                QListWidget {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background-color: #fafafa;
                }
                QListWidget::item {
                    padding: 3px;
                    border-bottom: 1px solid #eee;
                }
                QListWidget::item:selected {
                    background-color: #2196F3;
                    color: white;
                }
            """)
            self.kr_image_list_widget.itemClicked.connect(self.on_kr_image_list_click)
            layout.addWidget(self.kr_image_list_widget)
            
            # 현재 이미지 정보
            self.kr_current_image_label = QtWidgets.QLabel("이미지를 선택하세요")
            self.kr_current_image_label.setStyleSheet("""
                font-weight: bold; 
                color: #2196F3; 
                padding: 3px;
                background-color: #e3f2fd;
                border-radius: 3px;
            """)
            layout.addWidget(self.kr_current_image_label)
            
            # 소스 이미지 네비게이션 버튼
            nav_layout = QtWidgets.QHBoxLayout()
            prev_btn = QtWidgets.QPushButton("⬅️ 이전")
            prev_btn.clicked.connect(self.prev_kr_image)
            nav_layout.addWidget(prev_btn)
            
            next_btn = QtWidgets.QPushButton("다음 ➡️")
            next_btn.clicked.connect(self.next_kr_image)
            nav_layout.addWidget(next_btn)
            layout.addLayout(nav_layout)
            
        elif canvas_id == "jp":
            # 타겟 이미지 리스트 위젯
            list_label = QtWidgets.QLabel("📋 타겟 이미지 목록")
            list_label.setStyleSheet("font-weight: bold; color: #333; padding: 3px;")
            layout.addWidget(list_label)
            
            self.jp_image_list_widget = QtWidgets.QListWidget()
            self.jp_image_list_widget.setMaximumHeight(120)
            self.jp_image_list_widget.setStyleSheet("""
                QListWidget {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background-color: #fafafa;
                }
                QListWidget::item {
                    padding: 3px;
                    border-bottom: 1px solid #eee;
                }
                QListWidget::item:selected {
                    background-color: #2196F3;
                    color: white;
                }
            """)
            self.jp_image_list_widget.itemClicked.connect(self.on_jp_image_list_click)
            layout.addWidget(self.jp_image_list_widget)
            
            # 현재 이미지 정보
            self.jp_current_image_label = QtWidgets.QLabel("이미지를 선택하세요")
            self.jp_current_image_label.setStyleSheet("""
                font-weight: bold; 
                color: #2196F3; 
                padding: 3px;
                background-color: #e3f2fd;
                border-radius: 3px;
            """)
            layout.addWidget(self.jp_current_image_label)
            
            # 타겟 이미지 네비게이션 버튼
            nav_layout = QtWidgets.QHBoxLayout()
            prev_btn = QtWidgets.QPushButton("⬅️ 이전")
            prev_btn.clicked.connect(self.prev_jp_image)
            nav_layout.addWidget(prev_btn)
            
            next_btn = QtWidgets.QPushButton("다음 ➡️")
            next_btn.clicked.connect(self.next_jp_image)
            nav_layout.addWidget(next_btn)
            layout.addLayout(nav_layout)
        # 스크롤 가능한 캔버스 (개선된 버전)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(False)  # False로 설정하여 스크롤바 활성화
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumSize(400, 300)  # 최소 크기 설정
        
        canvas = ImageCanvas(canvas_id, owner=self)
        canvas.region_selected.connect(self.on_region_selected)
        canvas.text_dropped.connect(self.on_text_dropped)
        
        scroll_area.setWidget(canvas)
        layout.addWidget(scroll_area, 1)
        
        # 확대/축소 정보
        zoom_label = QtWidgets.QLabel("🔍 확대율: 1.0x")
        zoom_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(zoom_label)
        
        # 캔버스 참조 저장
        if canvas_id == "kr":
            self.kr_canvas = canvas
            self.kr_zoom_label = zoom_label
        else:
            self.jp_canvas = canvas
            self.jp_zoom_label = zoom_label
        
        return panel
    
    def create_text_panel(self):
        """텍스트 편집 패널 생성"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 제목
        title_label = QtWidgets.QLabel("📝 텍스트 편집")
        title_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        layout.addWidget(title_label)
        
        # 텍스트 테이블
        self.text_table = QtWidgets.QTableWidget()
        self.text_table.setColumnCount(5)
        self.text_table.setHorizontalHeaderLabels(["번호", "텍스트", "위치", "상태", "이미지명"])
        self.text_table.horizontalHeader().setStretchLastSection(True)
        self.text_table.setAlternatingRowColors(True)
        self.text_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.text_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        
        # 드래그 앤 드롭 설정
        self.text_table.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.text_table.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.text_table.setDragDropOverwriteMode(False)
        
        # 드래그 시작 이벤트 연결
        self.text_table.startDrag = self.start_text_drag
        
        # 더블클릭 이벤트 연결
        self.text_table.itemDoubleClicked.connect(self.on_table_item_double_clicked)
        
        # 텍스트 변경 이벤트 연결 (인라인 편집)
        self.text_table.itemChanged.connect(self.on_table_item_changed)
        
        # 행 선택 이벤트 연결
        self.text_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        
        # 컨텍스트 메뉴 설정
        self.text_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.text_table.customContextMenuRequested.connect(self.show_text_table_context_menu)
        
        # 테이블 스타일
        self.text_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                background-color: white;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #333;
                padding: 5px;
                border: none;
                font-weight: bold;
                border-bottom: 1px solid #d0d0d0;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e8e8e8;
                color: #333;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #333;
            }
            QLineEdit {
                color: #333;
                background-color: white;
                border: 2px solid #2196F3;
                border-radius: 3px;
                padding: 2px;
            }
        """)
        
        layout.addWidget(self.text_table, 1)
        
        # 버튼들
        button_layout = QtWidgets.QHBoxLayout()
        
        # 수동 라인 추가 버튼
        add_line_btn = QtWidgets.QPushButton("➕ 라인 추가")
        add_line_btn.clicked.connect(self.add_manual_text_line)
        add_line_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(add_line_btn)
        
        clear_btn = QtWidgets.QPushButton("🗑️ 전체 삭제")
        clear_btn.clicked.connect(self.clear_all_texts)
        button_layout.addWidget(clear_btn)
        
        delete_btn = QtWidgets.QPushButton("❌ 선택 삭제")
        delete_btn.clicked.connect(self.delete_selected_text)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        button_layout.addWidget(delete_btn)
        
        # 위치 초기화 버튼
        reset_position_btn = QtWidgets.QPushButton("🔄 위치 초기화")
        reset_position_btn.clicked.connect(self.reset_text_position)
        reset_position_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        button_layout.addWidget(reset_position_btn)
        
        # 라인 합치기 버튼
        merge_btn = QtWidgets.QPushButton("🔗 라인 합치기")
        merge_btn.clicked.connect(self.merge_selected_lines)
        merge_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(merge_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        return panel
    
    def create_statusbar(self):
        """상태바 생성"""
        self.statusbar = QtWidgets.QWidget()
        self.statusbar.setFixedHeight(30)
        self.statusbar.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-top: 1px solid #ddd;
                border-radius: 0px 0px 5px 5px;
            }
        """)
        
        layout = QtWidgets.QHBoxLayout(self.statusbar)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QtWidgets.QLabel("🚀 준비됨")
        self.status_label.setStyleSheet("color: #333; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.stats_label = QtWidgets.QLabel("📊 텍스트: 0개")
        self.stats_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.stats_label)
    
    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        # Ctrl + S: 저장
        save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_result)
        
        # Alt + S: 결과 저장
        alt_save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Alt+S"), self)
        alt_save_shortcut.activated.connect(self.save_result)
        
        # Ctrl + A: 라인 추가
        add_line_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self)
        add_line_shortcut.activated.connect(self.add_manual_text_line)
        
        # Ctrl + D: 선택된 텍스트 삭제
        delete_shortcut_ctrl_d = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self)
        delete_shortcut_ctrl_d.activated.connect(self.delete_selected_text)
        
        # Delete: 선택된 텍스트 삭제
        delete_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Delete"), self)
        delete_shortcut.activated.connect(self.delete_selected_text)
        
        # ESC: 텍스트 선택 해제
        escape_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self.clear_text_selection)
        
        # 화살표 키: 텍스트 박스 1px 이동
        up_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self)
        up_shortcut.activated.connect(lambda: self.move_selected_text_box(0, -1))
        
        down_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self)
        down_shortcut.activated.connect(lambda: self.move_selected_text_box(0, 1))
        
        left_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        left_shortcut.activated.connect(lambda: self.move_selected_text_box(-1, 0))
        
        right_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        right_shortcut.activated.connect(lambda: self.move_selected_text_box(1, 0))
        
    
    def move_selected_text_box(self, dx, dy):
        """선택된 텍스트 박스를 키보드로 1px씩 이동"""
        # jp_canvas에서 선택된 텍스트 박스 확인
        if not hasattr(self, 'jp_canvas') or not self.jp_canvas:
            return
        
        selected_index = getattr(self.jp_canvas, 'selected_text_index', -1)
        if selected_index < 0 or selected_index >= len(self.text_regions):
            return
        
        region = self.text_regions[selected_index]
        if not region.is_positioned or not region.target_bbox:
            return
        
        # 현재 이미지의 텍스트 박스인지 확인
        if hasattr(self, 'jp_image_path') and self.jp_image_path:
            current_filename = os.path.basename(self.jp_image_path)
            if region.image_filename != current_filename:
                return
        else:
            return
        
        # 현재 위치에서 1px 이동
        x1, y1, x2, y2 = region.target_bbox
        width = x2 - x1
        height = y2 - y1
        
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height
        
        # 이미지 범위 내로 제한
        if self.jp_image is not None:
            img_h, img_w = self.jp_image.shape[:2]
            new_x1 = max(0, min(new_x1, img_w - width))
            new_y1 = max(0, min(new_y1, img_h - height))
            new_x2 = new_x1 + width
            new_y2 = new_y1 + height
        
        # 위치 업데이트
        region.target_bbox = (new_x1, new_y1, new_x2, new_y2)
        
        # UI 업데이트
        if hasattr(self, 'update_display_for_current_image'):
            self.update_display_for_current_image()
        
        self.update_status(f"텍스트 박스 이동: ({new_x1}, {new_y1})", "blue")
    
    def clear_text_selection(self):
        """텍스트 선택 해제"""
        if hasattr(self, 'jp_canvas'):
            self.jp_canvas.selected_text_index = -1
            self.jp_canvas.resizing = False
            self.jp_canvas.moving = False
            self.jp_canvas.resize_handle = None
            # 드래그 시작 위치 초기화
            if hasattr(self.jp_canvas, 'drag_start_pos'):
                delattr(self.jp_canvas, 'drag_start_pos')
            if hasattr(self.jp_canvas, 'drag_start_bbox'):
                delattr(self.jp_canvas, 'drag_start_bbox')
            # 현재 이미지의 텍스트 박스만 표시
            self.update_display_for_current_image()
    
    def select_korean_image_folder(self):
        """소스 이미지 폴더 선택"""
        dialog = QtWidgets.QFileDialog(self, "소스 이미지 폴더 선택")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if self.kr_last_folder:
            dialog.setDirectory(self.kr_last_folder)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        folder_path = selected[0] if selected else ""
        if not folder_path:
            return
        
        # 지원하는 이미지 확장자
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # 폴더에서 이미지 파일들 찾기
        image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "오류", "선택한 폴더에 이미지 파일이 없습니다.")
            return
        
        # 파일명으로 정렬
        image_files.sort()
        
        self.kr_image_list = image_files
        self.kr_current_image_index = 0
        # 마지막 사용 폴더 저장
        self.kr_last_folder = folder_path
        self.save_settings()
        
        # 첫 번째 이미지 로드
        self.load_current_korean_image()
        
        # 이미지 목록 UI 업데이트
        self.update_kr_image_list_ui()
        
        self.update_status(f"소스 이미지 폴더 로드됨: {len(self.kr_image_list)}개 파일")
        QtWidgets.QMessageBox.information(self, "폴더 로드 완료", 
            f"{len(self.kr_image_list)}개의 소스 이미지 파일을 찾았습니다.\n" 
            f"폴더: {folder_path}")
    
    def load_current_korean_image(self):
        """현재 선택된 소스 이미지 로드"""
        if not hasattr(self, 'kr_image_list') or not self.kr_image_list or self.kr_current_image_index >= len(self.kr_image_list):
            return
        
        image_path = self.kr_image_list[self.kr_current_image_index]
        
        # 이미지 로드
        if self.kr_canvas.load_image(image_path):
            self.kr_image_path = image_path
            self.kr_image = self.kr_canvas.image
            self.update_status(f"소스 이미지 로드됨: {os.path.basename(image_path)}")
            
            # 현재 이미지 정보 표시
            if hasattr(self, 'kr_current_image_label'):
                filename = os.path.basename(image_path)
                self.kr_current_image_label.setText(f"현재: {filename} ({self.kr_current_image_index + 1}/{len(self.kr_image_list)})")
            else:
                QtWidgets.QMessageBox.critical(self, "오류", f"이미지를 로드할 수 없습니다:\n{image_path}")
    
    def select_japanese_image_folder(self):
        """타겟 이미지 폴더 선택"""
        dialog = QtWidgets.QFileDialog(self, "타겟 이미지 폴더 선택")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if self.jp_last_folder:
            dialog.setDirectory(self.jp_last_folder)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        folder_path = selected[0] if selected else ""
        if not folder_path:
            return
        
        # 지원하는 이미지 확장자
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # 폴더에서 이미지 파일들 찾기
        image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "오류", "선택한 폴더에 이미지 파일이 없습니다.")
            return
        
        # 파일명으로 정렬
        image_files.sort()
        
        self.jp_image_list = image_files
        self.jp_current_image_index = 0
        # 마지막 사용 폴더 저장
        self.jp_last_folder = folder_path
        self.save_settings()
        
        # 첫 번째 이미지 로드
        self.load_current_japanese_image()
        
        # 이미지 목록 UI 업데이트
        self.update_jp_image_list_ui()
        
        self.update_status(f"타겟 이미지 폴더 로드됨: {len(self.jp_image_list)}개 파일")
        QtWidgets.QMessageBox.information(self, "폴더 로드 완료", 
            f"{len(self.jp_image_list)}개의 타겟 이미지 파일을 찾았습니다.\n" 
            f"폴더: {folder_path}")
    
    def load_current_japanese_image(self):
        """현재 선택된 타겟 이미지 로드"""
        if not hasattr(self, 'jp_image_list') or not self.jp_image_list or self.jp_current_image_index >= len(self.jp_image_list):
            return
            
        image_path = self.jp_image_list[self.jp_current_image_index]
        
        # 이미지 로드
        if self.jp_canvas.load_image(image_path):
            self.jp_image_path = image_path
            self.jp_image = self.jp_canvas.image
            self.update_status(f"타겟 이미지 로드됨: {os.path.basename(image_path)}")
            
            # 캔버스 초기화 (이전 텍스트 박스 제거)
            self.jp_canvas.update_display_with_preview([])
            
            # 현재 이미지의 텍스트 박스만 표시
            self.update_display_for_current_image()
            
            # 현재 이미지 정보 표시
            if hasattr(self, 'jp_current_image_label'):
                filename = os.path.basename(image_path)
                self.jp_current_image_label.setText(f"현재: {filename} ({self.jp_current_image_index + 1}/{len(self.jp_image_list)})")
        else:
            QtWidgets.QMessageBox.critical(self, "오류", f"이미지를 로드할 수 없습니다:\n{image_path}")
    
    def update_display_for_current_image(self):
        """현재 이미지에 해당하는 텍스트 박스만 표시 (성능 최적화)"""
        if not self.jp_image_path:
            return
            
        current_filename = os.path.basename(self.jp_image_path)
        
        # 성능 최적화: 현재 이미지의 텍스트 박스만 필터링
        current_text_regions = []
        for region in self.text_regions:
            if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                current_text_regions.append(region)
        
        # 가운데 텍스트 영역은 모든 텍스트 표시
        self.update_text_table()
        
        # 타겟 이미지 영역에는 현재 이미지의 텍스트 박스만 표시
        if hasattr(self.jp_canvas, 'update_display_with_preview'):
            self.jp_canvas.update_display_with_preview(current_text_regions)
        
        # 선택된 텍스트 인덱스가 현재 이미지의 텍스트 박스가 아닌 경우 초기화
        if (hasattr(self.jp_canvas, 'selected_text_index') and 
            self.jp_canvas.selected_text_index >= 0 and 
            self.jp_canvas.selected_text_index < len(self.text_regions)):
            selected_region = self.text_regions[self.jp_canvas.selected_text_index]
            if (not hasattr(selected_region, 'image_filename') or 
                selected_region.image_filename != current_filename):
                self.jp_canvas.selected_text_index = -1
    
    
    def update_text_table_for_regions(self, regions):
        """특정 텍스트 영역들만 테이블에 표시"""
        self.text_table.setRowCount(len(regions))
        
        for i, region in enumerate(regions):
            # 전체 텍스트 박스 목록에서의 실제 인덱스 찾기
            actual_index = self.text_regions.index(region)
            
            self.text_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(actual_index + 1)))
            
            # 드래그 가능한 텍스트 아이템 생성
            text_item = DraggableTableWidgetItem(region.text, actual_index)
            self.text_table.setItem(i, 1, text_item)
            
            # 위치 정보 표시
            if region.is_positioned and region.target_bbox:
                pos_text = f"({region.target_bbox[0]}, {region.target_bbox[1]})"
                status_text = "✅ 위치 설정됨"
            else:
                pos_text = "미설정"
                status_text = "⏳ 대기 중"
            
            self.text_table.setItem(i, 2, QtWidgets.QTableWidgetItem(pos_text))
            self.text_table.setItem(i, 3, QtWidgets.QTableWidgetItem(status_text))
            
            # 이미지명 표시
            image_name = region.image_filename if region.image_filename else "미설정"
            image_item = QtWidgets.QTableWidgetItem(image_name)
            if region.image_filename:
                image_item.setBackground(QtGui.QColor(200, 255, 200))  # 연한 초록색
            else:
                image_item.setBackground(QtGui.QColor(255, 200, 200))  # 연한 빨간색
            self.text_table.setItem(i, 4, image_item)
        
        self.text_table.resizeColumnsToContents()
        self.update_stats_for_regions(regions)
    
    def update_stats_for_regions(self, regions):
        """특정 텍스트 영역들에 대한 통계 업데이트"""
        count = len(regions)
        self.stats_label.setText(f"📊 텍스트: {count}개 (현재 이미지)")
    
    def save_csv(self):
        """OCR 결과를 CSV 파일로 저장"""
        if not self.text_regions:
            QtWidgets.QMessageBox.warning(self, "경고", "저장할 텍스트 데이터가 없습니다.")
            return
        
        dialog = QtWidgets.QFileDialog(self, "CSV 파일 저장")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilter("CSV 파일 (*.csv)")
        if self.csv_last_folder:
            dialog.setDirectory(self.csv_last_folder)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected_files = dialog.selectedFiles()
        file_path = selected_files[0] if selected_files else ""
        
        if not file_path:
            return
        
        # 마지막 폴더 저장
        self.csv_last_folder = os.path.dirname(file_path)
        self.save_settings()
        
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                
                # 헤더 작성 (확장된 형식)
                writer.writerow([
                    '번호',          # 0
                    '텍스트',        # 1
                    '이미지파일명',   # 2
                    'x1', 'y1', 'x2', 'y2',  # 3-6: 박스 위치/크기
                    '폰트크기',      # 7
                    '폰트',          # 8
                    '색상B', '색상G', '색상R',  # 9-11
                    '여백',          # 12
                    '줄바꿈모드',    # 13 ("word" / "char")
                    '줄간격',        # 14
                    '볼드',          # 15 (0/1)
                    '정렬',          # 16 ("left"/"center"/"right")
                    'is_positioned', # 17 (0/1)
                    'is_manual'      # 18 (0/1)
                ])
                
                # 데이터 작성
                for i, region in enumerate(self.text_regions):
                    # 기본값 안전 처리
                    text = getattr(region, 'text', "")
                    image_filename = getattr(region, 'image_filename', "") or ""
                    
                    bbox = getattr(region, 'target_bbox', None)
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                    else:
                        x1 = y1 = x2 = y2 = ""
                    
                    font_size = getattr(region, 'font_size', getattr(self, 'default_font_size', 18))
                    font_family = getattr(region, 'font_family', getattr(self, 'default_font_family', "나눔고딕"))
                    
                    color = getattr(region, 'color', (0, 0, 0))
                    try:
                        b, g, r = color
                    except Exception:
                        b = g = r = 0
                    
                    margin = getattr(region, 'margin', 2)
                    wrap_mode = getattr(region, 'wrap_mode', "word")
                    line_spacing = getattr(region, 'line_spacing', 1.2)
                    # bold_level이 있으면 우선 사용, 없으면 bold(bool) 기반으로 설정
                    bold_level = getattr(region, 'bold_level', 1 if getattr(region, 'bold', False) else 0)
                    bold = bold_level
                    text_align = getattr(region, 'text_align', "center")
                    is_positioned = 1 if getattr(region, 'is_positioned', False) else 0
                    is_manual = 1 if getattr(region, 'is_manual', False) else 0
                    
                    writer.writerow([
                        i,
                        text,
                        image_filename,
                        x1, y1, x2, y2,
                        font_size,
                        font_family,
                        b, g, r,
                        margin,
                        wrap_mode,
                        line_spacing,
                        bold,
                        text_align,
                        is_positioned,
                        is_manual,
                    ])
            
            self.update_status(f"CSV 파일 저장 완료: {os.path.basename(file_path)}")
            QtWidgets.QMessageBox.information(self, "저장 완료", f"CSV 파일이 저장되었습니다:\n{file_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"CSV 파일 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def load_csv(self):
        """CSV 파일에서 OCR 결과 불러오기"""
        # 기존 텍스트가 있으면 확인 팝업 표시
        if self.text_regions:
            reply = QtWidgets.QMessageBox.question(
                self, "CSV 불러오기 확인", 
                f"현재 {len(self.text_regions)}개의 텍스트 라인이 있습니다.\n"
                "CSV 파일을 불러오면 기존 텍스트가 모두 삭제됩니다.\n"
                "계속하시겠습니까?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        dialog = QtWidgets.QFileDialog(self, "CSV 파일 불러오기")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setNameFilter("CSV 파일 (*.csv)")
        if self.csv_last_folder:
            dialog.setDirectory(self.csv_last_folder)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected_files = dialog.selectedFiles()
        file_path = selected_files[0] if selected_files else ""
        
        if not file_path:
            return
        
        # 마지막 폴더 저장
        self.csv_last_folder = os.path.dirname(file_path)
        self.save_settings()
        
        try:
            import csv
            
            with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile)
                # 헤더 읽기
                header = next(reader, None)
                
                # 기존 텍스트 영역 초기화
                self.text_regions.clear()
                
                # 헤더 기반 컬럼 인덱스 매핑 (확장 형식 및 구형 형식 모두 지원)
                col = {}
                if header:
                    for idx, name in enumerate(header):
                        col[name] = idx
                
                # 구형 형식(번호, 텍스트) 여부 판별
                is_legacy = not header or len(header) <= 2 or ('텍스트' in col and len(header) == 2)
                
                # 데이터 읽기
                for row in reader:
                    # 최소 텍스트 컬럼 확인
                    if is_legacy:
                        if len(row) < 2:
                            continue
                    else:
                        # 확장 형식에서도 텍스트 컬럼은 필수
                        text_idx = col.get('텍스트', 1)
                        if text_idx >= len(row):
                            continue
                    
                    try:
                        # 텍스트 영역 생성
                        region = TextRegion()
                        
                        # --- 공통: 텍스트 ---
                        if is_legacy:
                            region.text = row[1] if len(row) > 1 and row[1] else ""
                        else:
                            text_idx = col.get('텍스트', 1)
                            region.text = row[text_idx] if text_idx < len(row) and row[text_idx] else ""
                        
                        # 기본값 설정
                        region.font_size = self.default_font_size
                        region.font_family = self.default_font_family
                        region.color = (0, 0, 0)
                        region.margin = 2
                        region.wrap_mode = "word"
                        region.line_spacing = 1.2
                        region.bold = False
                        region.image_filename = None
                        region.is_positioned = False
                        region.is_manual = True  # CSV에서 불러온 텍스트는 수동으로 간주
                        region.text_align = "center"
                        
                        if not is_legacy:
                            # 확장 형식일 때만 추가 정보 파싱 (없으면 기본값 유지)
                            def get(name, default=None):
                                idx = col.get(name)
                                if idx is None or idx >= len(row):
                                    return default
                                return row[idx]
                            
                            # 이미지 파일명
                            img_name = get('이미지파일명', "")
                            region.image_filename = img_name or None
                            
                            # 위치/크기
                            try:
                                x1 = int(get('x1', "") or 0)
                                y1 = int(get('y1', "") or 0)
                                x2 = int(get('x2', "") or 0)
                                y2 = int(get('y2', "") or 0)
                                if x2 > x1 and y2 > y1:
                                    region.target_bbox = (x1, y1, x2, y2)
                                    region.is_positioned = True
                            except ValueError:
                                pass
                            
                            # 폰트
                            try:
                                fs = get('폰트크기')
                                if fs not in (None, ""):
                                    region.font_size = int(fs)
                            except ValueError:
                                pass
                            
                            ff = get('폰트')
                            if ff:
                                region.font_family = ff
                            
                            # 색상
                            try:
                                b = int(get('색상B', "") or 0)
                                g = int(get('색상G', "") or 0)
                                r = int(get('색상R', "") or 0)
                                region.color = (b, g, r)
                            except ValueError:
                                pass
                            
                            # 여백
                            try:
                                m = get('여백')
                                if m not in (None, ""):
                                    region.margin = int(m)
                            except ValueError:
                                pass
                            
                            # 줄바꿈 모드
                            wm = get('줄바꿈모드')
                            if wm in ("word", "char"):
                                region.wrap_mode = wm
                            
                            # 줄간격
                            try:
                                ls = get('줄간격')
                                if ls not in (None, ""):
                                    region.line_spacing = float(ls)
                            except ValueError:
                                pass
                            
                            # 볼드 (정수 레벨 또는 bool 호환)
                            bold_val = get('볼드')
                            if bold_val is not None and bold_val != "":
                                if bold_val in ("0", "1", "2"):
                                    try:
                                        region.bold_level = int(bold_val)
                                    except ValueError:
                                        region.bold_level = 1 if bold_val in ("1", "True", "true") else 0
                                else:
                                    region.bold_level = 1 if bold_val in ("1", "True", "true") else 0
                                region.bold = region.bold_level >= 1
                            
                            # 정렬
                            align = get('정렬')
                            if align in ("left", "center", "right"):
                                region.text_align = align
                            
                            # is_positioned (명시 값이 있으면 덮어씀)
                            ip = get('is_positioned')
                            if ip in ("1", "True", "true"):
                                region.is_positioned = bool(region.target_bbox)
                            
                            # is_manual
                            im = get('is_manual')
                            if im in ("0", "False", "false"):
                                region.is_manual = False
                        
                        self.text_regions.append(region)
                        
                    except Exception as e:
                        logger.error(f"CSV 행 처리 오류: {e}, 행: {row}")
                        continue
            
            # UI 업데이트 - 모든 텍스트 표시 (CSV 로딩 후)
            if hasattr(self, 'text_table'):
                self.update_text_table()
            if hasattr(self, 'jp_canvas'):
                self.jp_canvas.update_display()
            
            # 현재 이미지가 있으면 해당 이미지의 텍스트 박스만 표시
            if self.jp_image_path:
                self.update_display_for_current_image()
            
            self.update_status(f"CSV 파일 불러오기 완료: {os.path.basename(file_path)}")
            QtWidgets.QMessageBox.information(self, "불러오기 완료", 
                f"CSV 파일을 불러왔습니다:\n{file_path}\n총 {len(self.text_regions)}개의 텍스트 영역을 로드했습니다.")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"CSV 파일 불러오기 중 오류가 발생했습니다:\n{str(e)}")
    
    def add_font_file(self):
        """폰트 파일 추가"""
        # 폰트 파일 선택 다이얼로그
        font_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "폰트 파일 선택",
            "",
            "폰트 파일 (*.ttf *.otf *.ttc);;모든 파일 (*)"
        )
        
        if not font_path:
            return
        
        # 폰트 파일이 존재하는지 확인
        if not os.path.exists(font_path):
            QtWidgets.QMessageBox.warning(
                self,
                "오류",
                "선택한 폰트 파일이 존재하지 않습니다."
            )
            return
        
        # 폰트 파일 로드 시도
        try:
            # PIL로 폰트 로드하여 폰트 이름 확인
            from PIL import ImageFont
            test_font = ImageFont.truetype(font_path, 12)
            # 폰트 이름 추출 (파일명 기반 또는 폰트 메타데이터)
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            
            # 폰트 이름 입력 다이얼로그
            font_display_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "폰트 이름 설정",
                f"폰트 파일: {os.path.basename(font_path)}\n\n"
                f"텍스트 박스에서 사용할 폰트 이름을 입력하세요:",
                text=font_name
            )
            
            if not ok or not font_display_name.strip():
                return
            
            font_display_name = font_display_name.strip()
            
            # 폰트 추가
            self.custom_fonts[font_display_name] = font_path
            
            self.update_status(f"폰트 추가 완료: {font_display_name}", "green")
            QtWidgets.QMessageBox.information(
                self,
                "폰트 추가 완료",
                f"폰트가 추가되었습니다.\n\n"
                f"폰트 이름: {font_display_name}\n"
                f"파일 경로: {font_path}\n\n"
                f"이제 텍스트 박스 편집 시 이 폰트를 선택할 수 있습니다."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "폰트 로드 실패",
                f"폰트 파일을 로드할 수 없습니다.\n\n"
                f"오류: {str(e)}\n\n"
                f"올바른 폰트 파일인지 확인하세요."
            )
    
    def update_kr_image_list_ui(self):
        """소스 이미지 목록 UI 업데이트"""
        if not hasattr(self, 'kr_image_list_widget'):
            return
            
        self.kr_image_list_widget.clear()
        
        for i, image_path in enumerate(self.kr_image_list):
            filename = os.path.basename(image_path)
            item = QtWidgets.QListWidgetItem(f"{i+1}. {filename}")
            
            # 현재 선택된 이미지 강조
            if i == self.kr_current_image_index:
                item.setBackground(QtGui.QColor(219, 234, 252))  # 연한 파란색
                item.setForeground(QtGui.QColor(0, 0, 0))
            
            self.kr_image_list_widget.addItem(item)
    
    def on_kr_image_list_click(self, item):
        """소스 이미지 목록에서 이미지 선택"""
        row = self.kr_image_list_widget.row(item)
        if 0 <= row < len(self.kr_image_list):
            self.kr_current_image_index = row
            self.load_current_korean_image()
            self.update_kr_image_list_ui()
    
    def prev_kr_image(self):
        """이전 소스 이미지로 이동"""
        if not self.kr_image_list:
            QtWidgets.QMessageBox.warning(self, "오류", "소스 이미지 목록이 비어있습니다.")
            return
        
        if self.kr_current_image_index > 0:
            self.kr_current_image_index -= 1
            self.load_current_korean_image()
            self.update_kr_image_list_ui()
        else:
            QtWidgets.QMessageBox.information(self, "알림", "첫 번째 이미지입니다.")
    
    def next_kr_image(self):
        """다음 소스 이미지로 이동"""
        if not self.kr_image_list:
            QtWidgets.QMessageBox.warning(self, "오류", "소스 이미지 목록이 비어있습니다.")
            return
        
        if self.kr_current_image_index < len(self.kr_image_list) - 1:
            self.kr_current_image_index += 1
            self.load_current_korean_image()
            self.update_kr_image_list_ui()
        else:
            QtWidgets.QMessageBox.information(self, "알림", "마지막 이미지입니다.")
    
    def update_jp_image_list_ui(self):
        """타겟 이미지 목록 UI 업데이트"""
        if not hasattr(self, 'jp_image_list_widget'):
            return
        
        self.jp_image_list_widget.clear()
        
        for i, image_path in enumerate(self.jp_image_list):
            filename = os.path.basename(image_path)
            item = QtWidgets.QListWidgetItem(f"{i+1}. {filename}")
            
            # 현재 선택된 이미지 강조
            if i == self.jp_current_image_index:
                item.setBackground(QtGui.QColor(219, 234, 252))  # 연한 파란색
                item.setForeground(QtGui.QColor(0, 0, 0))
            
            self.jp_image_list_widget.addItem(item)
    
    def on_jp_image_list_click(self, item):
        """타겟 이미지 목록에서 이미지 선택"""
        row = self.jp_image_list_widget.row(item)
        if 0 <= row < len(self.jp_image_list):
            self.jp_current_image_index = row
            self.load_current_japanese_image()
            self.update_jp_image_list_ui()
    
    def prev_jp_image(self):
        """이전 타겟 이미지로 이동"""
        if not self.jp_image_list:
            QtWidgets.QMessageBox.warning(self, "오류", "타겟 이미지 목록이 비어있습니다.")
            return
        
        if self.jp_current_image_index > 0:
            self.jp_current_image_index -= 1
            self.load_current_japanese_image()
            self.update_jp_image_list_ui()
        else:
            QtWidgets.QMessageBox.information(self, "알림", "첫 번째 이미지입니다.")
    
    def next_jp_image(self):
        """다음 타겟 이미지로 이동"""
        if not self.jp_image_list:
            QtWidgets.QMessageBox.warning(self, "오류", "타겟 이미지 목록이 비어있습니다.")
            return
        
        if self.jp_current_image_index < len(self.jp_image_list) - 1:
            self.jp_current_image_index += 1
            self.load_current_japanese_image()
            self.update_jp_image_list_ui()
        else:
            QtWidgets.QMessageBox.information(self, "알림", "마지막 이미지입니다.")
    
    def clear_text_regions(self):
        """텍스트 영역 초기화"""
        self.text_regions.clear()
        if hasattr(self, 'text_table'):
            self.update_text_table()
        if hasattr(self, 'jp_canvas'):
            self.jp_canvas.update_display()
    
    def on_region_selected(self, region):
        """영역 선택 시 호출 (클라우드 비전 OCR 버전에서는 타겟 위치 선택만)"""
        if region['canvas_id'] == 'jp' and self.jp_image is not None:
            # 타겟 이미지에서 위치 선택
            self.select_target_position(region['bbox'])
    
    def select_target_position(self, bbox):
        """타겟 이미지에서 타겟 위치 선택"""
        if not self.text_regions:
            QtWidgets.QMessageBox.warning(self, "알림", "먼저 소스 이미지에서 텍스트를 추출하세요.")
            return
        
        # 마지막으로 추가된 텍스트 영역의 위치 업데이트
        last_region = self.text_regions[-1]
        last_region.bbox = bbox
        last_region.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # 현재 이미지명 설정
        if self.jp_image_path:
            last_region.image_filename = os.path.basename(self.jp_image_path)
        
        self.update_text_table()
        self.update_status(f"타겟 위치 설정됨: ({bbox[0]}, {bbox[1]})", "green")
    
    def update_text_table(self):
        """텍스트 테이블 업데이트"""
        if not hasattr(self, 'text_table'):
            logger.error("update_text_table: text_table 속성이 없습니다!")
            return
        
        # 테이블 업데이트 중 시그널 차단 (무한 루프 방지)
        self.text_table.blockSignals(True)
        try:
            self.text_table.setRowCount(len(self.text_regions))
            
            for i, region in enumerate(self.text_regions):
                self.text_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
                
                # 드래그 가능한 텍스트 아이템 생성
                text_item = DraggableTableWidgetItem(region.text, i)
                self.text_table.setItem(i, 1, text_item)
                
                # 위치 정보 표시
                if region.is_positioned and region.target_bbox:
                    pos_text = f"({region.target_bbox[0]}, {region.target_bbox[1]})"
                    status_text = "✅ 위치 설정됨"
                else:
                    pos_text = "미설정"
                    status_text = "⏳ 대기 중"
                
                self.text_table.setItem(i, 2, QtWidgets.QTableWidgetItem(pos_text))
                self.text_table.setItem(i, 3, QtWidgets.QTableWidgetItem(status_text))
                
                # 이미지명 표시
                image_name = region.image_filename if region.image_filename else "미설정"
                image_item = QtWidgets.QTableWidgetItem(image_name)
                if region.image_filename:
                    image_item.setBackground(QtGui.QColor(200, 255, 200))  # 연한 초록색
                else:
                    image_item.setBackground(QtGui.QColor(255, 200, 200))  # 연한 빨간색
                self.text_table.setItem(i, 4, image_item)
            
            self.text_table.resizeColumnsToContents()
            self.update_stats()
        finally:
            self.text_table.blockSignals(False)
    
    def update_stats(self):
        """통계 업데이트"""
        count = len(self.text_regions)
        self.stats_label.setText(f"📊 텍스트: {count}개")
    
    def update_status(self, message, color="blue"):
        """상태 업데이트"""
        self.status_label.setText(message)
        if color == "orange":
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        elif color == "green":
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif color == "red":
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
    
    def set_vision_credentials_dialog(self):
        """구글 클라우드 비전 API 서비스 계정 키 파일 설정 다이얼로그"""
        # 파일 선택 다이얼로그
        credentials_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "서비스 계정 키 파일 선택",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if credentials_path:
            # 파일 경로 설정
            success = self.ocr_engine.set_credentials_path(credentials_path)
            if success:
                self.vision_ocr_btn.setEnabled(True)
                self.update_status("구글 클라우드 비전 API 인증 완료", "green")
                QtWidgets.QMessageBox.information(
                    self, 
                    "인증 파일 설정 완료",
                    "구글 클라우드 비전 API 인증이 완료되었습니다.\n"
                    "이제 클라우드 비전 OCR 기능을 사용할 수 있습니다.\n\n"
                    "💡 사용 방법:\n"
                    "1. 소스 이미지 폴더 선택\n"
                    "2. '👁️ 클라우드 비전 OCR' 버튼 클릭\n"
                    "3. 전체 이미지가 OCR 처리되어 텍스트 리스트에 추가됩니다.\n\n"
                    f"인증 파일: {os.path.basename(credentials_path)}"
                )
            else:
                self.vision_ocr_btn.setEnabled(False)
                self.update_status("구글 클라우드 비전 API 인증 실패", "red")
                QtWidgets.QMessageBox.warning(
                    self,
                    "인증 파일 설정 실패",
                    "구글 클라우드 비전 API 인증에 실패했습니다.\n\n"
                    "확인 사항:\n"
                    "• 서비스 계정 키 파일이 올바른지 확인하세요\n"
                    "• google-cloud-vision 패키지 설치: pip install google-cloud-vision\n"
                    "• Google Cloud Console에서 Cloud Vision API 활성화 확인\n"
                    "• 서비스 계정에 'Cloud Vision API 사용자' 역할 부여 확인\n"
                    "• 키 파일이 손상되지 않았는지 확인하세요"
                )
    
    def run_vision_ocr(self):
        """구글 클라우드 비전 OCR 실행 (전체 이미지)"""
        if not CLOUD_VISION_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "패키지 미설치",
                "google-cloud-vision 패키지가 설치되지 않았습니다.\n"
                "pip install google-cloud-vision로 설치하세요.\n\n"
                "또한 Google Cloud Console에서:\n"
                "1. Cloud Vision API 활성화\n"
                "2. 서비스 계정 생성 및 키 파일 다운로드\n"
                "가 필요합니다."
            )
            return
        
        if not self.ocr_engine.vision_client:
            QtWidgets.QMessageBox.warning(
                self,
                "인증 파일 미설정",
                "구글 클라우드 비전 API 인증 파일이 설정되지 않았습니다.\n"
                "먼저 서비스 계정 키 파일을 설정하세요."
            )
            return
        
        if not self.kr_image_path:
            QtWidgets.QMessageBox.warning(
                self,
                "이미지 없음",
                "소스 이미지를 먼저 로드하세요."
            )
            return
        
        # 확인 대화상자 (API 비용 경고)
        reply = QtWidgets.QMessageBox.question(
            self,
            "클라우드 비전 OCR 실행 확인",
            f"현재 이미지 '{os.path.basename(self.kr_image_path)}'에 대해\n"
            "구글 클라우드 비전 OCR을 실행하시겠습니까?\n\n"
            "⚠️ 주의: API 사용 시 비용이 발생할 수 있습니다.\n"
            "전체 이미지가 OCR 처리됩니다.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # OCR 실행 (별도 스레드에서)
        self.update_status("구글 클라우드 비전 OCR 처리 중...", "orange")
        self.vision_ocr_btn.setEnabled(False)  # 중복 실행 방지
        
        def ocr_worker():
            try:
                # 전체 이미지 OCR 수행
                text_lines = self.ocr_engine.extract_text_full_image_vision(self.kr_image_path)
                
                # PyQt5 시그널을 통해 메인 스레드로 전달 (스레드 안전)
                self.vision_ocr_completed.emit(text_lines)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"클라우드 비전 OCR 오류: {error_msg}")
                import traceback
                logger.error(traceback.format_exc())
                # PyQt5 시그널을 통해 메인 스레드로 에러 전달
                self.vision_ocr_failed.emit(error_msg)
        
        # 별도 스레드에서 OCR 실행
        ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
        ocr_thread.start()
    
    def on_vision_ocr_completed(self, text_lines):
        """구글 클라우드 비전 OCR 완료 시 호출"""
        self.vision_ocr_btn.setEnabled(True)
        
        if not text_lines:
            logger.warning("on_vision_ocr_completed: text_lines가 비어있음")
            self.update_status("구글 클라우드 비전 OCR: 텍스트를 찾을 수 없습니다", "orange")
            QtWidgets.QMessageBox.information(
                self,
                "OCR 완료",
                "이미지에서 텍스트를 찾을 수 없습니다."
            )
            return
        
        # 각 텍스트 라인을 텍스트 영역으로 추가
        added_count = 0
        initial_count = len(self.text_regions)
        
        for text_line in text_lines:
            if text_line.strip():
                text_region = TextRegion(
                    text=text_line.strip(),
                    bbox=None,  # 영역 설정 없음
                    font_size=self.default_font_size,  # 기본 폰트 크기 사용
                    color=self.get_current_color(),
                    font_family=self.default_font_family,  # 기본 폰트 사용
                    margin=2
                )
                text_region.image_filename = None  # 아직 타겟 이미지에 배치되지 않음
                text_region.is_positioned = False
                text_region.is_manual = False  # OCR로 자동 추가됨
                self.text_regions.append(text_region)
                added_count += 1
        
        # UI 업데이트
        if hasattr(self, 'text_table'):
            self.update_text_table()
            # 테이블 강제 새로고침 및 스크롤 맨 위로 이동
            self.text_table.viewport().update()
            if len(self.text_regions) > 0:
                self.text_table.scrollToTop()
        else:
            logger.error("text_table 속성이 없습니다!")
        
        self.update_status(f"구글 클라우드 비전 OCR 완료: {added_count}개 텍스트 라인 추가됨", "green")
        
        QtWidgets.QMessageBox.information(
            self,
            "OCR 완료",
            f"구글 클라우드 비전 OCR이 완료되었습니다.\n"
            f"{added_count}개의 텍스트 라인이 추가되었습니다.\n\n"
            f"추가된 텍스트:\n" + "\n".join([f"- {line[:30]}..." if len(line) > 30 else f"- {line}" 
                                            for line in text_lines[:10]]) +
            (f"\n... 외 {len(text_lines) - 10}개" if len(text_lines) > 10 else "")
        )
    
    def on_vision_ocr_failed(self, error_message):
        """구글 클라우드 비전 OCR 실패 시 호출"""
        self.vision_ocr_btn.setEnabled(True)
        self.update_status(f"구글 클라우드 비전 OCR 실패", "red")
        
        # 오류 메시지가 여러 줄인 경우 (이미 포맷된 경우)
        if "\n" in error_message:
            QtWidgets.QMessageBox.critical(
                self,
                "OCR 실패",
                error_message
            )
        else:
            # 일반 오류 메시지
            QtWidgets.QMessageBox.critical(
                self,
                "OCR 실패",
                f"구글 클라우드 비전 OCR 처리 중 오류가 발생했습니다:\n\n{error_message}\n\n"
                "확인 사항:\n"
                "• API 키가 올바른지 확인하세요\n"
                "• Google Cloud Console에서 Generative Language API가 활성화되어 있는지 확인하세요\n"
                "• 인터넷 연결 상태를 확인하세요\n"
                "• API 사용 한도를 확인하세요"
            )
    
    def choose_color(self):
        """텍스트 색상 선택"""
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    color: {'white' if color.lightness() < 128 else 'black'};
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    width: 30px;
                    height: 25px;
                }}
            """)
            
            # 선택된 텍스트 영역에 색상 적용
            current_row = self.text_table.currentRow()
            if current_row >= 0 and current_row < len(self.text_regions):
                region = self.text_regions[current_row]
                region.color = (color.blue(), color.green(), color.red())  # BGR 순서
                
                # UI 업데이트
                self.update_text_table()
                if hasattr(self, 'jp_canvas'):
                    # 현재 이미지의 텍스트 박스만 표시
                    self.update_display_for_current_image()
            
            # 기본 색상 값도 갱신
            self.default_color_bgr = (color.blue(), color.green(), color.red())
            # 설정 파일 저장
            self.save_settings()
    
    def get_current_color(self):
        """현재 선택된 색상 반환"""
        # 색상 버튼의 배경색에서 RGB 값 추출
        style = self.color_btn.styleSheet() if hasattr(self, "color_btn") else ""
        if "background-color:" in style:
            color_str = style.split("background-color:")[1].split(";")[0].strip()
            if color_str.startswith("#"):
                # HEX 색상을 RGB로 변환
                hex_color = color_str.lstrip("#")
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (b, g, r)  # OpenCV는 BGR 순서
        return (0, 0, 0)  # 기본값: 검은색

    def apply_default_color_to_button(self):
        """기본 색상을 색상 버튼에 적용"""
        if not hasattr(self, "color_btn"):
            return
        b, g, r = getattr(self, "default_color_bgr", (0, 0, 0))
        # BGR → HEX (Qt는 RGB)
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        # 간단한 밝기 계산으로 글자색 결정
        lightness = (r + g + b) // 3
        text_color = "white" if lightness < 128 else "black"
        self.color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_hex};
                color: {text_color};
                border: 1px solid #ccc;
                border-radius: 3px;
                width: 30px;
                height: 25px;
            }}
        """)
    
    def clear_all_texts(self):
        """모든 텍스트 삭제"""
        reply = QtWidgets.QMessageBox.question(
            self, "전체 삭제", "모든 텍스트를 삭제하시겠습니까?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.text_regions.clear()
            self.update_text_table()
            self.update_status("모든 텍스트가 삭제되었습니다", "green")
    
    def delete_selected_text(self):
        """선택된 텍스트 박스 삭제"""
        current_row = self.text_table.currentRow()
        
        if current_row < 0 or current_row >= len(self.text_regions):
            QtWidgets.QMessageBox.warning(
                self, 
                "삭제 불가", 
                "삭제할 텍스트를 선택하세요.\n\n"
                "텍스트 테이블에서 행을 클릭하여 선택한 후 삭제 버튼을 클릭하세요."
            )
            return
        
        # 삭제 확인 대화상자
        region = self.text_regions[current_row]
        reply = QtWidgets.QMessageBox.question(
            self,
            "텍스트 삭제 확인",
            f"다음 텍스트를 삭제하시겠습니까?\n\n"
            f"텍스트: {region.text[:50]}{'...' if len(region.text) > 50 else ''}\n"
            f"번호: {current_row + 1}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # 텍스트 삭제
            del self.text_regions[current_row]
            self.update_text_table()
            self.update_status(f"텍스트 {current_row + 1} 삭제됨", "green")
            
            # 캔버스 선택 상태 초기화
            if hasattr(self, 'jp_canvas'):
                self.jp_canvas.selected_text_index = -1
                self.jp_canvas.resizing = False
                self.jp_canvas.moving = False
                self.jp_canvas.resize_handle = None
            
            # 현재 이미지의 텍스트 박스만 표시
            if hasattr(self, 'update_display_for_current_image'):
                self.update_display_for_current_image()
    
    def reset_text_position(self):
        """선택된 텍스트 박스의 이미지 및 위치 정보 초기화"""
        current_row = self.text_table.currentRow()
        
        if current_row < 0 or current_row >= len(self.text_regions):
            QtWidgets.QMessageBox.warning(
                self, 
                "초기화 불가", 
                "초기화할 텍스트를 선택하세요.\n\n"
                "텍스트 테이블에서 행을 클릭하여 선택한 후 위치 초기화 버튼을 클릭하세요."
            )
            return
        
        region = self.text_regions[current_row]
        
        # 위치 정보가 없는 경우 경고
        if not region.is_positioned:
            QtWidgets.QMessageBox.warning(
                self,
                "초기화 불가",
                "선택한 텍스트에 위치 정보가 없습니다.\n\n"
                "이미 초기화된 상태이거나 위치가 설정되지 않은 텍스트입니다."
            )
            return
        
        # 초기화 확인 대화상자
        reply = QtWidgets.QMessageBox.question(
            self,
            "위치 초기화 확인",
            f"다음 텍스트의 이미지 및 위치 정보를 초기화하시겠습니까?\n\n"
            f"텍스트: {region.text[:50]}{'...' if len(region.text) > 50 else ''}\n"
            f"번호: {current_row + 1}\n\n"
            f"⚠️ 주의: 텍스트 내용은 유지되며, 위치 정보만 초기화됩니다.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # 위치 정보 초기화
            region.target_bbox = None
            region.is_positioned = False
            region.image_filename = None
            
            # 테이블 업데이트
            self.update_text_table()
            self.update_status(f"텍스트 {current_row + 1}의 위치 정보 초기화됨", "green")
            
            # 캔버스 선택 상태 초기화
            if hasattr(self, 'jp_canvas'):
                self.jp_canvas.selected_text_index = -1
                self.jp_canvas.resizing = False
                self.jp_canvas.moving = False
                self.jp_canvas.resize_handle = None
            
            # 현재 이미지의 텍스트 박스만 표시
            if hasattr(self, 'update_display_for_current_image'):
                self.update_display_for_current_image()
    
    def merge_selected_lines(self):
        """선택된 여러 라인을 하나로 합치기"""
        # 선택된 행들의 인덱스 가져오기
        selected_rows = []
        for item in self.text_table.selectedItems():
            row = item.row()
            if row not in selected_rows:
                selected_rows.append(row)
        
        # 선택된 행이 없거나 1개만 있으면 경고
        if len(selected_rows) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "라인 합치기",
                "합치려면 최소 2개 이상의 라인을 선택하세요.\n\n"
                "Ctrl 키를 누른 채로 여러 라인을 클릭하여 선택할 수 있습니다."
            )
            return
        
        # 행 번호를 오름차순으로 정렬 (첫 번째 행이 합쳐질 대상)
        selected_rows.sort()
        
        # 첫 번째 선택된 라인에 모든 텍스트 합치기
        first_row = selected_rows[0]
        merged_texts = []
        
        for row in selected_rows:
            if 0 <= row < len(self.text_regions):
                text = self.text_regions[row].text
                if text.strip():
                    merged_texts.append(text.strip())
        
        if merged_texts:
            # 첫 번째 라인의 텍스트를 합친 텍스트로 변경 (줄 바꿈으로 합치기)
            self.text_regions[first_row].text = "\n".join(merged_texts)
            
            # 나머지 라인들 삭제 (내림차순으로 삭제해야 인덱스 문제 없음)
            rows_to_delete = selected_rows[1:]  # 첫 번째 행 제외
            rows_to_delete.sort(reverse=True)  # 내림차순 정렬
            
            for row in rows_to_delete:
                if 0 <= row < len(self.text_regions):
                    del self.text_regions[row]
            
            # 테이블 업데이트 (시그널 차단하여 선택 상태 변경 방지)
            self.text_table.blockSignals(True)
            self.update_text_table()
            self.text_table.blockSignals(False)
            
            # 합쳐진 라인으로 포커스 이동
            # 삭제된 행들 때문에 인덱스가 변경되었을 수 있으므로, 
            # 합쳐진 라인(first_row)의 새로운 인덱스를 계산
            # first_row는 삭제되지 않았으므로, 삭제된 행들 중 first_row보다 작은 행의 개수를 빼면 됨
            deleted_before_first = sum(1 for r in rows_to_delete if r < first_row)
            merged_row_index = first_row - deleted_before_first
            
            # 합쳐진 라인 선택
            if 0 <= merged_row_index < self.text_table.rowCount():
                self.text_table.selectRow(merged_row_index)
                # 테이블 스크롤하여 선택된 행이 보이도록
                self.text_table.scrollTo(self.text_table.model().index(merged_row_index, 0))
            
            # 캔버스에서도 합쳐진 라인 선택
            if hasattr(self, 'jp_canvas'):
                # 텍스트 영역 인덱스는 삭제 후의 인덱스로 조정
                # text_regions에서 first_row에 해당하는 인덱스 찾기
                if 0 <= merged_row_index < len(self.text_regions):
                    self.jp_canvas.selected_text_index = merged_row_index
                else:
                    self.jp_canvas.selected_text_index = -1
                self.jp_canvas.resizing = False
                self.jp_canvas.moving = False
                self.jp_canvas.resize_handle = None
            
            self.update_status(
                f"{len(selected_rows)}개 라인이 {first_row + 1}번 라인으로 합쳐졌습니다", 
                "green"
            )
            
            # 현재 이미지의 텍스트 박스만 표시 (선택 상태 변경 없이)
            if hasattr(self, 'update_display_for_current_image'):
                self.update_display_for_current_image()
            
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "라인 합치기",
                "합칠 텍스트가 없습니다."
            )
    
    def show_text_table_context_menu(self, position):
        """텍스트 테이블 컨텍스트 메뉴 표시"""
        # 선택된 행들의 인덱스 가져오기
        selected_rows = []
        for item in self.text_table.selectedItems():
            row = item.row()
            if row not in selected_rows:
                selected_rows.append(row)
        
        # 컨텍스트 메뉴 생성
        menu = QtWidgets.QMenu(self)
        
        # 텍스트 합치기 액션 (2개 이상 선택된 경우에만 활성화)
        merge_action = menu.addAction("🔗 텍스트 합치기")
        merge_action.setEnabled(len(selected_rows) >= 2)
        
        if len(selected_rows) < 2:
            merge_action.setToolTip("2개 이상의 라인을 선택해야 합니다.")
        
        # 메뉴 표시
        action = menu.exec_(self.text_table.viewport().mapToGlobal(position))
        
        # 액션 처리
        if action == merge_action and len(selected_rows) >= 2:
            self.merge_selected_lines()
    
    def add_manual_text_line(self):
        """수동으로 텍스트 라인 추가"""
        # 텍스트 입력 다이얼로그
        text, ok = QtWidgets.QInputDialog.getText(
            self, "텍스트 라인 추가", "추가할 텍스트를 입력하세요:"
        )
        
        # 빈 문자열이 아닌 경우 모두 허용 (스페이스만 있어도 허용)
        if ok and text:
            # 새로운 텍스트 영역 생성
            region = TextRegion()
            region.text = text  # strip() 제거하여 원본 텍스트 유지 (스페이스 포함)
            region.font_size = self.default_font_size  # 기본 폰트 크기 사용
            region.font_family = self.default_font_family  # 기본 폰트 사용
            region.color = self.get_current_color()
            region.margin = 2
            region.wrap_mode = "word"
            region.line_spacing = 1.2
            region.bold = False
            region.image_filename = None
            region.is_positioned = False
            region.is_manual = True  # 수동 추가 표시
            
            # 텍스트 영역 리스트에 추가
            self.text_regions.append(region)
            
            # UI 업데이트
            self.update_text_table()
            if hasattr(self, 'jp_canvas'):
                self.jp_canvas.update_display()
            
            self.update_status(f"수동 텍스트 라인 추가됨: {text[:20]}...", "green")
    
    def start_text_drag(self, supportedActions):
        """텍스트 드래그 시작"""
        current_row = self.text_table.currentRow()
        if current_row >= 0 and current_row < len(self.text_regions):
            # 드래그 데이터 생성
            mime_data = QtCore.QMimeData()
            mime_data.setText(f"text_index:{current_row}")
            
            # 드래그 시작
            drag = QtGui.QDrag(self.text_table)
            drag.setMimeData(mime_data)
            
            # 드래그 아이콘 설정
            pixmap = QtGui.QPixmap(200, 30)
            pixmap.fill(QtGui.QColor(100, 100, 100, 150))
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, f"텍스트: {self.text_regions[current_row].text[:20]}...")
            painter.end()
            drag.setPixmap(pixmap)
            
            # 드래그 실행
            drag.exec_(QtCore.Qt.MoveAction)
    
    def on_text_dropped(self, text_index, position_data):
        """텍스트가 타겟 이미지에 드롭되었을 때"""
        if text_index >= 0 and text_index < len(self.text_regions):
            region = self.text_regions[text_index]
            region.target_bbox = position_data['bbox']
            region.is_positioned = True
            
            # 현재 이미지 파일명 저장
            if self.jp_image_path:
                region.image_filename = os.path.basename(self.jp_image_path)
            
            self.update_text_table()
            self.update_status(f"텍스트 '{region.text[:20]}...' 위치 설정됨", "green")
            
            # 현재 이미지의 텍스트 박스만 다시 표시 (다른 페이지 텍스트 박스 제거)
            self.update_display_for_current_image()
    
    def show_text_preview(self, text_index):
        """타겟 이미지에 텍스트 미리보기 표시"""
        if text_index >= 0 and text_index < len(self.text_regions):
            region = self.text_regions[text_index]
            if region.is_positioned and region.target_bbox:
                # 현재 이미지의 텍스트 박스만 표시
                self.update_display_for_current_image()
    
    def on_table_item_changed(self, item):
        """테이블 아이템 변경 이벤트 (인라인 편집)"""
        row = item.row()
        col = item.column()
        
        if row >= 0 and row < len(self.text_regions):
            region = self.text_regions[row]
            
            if col == 1:  # 텍스트 컬럼 변경
                new_text = item.text().strip()
                if new_text and new_text != region.text:
                    region.text = new_text
                    self.update_status(f"텍스트 수정됨: {new_text[:20]}...", "green")
                    
                    # 타겟 이미지 미리보기 업데이트
                    if hasattr(self, 'jp_canvas'):
                        # 현재 이미지의 텍스트 박스만 표시
                        self.update_display_for_current_image()
    
    def on_table_item_double_clicked(self, item):
        """테이블 아이템 더블클릭 이벤트"""
        row = item.row()
        col = item.column()
        
        if row >= 0 and row < len(self.text_regions):
            region = self.text_regions[row]
            
            if col == 1:  # 텍스트 컬럼 더블클릭 - 인라인 편집 활성화
                # 편집 모드로 전환
                self.text_table.editItem(item)
            
            elif col == 2:  # 위치 컬럼 더블클릭
                # 위치 수동 설정
                if self.jp_image is None:
                    QtWidgets.QMessageBox.warning(self, "알림", "타겟 이미지를 먼저 로드하세요.")
                    return
                
                # 현재 위치 표시
                current_pos = "미설정"
                if region.is_positioned and region.target_bbox:
                    current_pos = f"({region.target_bbox[0]}, {region.target_bbox[1]})"
                
                # 위치 입력 대화상자
                pos_text, ok = QtWidgets.QInputDialog.getText(
                    self, "위치 설정", 
                    f"텍스트 '{region.text[:20]}...'의 위치를 설정하세요:\n"
                    f"형식: x,y,width,height\n"
                    f"현재: {current_pos}",
                    text=current_pos if current_pos != "미설정" else "100,100,200,50"
                )
                
                if ok and pos_text.strip():
                    try:
                        # 위치 파싱
                        parts = pos_text.split(',')
                        if len(parts) == 4:
                            x, y, w, h = map(int, parts)
                            region.target_bbox = (x, y, x + w, y + h)
                            region.is_positioned = True
                            
                            self.update_text_table()
                            # 현재 이미지의 텍스트 박스만 표시
                            self.update_display_for_current_image()
                            self.update_status(f"위치 설정됨: ({x}, {y})", "green")
                        else:
                            QtWidgets.QMessageBox.warning(self, "오류", "위치 형식이 올바르지 않습니다.\n형식: x,y,width,height")
                    except ValueError:
                        QtWidgets.QMessageBox.warning(self, "오류", "위치 값이 올바르지 않습니다.")
    
    def on_font_size_changed(self, value):
        """폰트 크기 스핀박스 변경 시"""
        self.font_size_slider.blockSignals(True)
        self.font_size_slider.setValue(value)
        self.font_size_slider.blockSignals(False)
        
        # 현재 선택된 텍스트의 폰트 크기 업데이트
        current_row = self.text_table.currentRow()
        if current_row >= 0 and current_row < len(self.text_regions):
            self.text_regions[current_row].font_size = value
            # 현재 이미지의 텍스트 박스만 표시
            self.update_display_for_current_image()
    
    def on_font_size_slider_changed(self, value):
        """폰트 크기 슬라이더 변경 시"""
        self.font_size_spin.blockSignals(True)
        self.font_size_spin.setValue(value)
        self.font_size_spin.blockSignals(False)
        
        # 현재 선택된 텍스트의 폰트 크기 업데이트
        current_row = self.text_table.currentRow()
        if current_row >= 0 and current_row < len(self.text_regions):
            self.text_regions[current_row].font_size = value
            # 현재 이미지의 텍스트 박스만 표시
            self.update_display_for_current_image()
    
    def change_default_font_size(self):
        """기본 폰트 크기 변경"""
        # 현재 기본값 표시
        current_size = self.default_font_size
        
        # 기본 폰트 크기 선택 다이얼로그
        sizes = [18, 20, 22, 24, 26, 28, 30]
        size_text, ok = QtWidgets.QInputDialog.getItem(
            self,
            "기본 폰트 크기 변경",
            f"새로운 기본 폰트 크기를 선택하세요:\n\n"
            f"현재 기본값: {current_size}\n\n"
            f"⚠️ 주의: 이미 추가된 텍스트는 변경되지 않으며,\n"
            f"새로 추가되는 텍스트에만 적용됩니다.",
            [str(s) for s in sizes],
            sizes.index(current_size) if current_size in sizes else 0,
            False
        )
        
        if ok and size_text:
            try:
                new_size = int(size_text)
                if 8 <= new_size <= 72:
                    self.default_font_size = new_size
                    
                    # 버튼 텍스트 업데이트
                    if hasattr(self, 'default_font_size_btn'):
                        self.default_font_size_btn.setText(f"📏 기본: {self.default_font_size}")
                    
                    # 스핀박스와 슬라이더도 업데이트
                    self.font_size_spin.blockSignals(True)
                    self.font_size_slider.blockSignals(True)
                    self.font_size_spin.setValue(self.default_font_size)
                    self.font_size_slider.setValue(self.default_font_size)
                    self.font_size_spin.blockSignals(False)
                    self.font_size_slider.blockSignals(False)
                    
                    self.update_status(f"기본 폰트 크기 변경: {current_size} → {new_size}", "green")
                    QtWidgets.QMessageBox.information(
                        self,
                        "기본 폰트 크기 변경 완료",
                        f"기본 폰트 크기가 {current_size}에서 {new_size}로 변경되었습니다.\n\n"
                        f"새로 추가되는 텍스트는 {new_size} 크기로 설정됩니다."
                    )
                    
                    # 설정 파일 저장
                    self.save_settings()
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "오류",
                        "폰트 크기는 8~72 사이의 값이어야 합니다."
                    )
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self,
                    "오류",
                    "올바른 숫자를 입력하세요."
                )
    
    def change_default_font(self):
        """기본 폰트 변경"""
        # 기본 폰트 목록
        default_fonts = ["Arial", "Times New Roman", "Courier New", "굴림", "맑은 고딕", "나눔고딕"]
        
        # 사용자 추가 폰트 목록 (⭐ 표시)
        custom_font_names = []
        for font_name in self.custom_fonts.keys():
            if font_name not in default_fonts:
                custom_font_names.append(f"⭐ {font_name}")
        
        # 전체 폰트 목록 (기본 폰트 + 사용자 추가 폰트)
        all_fonts = default_fonts + custom_font_names
        
        # 현재 기본 폰트 인덱스 찾기
        current_index = 0
        if self.default_font_family in default_fonts:
            current_index = default_fonts.index(self.default_font_family)
        else:
            # 사용자 추가 폰트인 경우
            for i, custom_name in enumerate(custom_font_names):
                if custom_name == f"⭐ {self.default_font_family}":
                    current_index = len(default_fonts) + i
                    break
        
        # 폰트 선택 다이얼로그
        font_name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "기본 폰트 변경",
            f"새로운 기본 폰트를 선택하세요:\n\n"
            f"현재 기본 폰트: {self.default_font_family}\n\n"
            f"⭐ 표시는 사용자가 추가한 폰트입니다.",
            all_fonts,
            current_index,
            False
        )
        
        if ok and font_name:
            # ⭐ 표시 제거 (사용자 추가 폰트인 경우)
            if font_name.startswith("⭐ "):
                new_font = font_name[2:]  # "⭐ " 제거
            else:
                new_font = font_name
            
            old_font = self.default_font_family
            self.default_font_family = new_font
            
            # 버튼 텍스트 업데이트
            if hasattr(self, 'default_font_btn'):
                self.default_font_btn.setText(f"🔤 기본 폰트: {self.default_font_family}")
            
            self.update_status(f"기본 폰트 변경: {old_font} → {new_font}", "green")
            QtWidgets.QMessageBox.information(
                self,
                "기본 폰트 변경 완료",
                f"기본 폰트가 '{old_font}'에서 '{new_font}'로 변경되었습니다.\n\n"
                f"새로 추가되는 텍스트는 '{new_font}' 폰트로 설정됩니다."
            )
            
            # 설정 파일 저장
            self.save_settings()
    
    def save_result(self):
        """결과 이미지 저장"""
        if self.jp_image is None:
            QtWidgets.QMessageBox.warning(self, "알림", "타겟 이미지를 먼저 로드하세요.")
            return
        
        # 현재 이미지의 텍스트 박스가 있는지 확인 (성능 최적화)
        current_filename = os.path.basename(self.jp_image_path) if self.jp_image_path else None
        current_text_regions = []
        for region in self.text_regions:
            if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                current_text_regions.append(region)
        
        # 텍스트 박스가 없어도 저장 가능 (원본 이미지만 저장)
        # 저장 옵션 선택 다이얼로그
        save_option = self.show_save_option_dialog()
        if save_option is None:
            return  # 사용자가 취소한 경우
        
        dialog = QtWidgets.QFileDialog(self, "결과 이미지 저장")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilter("이미지 파일 (*.png *.jpg *.jpeg)")
        if self.result_last_folder:
            dialog.setDirectory(self.result_last_folder)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected_files = dialog.selectedFiles()
        file_path = selected_files[0] if selected_files else ""
        
        if file_path:
            # 마지막 폴더 저장
            self.result_last_folder = os.path.dirname(file_path)
            self.save_settings()
            try:
                self.update_status("이미지 생성 중...", "orange")
                
                # 저장 전에 핸들 숨기기 (모든 저장 방식에서 핸들이 저장되지 않도록)
                old_show_handles = None
                if hasattr(self, 'jp_canvas') and self.jp_canvas:
                    old_show_handles = getattr(self.jp_canvas, 'show_handles', True)
                    self.jp_canvas.show_handles = False
                    # 화면 업데이트 (핸들 제거)
                    if hasattr(self, 'update_display_for_current_image'):
                        self.update_display_for_current_image()
                
                # 선택한 옵션에 따라 저장 방식 결정
                if save_option == "widget_capture":
                    # 위젯 캡처 방식 (화면 그대로)
                    self.save_with_widget_capture(file_path)
                elif save_option == "pil_screen":
                    # 화면과 동일한 PIL 방식
                    self.save_with_pil_screen(file_path)
                elif save_option == "pil_hires":
                    # 고해상도 PIL 방식 (2배 해상도)
                    self.save_with_pil_hires(file_path)
                else:  # "qpainter"
                    # QPainter 방식
                    self.save_with_qpainter(file_path)
                
                # 저장 후 핸들 표시 상태 복원
                if old_show_handles is not None and hasattr(self, 'jp_canvas') and self.jp_canvas:
                    self.jp_canvas.show_handles = old_show_handles
                    # 화면 업데이트 (핸들 복원)
                    if hasattr(self, 'update_display_for_current_image'):
                        self.update_display_for_current_image()
                
                self.update_status(f"결과 저장됨: {os.path.basename(file_path)}", "green")
                QtWidgets.QMessageBox.information(
                    self, "저장 완료", 
                    f"결과 이미지가 저장되었습니다:\n{file_path}\n\n"
                    f"저장 방식: {self.get_save_option_name(save_option)}"
                )
                
            except Exception as e:
                # 오류 발생 시에도 핸들 표시 상태 복원
                if 'old_show_handles' in locals() and old_show_handles is not None:
                    if hasattr(self, 'jp_canvas') and self.jp_canvas:
                        self.jp_canvas.show_handles = old_show_handles
                        if hasattr(self, 'update_display_for_current_image'):
                            self.update_display_for_current_image()
                
                self.update_status(f"저장 오류: {str(e)}", "red")
                QtWidgets.QMessageBox.critical(self, "저장 오류", f"이미지 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def show_save_option_dialog(self):
        """저장 옵션 선택 다이얼로그"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("저장 옵션 선택")
        dialog.setModal(True)
        dialog.resize(550, 350)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 설명 라벨
        info_label = QtWidgets.QLabel(
            "저장 방식을 선택하세요:\n\n"
            "• 화면과 동일 (PIL): 화면에서 보는 것과 완전히 동일하게 저장 (기본값)\n"
            "• 고해상도 (PIL 2x): 더 선명한 텍스트를 위해 2배 해상도로 렌더링\n"
            "• 위젯 캡처: 화면에 표시된 위젯을 그대로 캡처\n"
            "• QPainter: QPainter 방식 (호환성용)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 라디오 버튼들
        button_group = QtWidgets.QButtonGroup(dialog)
        
        pil_screen_radio = QtWidgets.QRadioButton("✅ 화면과 동일 (PIL) - 기본값")
        pil_screen_radio.setChecked(True)  # 기본 선택
        pil_screen_radio.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px;")
        button_group.addButton(pil_screen_radio, 0)
        layout.addWidget(pil_screen_radio)
        
        pil_hires_radio = QtWidgets.QRadioButton("🔍 고해상도 (PIL 2x)")
        pil_hires_radio.setStyleSheet("padding: 5px;")
        button_group.addButton(pil_hires_radio, 1)
        layout.addWidget(pil_hires_radio)
        
        widget_capture_radio = QtWidgets.QRadioButton("📸 위젯 캡처 (화면 그대로)")
        widget_capture_radio.setStyleSheet("padding: 5px;")
        button_group.addButton(widget_capture_radio, 2)
        layout.addWidget(widget_capture_radio)
        
        qpainter_radio = QtWidgets.QRadioButton("🖌️ QPainter (기존 방식)")
        qpainter_radio.setStyleSheet("padding: 5px;")
        button_group.addButton(qpainter_radio, 3)
        layout.addWidget(qpainter_radio)
        
        layout.addStretch()
        
        # 버튼들
        button_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("확인")
        cancel_button = QtWidgets.QPushButton("취소")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            if pil_screen_radio.isChecked():
                return "pil_screen"
            elif pil_hires_radio.isChecked():
                return "pil_hires"
            elif widget_capture_radio.isChecked():
                return "widget_capture"
            else:
                return "qpainter"
        else:
            return None
    
    def get_save_option_name(self, option):
        """저장 옵션 이름 반환"""
        names = {
            "widget_capture": "위젯 캡처 (화면 그대로)",
            "pil_screen": "화면과 동일 (PIL)",
            "pil_hires": "고해상도 (PIL 2x)",
            "qpainter": "QPainter (기존 방식)"
        }
        return names.get(option, "알 수 없음")
    
    def save_with_widget_capture(self, file_path):
        """위젯을 QPixmap으로 캡처하여 저장 (화면에 보이는 그대로)"""
        try:
            if not hasattr(self, 'jp_canvas') or self.jp_canvas is None:
                raise Exception("타겟 이미지 캔버스를 찾을 수 없습니다.")
            
            # 원본 이미지 크기로 QPixmap 생성
            if self.jp_image is None:
                raise Exception("타겟 이미지가 없습니다.")
            
            img_height, img_width = self.jp_image.shape[:2]
            
            # 원본 이미지를 QPixmap으로 변환
            jp_rgb = cv2.cvtColor(self.jp_image, cv2.COLOR_BGR2RGB)
            jp_qimage = QImage(jp_rgb.data, img_width, img_height, img_width * 3, QImage.Format_RGB888)
            base_pixmap = QtGui.QPixmap.fromImage(jp_qimage)
            
            # QPainter로 텍스트 오버레이 그리기
            result_pixmap = QtGui.QPixmap(base_pixmap.size())
            result_pixmap.fill(QtCore.Qt.white)
            
            painter = QPainter(result_pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            
            # 배경 이미지 그리기
            painter.drawPixmap(0, 0, base_pixmap)
            
            # 현재 이미지의 텍스트 박스만 저장
            current_filename = os.path.basename(self.jp_image_path) if self.jp_image_path else None
            current_text_regions = []
            for region in self.text_regions:
                if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                    current_text_regions.append(region)
            
            # 화면 렌더링과 동일한 방식으로 텍스트 그리기
            for region in current_text_regions:
                # visible 속성 확인 (기본값 True)
                if not getattr(region, 'visible', True):
                    continue  # 숨김 처리된 텍스트 박스는 저장하지 않음
                
                if not region.is_positioned or not region.target_bbox:
                    continue
                
                x1, y1, x2, y2 = region.target_bbox
                
                # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
                bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
                if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                    painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3]))
                
                # 폰트 설정 (화면과 동일한 계산)
                box_height = y2 - y1
                font_size = max(8, min(int(box_height * 0.6), int(region.font_size)))
                
                # Bold 처리 (bold_level에 따라 굵기/크기 조정)
                bold_level = getattr(region, 'bold_level', 1 if getattr(region, 'bold', False) else 0)
                if bold_level >= 1:
                    # 진하게: 10% 확대
                    font_size = int(font_size * 1.1)
                if bold_level >= 2:
                    # 더 진하게: 추가로 5% 더 확대
                    font_size = int(font_size * 1.15)
                
                font = QFont(region.font_family, font_size)
                font.setPixelSize(font_size)
                if bold_level >= 1:
                    font.setBold(True)
                    # 더 진하게는 더 높은 weight 사용
                    if bold_level >= 2:
                        font.setWeight(QFont.Black)
                    else:
                        font.setWeight(QFont.Bold)
                painter.setFont(font)
                
                # 텍스트 색상 설정 (BGR → RGB)
                text_color = QColor(region.color[2], region.color[1], region.color[0])
                painter.setPen(QPen(text_color))
                
                # 여백 계산
                margin = region.margin
                text_x1 = x1 + margin
                text_y1 = y1 + margin
                text_x2 = x2 - margin
                text_y2 = y2 - margin
                
                # 텍스트 영역이 너무 작으면 최소 크기로 조정
                if text_x2 <= text_x1 or text_y2 <= text_y1:
                    min_width = max(20, font_size * 2)
                    min_height = max(15, font_size)
                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = max(x1 + min_width, x2)
                    text_y2 = max(y1 + min_height, y2)
                
                # 줄바꿈 계산
                box_width = max(10, text_x2 - text_x1)
                if margin < 0:
                    wrap_width = box_width - (margin * 2)
                else:
                    wrap_width = box_width
                
                # 폰트 로드 (줄바꿈 계산용)
                pil_font = self.jp_canvas.load_font_for_overlay(region.font_family, font_size)
                if hasattr(region, 'bold') and region.bold:
                    bold_font_size = int(font_size * 1.1)
                    try:
                        pil_font = self.jp_canvas.load_font_for_overlay(region.font_family, bold_font_size)
                    except:
                        pass
                
                # 텍스트 줄바꿈
                if region.wrap_mode == "word":
                    text_lines = self.jp_canvas.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, pil_font)
                else:
                    text_lines = self.jp_canvas.wrap_text_for_box(region.text, wrap_width, font_size, pil_font)
                
                # 줄간격 계산 (폰트가 안 잘리도록 20% 여유 증가)
                base_line_height = int(font_size * 1.0)
                line_height = int(base_line_height * region.line_spacing)
                total_text_height = len(text_lines) * line_height
                
                # 텍스트가 박스를 넘치면 조정
                available_height = text_y2 - text_y1
                if total_text_height > available_height:
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
                    
                    if total_text_height > available_height:
                        scale_factor = available_height / total_text_height
                        font_size = max(8, int(font_size * scale_factor))
                        line_height = max(font_size, available_height // len(text_lines))
                        total_text_height = len(text_lines) * line_height
                        
                        # 폰트 다시 설정
                        font = QFont(region.font_family, font_size)
                        font.setPixelSize(font_size)
                        if hasattr(region, 'bold') and region.bold:
                            font.setBold(True)
                            font.setWeight(QFont.Bold)
                        painter.setFont(font)
                
                # 텍스트 시작 위치 계산
                start_y = text_y1 + (available_height - total_text_height) // 2
                
                # 각 줄의 텍스트 그리기
                for line_idx, line_text in enumerate(text_lines):
                    if line_text.strip():
                        # 텍스트 너비 계산
                        text_metrics = painter.fontMetrics()
                        line_width = text_metrics.width(line_text)
                        
                        # 텍스트 위치 계산 (정렬 적용)
                        text_align = getattr(region, 'text_align', 'center')
                        if text_align == "left":
                            line_x = text_x1
                        elif text_align == "right":
                            line_x = text_x2 - line_width
                        else:  # "center"
                            line_x = text_x1 + (text_x2 - text_x1 - line_width) // 2
                        line_y = start_y + line_idx * line_height + font_size
                        
                        # 텍스트가 박스 범위 내에 있는지 확인
                        if line_y <= text_y2:
                            # 테두리 적용
                            stroke_color = getattr(region, 'stroke_color', None)
                            stroke_width = getattr(region, 'stroke_width', 0)
                            if stroke_color is not None and stroke_width > 0:
                                # QPainterPath를 사용하여 stroke 구현
                                path = QPainterPath()
                                path.addText(line_x, line_y, font, line_text)
                                # 테두리 그리기
                                stroke_qcolor = QColor(stroke_color[0], stroke_color[1], stroke_color[2])
                                stroke_pen = QPen(stroke_qcolor)
                                stroke_pen.setWidth(stroke_width)
                                stroke_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                                stroke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                                painter.strokePath(path, stroke_pen)
                                # 텍스트 그리기
                                painter.fillPath(path, text_color)
                            else:
                                painter.drawText(line_x, line_y, line_text)
            
            painter.end()
            
            # QPixmap을 이미지 파일로 저장
            success = result_pixmap.save(file_path, "PNG", quality=95)
            
            if not success:
                raise Exception("이미지 저장에 실패했습니다.")
            
        except Exception as e:
            logger.error(f"위젯 캡처 저장 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
    
    def save_with_pil_screen(self, file_path):
        """화면과 완전히 동일한 PIL 방식으로 저장 (권장)"""
        try:
            # 타겟 이미지 복사
            result_image = self.jp_image.copy()
            
            # PIL 이미지로 변환 (화면 렌더링과 동일한 방식)
            base_img = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
            text_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(text_layer)
            
            # 현재 이미지의 텍스트 박스만 저장
            current_filename = os.path.basename(self.jp_image_path) if self.jp_image_path else None
            current_text_regions = []
            for region in self.text_regions:
                if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                    current_text_regions.append(region)
            
            # 화면 렌더링과 동일한 방식으로 텍스트 그리기
            for region in current_text_regions:
                # visible 속성 확인 (기본값 True)
                if not getattr(region, 'visible', True):
                    continue  # 숨김 처리된 텍스트 박스는 저장하지 않음
                
                if not region.is_positioned or not region.target_bbox:
                    continue
                
                x1, y1, x2, y2 = region.target_bbox
                
                # 이미지 크기 가져오기
                img_height, img_width = base_img.size[1], base_img.size[0]
                
                # 안전 클램핑
                x1 = max(0, min(int(x1), img_width - 2))
                y1 = max(0, min(int(y1), img_height - 2))
                x2 = max(x1 + 2, min(int(x2), img_width - 1))
                y2 = max(y1 + 2, min(int(y2), img_height - 1))
                
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                
                # 텍스트 박스 크기 계산
                box_width = x2 - x1
                box_height = y2 - y1
                
                # 폰트 크기를 박스 크기에 맞게 계산 (박스 높이의 60%로 제한)
                font_size = max(8, min(int(box_height * 0.6), int(region.font_size)))
                
                # 여백 계산
                margin = region.margin
                text_x1 = x1 + margin
                text_y1 = y1 + margin
                text_x2 = x2 - margin
                text_y2 = y2 - margin
                
                # 텍스트 영역이 너무 작으면 최소 크기로 조정
                if text_x2 <= text_x1 or text_y2 <= text_y1:
                    min_width = max(20, font_size * 2)
                    min_height = max(15, font_size)
                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = max(x1 + min_width, x2)
                    text_y2 = max(y1 + min_height, y2)
                
                # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
                bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
                if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                    draw.rectangle([x1, y1, x2, y2], fill=bg_color)
                
                # 폰트 로드 (굵기 레벨에 따라 Bold/ExtraBold 폰트 우선 시도)
                bold_level = getattr(region, 'bold_level', 1 if getattr(region, 'bold', False) else 0)
                effective_font_size = font_size
                if bold_level >= 1:
                    effective_font_size = int(effective_font_size * 1.1)
                if bold_level >= 2:
                    effective_font_size = int(effective_font_size * 1.15)
                
                font = self._load_pil_font_with_bold(region.font_family, effective_font_size, bold_level)
                
                # 줄바꿈 계산
                box_width = max(10, text_x2 - text_x1)
                if margin < 0:
                    wrap_width = box_width - (margin * 2)
                else:
                    wrap_width = box_width
                
                # 텍스트 줄바꿈
                if region.wrap_mode == "word":
                    text_lines = self.jp_canvas.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                else:
                    text_lines = self.jp_canvas.wrap_text_for_box(region.text, wrap_width, font_size, font)
                
                # 줄간격 계산
                base_line_height = int(effective_font_size * 1.0)
                line_height = int(base_line_height * region.line_spacing)
                total_text_height = len(text_lines) * line_height
                
                # 텍스트가 박스를 넘치면 조정
                available_height = text_y2 - text_y1
                if total_text_height > available_height:
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
                    
                    if total_text_height > available_height:
                        scale_factor = available_height / total_text_height
                        font_size = max(8, int(font_size * scale_factor))
                        line_height = max(font_size, available_height // len(text_lines))
                        total_text_height = len(text_lines) * line_height
                        
                        # 폰트 다시 로드
                        font = self.jp_canvas.load_font_for_overlay(region.font_family, font_size)
                        if hasattr(region, 'bold') and region.bold:
                            bold_font_size = int(font_size * 1.1)
                            try:
                                font = self.jp_canvas.load_font_for_overlay(region.font_family, bold_font_size)
                            except:
                                pass
                        
                        # 줄바꿈 다시 계산
                        if region.wrap_mode == "word":
                            text_lines = self.jp_canvas.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                        else:
                            text_lines = self.jp_canvas.wrap_text_for_box(region.text, wrap_width, font_size, font)
                        
                        line_height = max(font_size, available_height // len(text_lines))
                        total_text_height = len(text_lines) * line_height
                
                # 텍스트 시작 위치 계산
                start_y = text_y1 + (available_height - total_text_height) // 2
                
                # 텍스트 색상 설정 (BGR → RGB)
                text_color = (region.color[2], region.color[1], region.color[0])
                
                # 각 줄의 텍스트 그리기
                for line_idx, line_text in enumerate(text_lines):
                    if line_text.strip():
                        try:
                            text_width = draw.textlength(line_text, font=font)
                        except Exception:
                            text_width = len(line_text) * font_size * 0.6
                        
                        # 텍스트 위치 계산 (정렬 적용)
                        text_align = getattr(region, 'text_align', 'center')
                        if text_align == "left":
                            text_x = text_x1
                        elif text_align == "right":
                            text_x = text_x2 - text_width
                        else:  # "center"
                            text_x = text_x1 + (text_x2 - text_x1 - text_width) // 2
                        text_y = start_y + line_idx * line_height
                        
                        tolerance = 20
                        if text_x >= text_x1 - tolerance and text_x + text_width <= text_x2 + tolerance and text_y <= text_y2 + tolerance:
                            if text_y + font_size <= text_y2 + tolerance:
                                # 테두리 적용
                                stroke_color = getattr(region, 'stroke_color', None)
                                stroke_width = getattr(region, 'stroke_width', 0)
                                if stroke_color is not None and stroke_width > 0:
                                    draw.text((text_x, text_y), line_text, font=font, fill=text_color,
                                             stroke_width=stroke_width, stroke_fill=stroke_color)
                                else:
                                    draw.text((text_x, text_y), line_text, font=font, fill=text_color)
            
            # 알파 블렌딩
            blended = Image.alpha_composite(base_img, text_layer)
            final_image = blended.convert("RGB")
            
            # 저장
            final_image.save(file_path, "PNG", quality=95)
            
        except Exception as e:
            logger.error(f"PIL 화면 동일 저장 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
    
    def save_with_pil_hires(self, file_path):
        """고해상도 PIL 방식으로 저장 (2배 해상도)"""
        try:
            # 타겟 이미지 복사
            result_image = self.jp_image.copy()
            
            # 2배 해상도로 이미지 확대
            scale = 2
            img_height, img_width = result_image.shape[:2]
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            
            # PIL 이미지로 변환 후 확대
            base_img = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
            base_img = base_img.resize((scaled_width, scaled_height), Image.LANCZOS)
            
            text_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(text_layer)
            
            # 현재 이미지의 텍스트 박스만 저장
            current_filename = os.path.basename(self.jp_image_path) if self.jp_image_path else None
            current_text_regions = []
            for region in self.text_regions:
                if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                    current_text_regions.append(region)
            
            # 텍스트 그리기 (2배 해상도로)
            for region in current_text_regions:
                if not region.is_positioned or not region.target_bbox:
                    continue
                
                # bbox를 2배로 확대
                x1, y1, x2, y2 = region.target_bbox
                x1 = int(x1 * scale)
                y1 = int(y1 * scale)
                x2 = int(x2 * scale)
                y2 = int(y2 * scale)
                
                # 이미지 크기 가져오기
                img_height_scaled, img_width_scaled = base_img.size[1], base_img.size[0]
                
                # 안전 클램핑
                x1 = max(0, min(x1, img_width_scaled - 2))
                y1 = max(0, min(y1, img_height_scaled - 2))
                x2 = max(x1 + 2, min(x2, img_width_scaled - 1))
                y2 = max(y1 + 2, min(y2, img_height_scaled - 1))
                
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                
                # 텍스트 박스 크기 계산
                box_width = x2 - x1
                box_height = y2 - y1
                
                # 폰트 크기를 2배로 (고해상도)
                font_size = max(8, min(int(box_height * 0.6), int(region.font_size * scale)))
                
                # 여백 계산 (2배)
                margin = region.margin * scale
                text_x1 = x1 + margin
                text_y1 = y1 + margin
                text_x2 = x2 - margin
                text_y2 = y2 - margin
                
                # 텍스트 영역이 너무 작으면 최소 크기로 조정
                if text_x2 <= text_x1 or text_y2 <= text_y1:
                    min_width = max(20, font_size * 2)
                    min_height = max(15, font_size)
                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = max(x1 + min_width, x2)
                    text_y2 = max(y1 + min_height, y2)
                
                # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
                bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
                if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                    draw.rectangle([x1, y1, x2, y2], fill=bg_color)
                
                # 폰트 로드 (2배 크기, 굵기 레벨 적용)
                bold_level = getattr(region, 'bold_level', 1 if getattr(region, 'bold', False) else 0)
                effective_font_size = font_size
                if bold_level >= 1:
                    effective_font_size = int(effective_font_size * 1.1)
                if bold_level >= 2:
                    effective_font_size = int(effective_font_size * 1.15)
                
                font = self._load_pil_font_with_bold(region.font_family, effective_font_size, bold_level)
                
                # 줄바꿈 계산 (2배 너비)
                box_width = max(10, text_x2 - text_x1)
                if margin < 0:
                    wrap_width = box_width - (margin * 2)
                else:
                    wrap_width = box_width
                
                # 텍스트 줄바꿈
                if region.wrap_mode == "word":
                    text_lines = self.jp_canvas.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                else:
                    text_lines = self.jp_canvas.wrap_text_for_box(region.text, wrap_width, font_size, font)
                
                # 줄간격 계산
                base_line_height = int(effective_font_size * 1.0)
                line_height = int(base_line_height * region.line_spacing)
                total_text_height = len(text_lines) * line_height
                
                # 텍스트가 박스를 넘치면 조정
                available_height = text_y2 - text_y1
                if total_text_height > available_height:
                    line_height = max(font_size, available_height // len(text_lines))
                    total_text_height = len(text_lines) * line_height
                    
                    if total_text_height > available_height:
                        scale_factor = available_height / total_text_height
                        font_size = max(8, int(font_size * scale_factor))
                        line_height = max(font_size, available_height // len(text_lines))
                        total_text_height = len(text_lines) * line_height
                        
                        # 폰트 다시 로드
                        font = self.jp_canvas.load_font_for_overlay(region.font_family, font_size)
                        if hasattr(region, 'bold') and region.bold:
                            bold_font_size = int(font_size * 1.1)
                            try:
                                font = self.jp_canvas.load_font_for_overlay(region.font_family, bold_font_size)
                            except:
                                pass
                        
                        # 줄바꿈 다시 계산
                        if region.wrap_mode == "word":
                            text_lines = self.jp_canvas.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, font)
                        else:
                            text_lines = self.jp_canvas.wrap_text_for_box(region.text, wrap_width, font_size, font)
                        
                        line_height = max(font_size, available_height // len(text_lines))
                        total_text_height = len(text_lines) * line_height
                
                # 텍스트 시작 위치 계산
                start_y = text_y1 + (available_height - total_text_height) // 2
                
                # 텍스트 색상 설정 (BGR → RGB)
                text_color = (region.color[2], region.color[1], region.color[0])
                
                # 각 줄의 텍스트 그리기
                for line_idx, line_text in enumerate(text_lines):
                    if line_text.strip():
                        try:
                            text_width = draw.textlength(line_text, font=font)
                        except Exception:
                            text_width = len(line_text) * font_size * 0.6
                        
                        # 텍스트 위치 계산 (정렬 적용)
                        text_align = getattr(region, 'text_align', 'center')
                        if text_align == "left":
                            text_x = text_x1
                        elif text_align == "right":
                            text_x = text_x2 - text_width
                        else:  # "center"
                            text_x = text_x1 + (text_x2 - text_x1 - text_width) // 2
                        text_y = start_y + line_idx * line_height
                        
                        tolerance = 20 * scale
                        if text_x >= text_x1 - tolerance and text_x + text_width <= text_x2 + tolerance and text_y <= text_y2 + tolerance:
                            if text_y + font_size <= text_y2 + tolerance:
                                # 테두리 적용
                                stroke_color = getattr(region, 'stroke_color', None)
                                stroke_width = getattr(region, 'stroke_width', 0)
                                if stroke_color is not None and stroke_width > 0:
                                    draw.text((text_x, text_y), line_text, font=font, fill=text_color,
                                             stroke_width=stroke_width, stroke_fill=stroke_color)
                                else:
                                    draw.text((text_x, text_y), line_text, font=font, fill=text_color)
            
            # 알파 블렌딩
            blended = Image.alpha_composite(base_img, text_layer)
            final_image = blended.convert("RGB")
            
            # 원본 크기로 다운스케일링 (고품질)
            final_image = final_image.resize((img_width, img_height), Image.LANCZOS)
            
            # 저장
            final_image.save(file_path, "PNG", quality=95)
            
        except Exception as e:
            logger.error(f"PIL 고해상도 저장 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

    def _load_pil_font_with_bold(self, font_family, font_size, bold_level):
        """
        굵기 레벨에 따라 Bold / ExtraBold 폰트를 우선적으로 로드하고,
        없으면 기존 load_font_for_overlay 결과를 사용.
        bold_level: 0=보통, 1=진하게, 2=더 진하게
        """
        try:
            # 폰트 후보 경로 매핑 (Windows 기본 폰트 기준)
            base_paths = []
            bold_paths = []
            extra_paths = []
            
            if font_family in ("나눔고딕", "NanumGothic"):
                local_appdata = os.environ.get("LOCALAPPDATA", r"C:/Users")
                base_paths = [
                    os.path.join(local_appdata, "Microsoft", "Windows", "Fonts", "NanumGothic.ttf"),
                    resource_path("fonts/NanumGothic.ttf"),
                ]                
                bold_paths = [
                    os.path.join(local_appdata, "Microsoft", "Windows", "Fonts", "NanumGothicBold.ttf"),
                ]
                # ExtraBold: 시스템 폴더 + 사용자 폴더(%LOCALAPPDATA%) 후보                
                extra_paths = [                    
                    os.path.join(local_appdata, "Microsoft", "Windows", "Fonts", "NanumGothicExtraBold.ttf"),
                ] + bold_paths  # ExtraBold 없으면 Bold로 폴백
            elif font_family in ("맑은 고딕", "Malgun Gothic"):
                base_paths = [
                    "C:/Windows/Fonts/malgun.ttf",
                    resource_path("fonts/malgun.ttf"),
                ]
                bold_paths = [
                    "C:/Windows/Fonts/malgunbd.ttf",
                ]
                extra_paths = bold_paths  # 별도 ExtraBold 없음
            elif font_family in ("굴림", "Gulim"):
                base_paths = [
                    "C:/Windows/Fonts/gulim.ttc",
                    resource_path("fonts/gulim.ttc"),
                ]
                bold_paths = [
                    "C:/Windows/Fonts/gulim.ttc",  # 굴림은 한 파일에 굵기 포함
                ]
                extra_paths = bold_paths
            elif font_family in ("Arial",):
                base_paths = [
                    "C:/Windows/Fonts/arial.ttf",
                ]
                bold_paths = [
                    "C:/Windows/Fonts/arialbd.ttf",
                ]
                extra_paths = bold_paths
            elif font_family in ("Times New Roman",):
                base_paths = [
                    "C:/Windows/Fonts/times.ttf",
                ]
                bold_paths = [
                    "C:/Windows/Fonts/timesbd.ttf",
                ]
                extra_paths = bold_paths
            
            # bold_level에 따라 우선순위 리스트 구성
            candidate_paths = []
            if bold_level >= 2:
                candidate_paths.extend(extra_paths)
            if bold_level >= 1:
                candidate_paths.extend(bold_paths)
            candidate_paths.extend(base_paths)
            
            from PIL import ImageFont as _PILFont
            
            for p in candidate_paths:
                if p and os.path.exists(p):
                    try:
                        return _PILFont.truetype(p, font_size)
                    except Exception:
                        continue
            
        except Exception:
            pass
        
        # 폰트 매핑에 실패하면 기존 로더로 폴백
        return self.jp_canvas.load_font_for_overlay(font_family, font_size)
    
    def save_with_qpainter(self, file_path):
        """QPainter를 사용하여 화면과 완전히 동일하게 저장"""
        try:
            # 타겟 이미지를 QPixmap으로 변환
            jp_height, jp_width = self.jp_image.shape[:2]
            jp_rgb = cv2.cvtColor(self.jp_image, cv2.COLOR_BGR2RGB)
            jp_qimage = QImage(jp_rgb.data, jp_width, jp_height, jp_width * 3, QImage.Format_RGB888)
            jp_pixmap = QtGui.QPixmap.fromImage(jp_qimage)
            
            # 화면 크기와 동일한 QImage 생성
            img = QImage(jp_pixmap.size(), QImage.Format_RGB888)
            
            # 고해상도 디스플레이 대응: 픽셀 비율 설정
            ratio = jp_pixmap.devicePixelRatio()
            img.setDevicePixelRatio(ratio)
            
            painter = QPainter(img)
            
            # AA, 힌팅 모두 OFF → 화면과 완전히 같은 픽셀 그리기
            painter.setRenderHints(QPainter.RenderHint(0))
            
            # 배경 이미지 그리기
            painter.drawPixmap(0, 0, jp_pixmap)
            
            # 현재 이미지의 텍스트 박스만 저장 (성능 최적화)
            current_filename = os.path.basename(self.jp_image_path) if self.jp_image_path else None
            current_text_regions = []
            for region in self.text_regions:
                if hasattr(region, 'image_filename') and region.image_filename == current_filename:
                    current_text_regions.append(region)
            
            # 텍스트 박스들 그대로 그림 (화면 렌더링과 동일한 방식)
            for region in current_text_regions:
                # visible 속성 확인 (기본값 True)
                if not getattr(region, 'visible', True):
                    continue  # 숨김 처리된 텍스트 박스는 저장하지 않음
                
                if not region.is_positioned or not region.target_bbox:
                    continue
                
                x1, y1, x2, y2 = region.target_bbox
                
                # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
                bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
                if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                    painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3]))
                
                # 폰트 설정 (화면과 동일한 계산)
                box_height = y2 - y1
                font_size = max(8, min(int(box_height * 0.6), int(region.font_size)))
                
                # 화면과 동일한 bold 처리 (폰트 크기 조정)
                if hasattr(region, 'bold') and region.bold:
                    font_size = int(font_size * 1.1)  # 10% 크게
                
                font = QFont(region.font_family, font_size)
                font.setPixelSize(font_size)
                # 폰트 굵기 설정 (사용자 선택에 따라)
                if hasattr(region, 'bold') and region.bold:
                    font.setBold(True)
                    font.setWeight(QFont.Bold)
                else:
                    font.setBold(False)
                    font.setWeight(QFont.Normal)
                painter.setFont(font)
                
                # 텍스트 색상 설정 (BGR → RGB)
                text_color = QColor(region.color[2], region.color[1], region.color[0])
                # 펜 굵기를 조정하여 텍스트를 더 진하게 표시
                pen = QPen(text_color)
                pen.setWidth(1)  # 펜 굵기 설정
                painter.setPen(pen)
                
                # 여백 계산 (사용자 설정 여백 사용, 음수 허용)
                box_width = x2 - x1
                box_height = y2 - y1
                margin = region.margin
                text_x1 = x1 + margin
                text_y1 = y1 + margin
                text_x2 = x2 - margin
                text_y2 = y2 - margin
                
                # 텍스트 영역이 너무 작으면 최소 크기로 조정
                if text_x2 <= text_x1 or text_y2 <= text_y1:
                    min_width = max(20, font_size * 2)
                    min_height = max(15, font_size)
                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = max(x1 + min_width, x2)
                    text_y2 = max(y1 + min_height, y2)
                
                # 줄바꿈 계산 (음수 여백 고려)
                box_width = max(10, text_x2 - text_x1)  # 최소 너비 보장
                # 음수 여백일 때는 텍스트가 박스를 넘어갈 수 있도록 허용
                if margin < 0:
                    wrap_width = box_width - (margin * 2)  # 음수 여백만큼 더 넓게
                else:
                    wrap_width = box_width  # 정상 여백일 때는 박스 크기 그대로
                
                # 텍스트 줄바꿈 처리
                try:
                    # PIL의 줄바꿈 함수를 사용하여 동일한 결과 얻기
                    from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
                    temp_img = PILImage.new('RGB', (100, 100), (255, 255, 255))
                    temp_draw = PILImageDraw.Draw(temp_img)
                    temp_font = PILImageFont.truetype(resource_path("fonts/NanumGothic.ttf"), font_size) if os.path.exists(resource_path("fonts/NanumGothic.ttf")) else PILImageFont.load_default()
                    
                    if region.wrap_mode == "word":
                        text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, temp_font)
                    else:
                        text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, temp_font)
                except Exception:
                    text_lines = [region.text]
                
                # 줄간격 계산 (화면과 동일, 폰트가 안 잘리도록 20% 여유 증가)
                base_line_height = int(font_size * 1.0)
                line_height = int(base_line_height * region.line_spacing)
                total_height = len(text_lines) * line_height
                
                # 화면과 동일한 폰트 크기 동적 조정 로직
                available_height = text_y2 - text_y1
                if total_height > available_height:
                    scale_factor = available_height / total_height
                    font_size = max(8, int(font_size * scale_factor))
                    line_height = max(font_size, available_height // len(text_lines))
                    total_height = len(text_lines) * line_height
                    
                    # 폰트 크기 변경 후 폰트 다시 로드 (화면과 동일한 bold 처리)
                    if hasattr(region, 'bold') and region.bold:
                        # 화면과 동일: 폰트 크기를 10% 크게
                        bold_font_size = int(font_size * 1.1)
                        font = QFont(region.font_family, bold_font_size)
                        font.setPixelSize(bold_font_size)
                        font.setBold(True)
                        font.setWeight(QFont.Bold)
                    else:
                        font = QFont(region.font_family, font_size)
                        font.setPixelSize(font_size)
                        font.setBold(False)
                        font.setWeight(QFont.Normal)
                    painter.setFont(font)
                    
                    # 줄바꿈 다시 계산 (새로운 폰트 크기로)
                    try:
                        from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
                        temp_img = PILImage.new('RGB', (100, 100), (255, 255, 255))
                        temp_draw = PILImageDraw.Draw(temp_img)
                        temp_font = PILImageFont.truetype(resource_path("fonts/NanumGothic.ttf"), font_size) if os.path.exists(resource_path("fonts/NanumGothic.ttf")) else PILImageFont.load_default()
                        
                        if region.wrap_mode == "word":
                            text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, font_size, temp_font)
                        else:
                            text_lines = self.wrap_text_for_box(region.text, wrap_width, font_size, temp_font)
                    except Exception:
                        pass  # 줄바꿈 재계산 실패 시 기존 텍스트 사용
                    
                    # 줄 수가 변경되었으므로 높이 재계산
                    line_height = max(font_size, available_height // len(text_lines))
                    total_height = len(text_lines) * line_height
                
                start_y = text_y1 + (text_y2 - text_y1 - total_height) // 2
                
                # 각 줄의 텍스트 그리기
                for line_idx, line_text in enumerate(text_lines):
                    if line_text.strip():
                        # 텍스트 너비 계산
                        text_metrics = painter.fontMetrics()
                        line_width = text_metrics.width(line_text)
                        
                        # 텍스트 위치 계산 (정렬 적용)
                        text_align = getattr(region, 'text_align', 'center')
                        if text_align == "left":
                            line_x = text_x1
                        elif text_align == "right":
                            line_x = text_x2 - line_width
                        else:  # "center"
                            line_x = text_x1 + (text_x2 - text_x1 - line_width) // 2
                        line_y = start_y + line_idx * line_height + font_size
                        
                        # 텍스트가 박스 범위 내에 있는지 확인
                        if line_y <= text_y2:
                            # 테두리 적용
                            stroke_color = getattr(region, 'stroke_color', None)
                            stroke_width = getattr(region, 'stroke_width', 0)
                            if stroke_color is not None and stroke_width > 0:
                                # QPainterPath를 사용하여 stroke 구현
                                path = QPainterPath()
                                path.addText(line_x, line_y, font, line_text)
                                # 테두리 그리기
                                stroke_qcolor = QColor(stroke_color[0], stroke_color[1], stroke_color[2])
                                stroke_pen = QPen(stroke_qcolor)
                                stroke_pen.setWidth(stroke_width)
                                stroke_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                                stroke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                                painter.strokePath(path, stroke_pen)
                                # 텍스트 그리기
                                painter.fillPath(path, text_color)
                            else:
                                painter.drawText(line_x, line_y, line_text)
            
            painter.end()
            img.save(file_path, "PNG")
            
        except Exception as e:
            logger.error(f"QPainter 저장 오류: {e}")
            raise e
    
    def create_overlay_image(self):
        """텍스트 오버레이가 적용된 이미지 생성"""
        # 타겟 이미지 복사
        result_image = self.jp_image.copy()
        
        # PIL 이미지로 안전한 변환 (텍스트 렌더링을 위해)
        try:
            pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
        except Exception as e:
            return None
        
        # 각 텍스트 영역에 대해 텍스트 삽입
        for region in self.text_regions:
            if not region.is_positioned or not region.target_bbox:
                continue  # 위치가 설정되지 않은 텍스트는 건너뛰기
            
            x1, y1, x2, y2 = region.target_bbox
            
            # bbox 경계 클램핑 (이미지 범위 내로 제한)
            img_height, img_width = result_image.shape[:2]
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))
            
            # 사용자 설정 폰트 로드 (화면과 동일한 크기로)
            try:
                # 화면 렌더링과 동일한 폰트 크기 계산 적용
                box_height = y2 - y1
                # 사용자 설정 폰트 크기를 우선 사용하되, 박스가 너무 작으면 조정
                user_font_size = int(region.font_size)
                max_font_size = int(box_height * 0.6)
                font_size = max(8, min(max_font_size, user_font_size))
                font = self.load_font_for_overlay(region.font_family, font_size)
            except Exception as e:
                logger.error(f"폰트 로드 실패: {e}")
                font_size = region.font_size
                font = ImageFont.load_default()
            
            # 텍스트 색상 (BGR → RGB)
            text_color = (region.color[2], region.color[1], region.color[0])
            
            # 설정된 여백 사용 (음수 허용)
            margin = region.margin
            text_rect = (x1 + margin, y1 + margin, 
                        x2 - margin, y2 - margin)
            
            if text_rect[2] - text_rect[0] <= 0 or text_rect[3] - text_rect[1] <= 0:
                continue  # 유효하지 않은 텍스트 영역 건너뛰기
            
            # 배경 박스 그리기 (배경색이 설정되어 있고 투명하지 않은 경우만)
            bg_color = getattr(region, 'bg_color', (255, 255, 255, 255))
            if bg_color is not None and len(bg_color) >= 4 and bg_color[3] > 0:
                padding = 1  # 패딩을 5에서 1로 줄여서 더 타이트하게
                bg_x1 = max(0, x1 - padding)
                bg_y1 = max(0, y1 - padding)
                bg_x2 = min(pil_image.width, x2 + padding)
                bg_y2 = min(pil_image.height, y2 + padding)
                
                # 배경색 적용 (RGBA)
                overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
                
                # 배경을 원본 이미지에 합성
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(pil_image)
            else:
                # 투명 배경인 경우 draw만 업데이트
                draw = ImageDraw.Draw(pil_image)
            
            # 텍스트를 여러 줄로 분할 (자동 줄바꿈) - 화면과 동일한 처리
            try:
                box_width = max(10, text_rect[2] - text_rect[0])  # 최소 너비 보장
                # 줄바꿈 계산용 너비 (좌우 20px 허용)
                wrap_width = box_width + 40  # 좌우 각각 20px씩 추가
                if region.wrap_mode == "word":
                    text_lines = self.wrap_text_for_overlay_safe_word(region.text, wrap_width, region.font_size, font)
                else:  # "char" 기본값
                    text_lines = self.wrap_text_for_box(region.text, wrap_width, region.font_size, font)
            except Exception as e:
                text_lines = [region.text]  # 오류 시 원본 텍스트 사용
            
            # 각 줄의 텍스트 그리기 (안전한 줄간격 처리)
            # 줄간격 처리 (사용자 설정 적용)
            # 화면 렌더링과 동일한 폰트 크기 및 줄간격 계산 사용
            # 폰트가 안 잘리도록 20% 여유 증가
            base_line_height = int(font_size * 1.0)
            line_height = int(base_line_height * region.line_spacing)
            total_height = len(text_lines) * line_height
            start_y = text_rect[1] + (text_rect[3] - text_rect[1] - total_height) // 2
            
            for line_idx, line_text in enumerate(text_lines):
                if line_text.strip():
                    # 텍스트 크기 계산 (안전한 textlength 사용)
                    try:
                        text_width = max(1, draw.textlength(line_text, font=font))
                    except Exception:
                        # textlength가 지원되지 않는 경우 대체 방법
                        text_width = len(line_text) * region.font_size // 2
                    
                    # 텍스트 위치 계산 (중앙 정렬, 하단 잘림 방지)
                    text_x = text_rect[0] + (text_rect[2] - text_rect[0] - text_width) // 2
                    # textbbox를 사용하여 정확한 텍스트 높이 계산 (모음 잘림 방지)
                    try:
                        bbox = draw.textbbox((0, 0), line_text, font=font)
                        text_height = bbox[3] - bbox[1]
                        # 박스 중앙에서 텍스트 높이의 절반만큼 위로 조정
                        text_y = start_y + line_idx * line_height + (line_height - text_height) // 2
                    except Exception:
                        # textbbox 실패 시 기본 계산
                        text_y = start_y + line_idx * line_height
                    
                    # 텍스트가 박스 범위를 벗어나지 않도록 확인 (20px 허용)
                    tolerance = 20
                    if text_y + font_size > text_rect[3] + tolerance:
                        continue  # 박스를 벗어나면 해당 줄 건너뛰기
                    
                    # 텍스트 그리기 (고해상도 렌더링)
                    try:
                        # 고해상도 렌더링으로 텍스트 품질 향상
                        scale = 2
                        try:
                            hires_font = ImageFont.truetype(resource_path("fonts/NanumGothic.ttf"), font_size * scale)
                        except Exception:
                            try:
                                hires_font = ImageFont.truetype("C:/Booxen/BooxenEBook/reader/fonts/epub/NanumGothic.ttf", font_size * scale)
                            except Exception:
                                hires_font = font  # 폰트 로딩 실패 시 기본 폰트 사용
                        
                        # 텍스트 크기 재계산 (폰트에 맞게)
                        try:
                            text_width = max(1, draw.textlength(line_text, font=font))
                            text_height = font_size
                        except Exception:
                            text_width = len(line_text) * font_size * 0.6
                            text_height = font_size
                        
                        # 텍스트 색상 설정 (BGR → RGB)
                        text_color = (region.color[2], region.color[1], region.color[0])
                        
                        # 고해상도 레이어 생성 (여백 추가)
                        padding = 4  # 여백 추가
                        hires_width = int(text_width * scale) + padding * 2
                        hires_height = int(text_height * scale) + padding * 2
                        hires_layer = Image.new("RGBA", (hires_width, hires_height), (255, 255, 255, 0))
                        hires_draw = ImageDraw.Draw(hires_layer)
                        
                        # 고해상도로 텍스트 렌더링 (여백 고려)
                        # 테두리 적용
                        stroke_color = getattr(region, 'stroke_color', None)
                        stroke_width = getattr(region, 'stroke_width', 0)
                        if stroke_color is not None and stroke_width > 0:
                            # stroke_width를 스케일에 맞게 조정
                            scaled_stroke_width = int(stroke_width * scale)
                            hires_draw.text((padding, padding), line_text, font=hires_font, 
                                           fill=(text_color[0], text_color[1], text_color[2], 255),
                                           stroke_width=scaled_stroke_width, stroke_fill=stroke_color)
                        else:
                            hires_draw.text((padding, padding), line_text, font=hires_font, fill=(text_color[0], text_color[1], text_color[2], 255))
                        
                        # 원본 크기로 다운스케일링 (LANCZOS 필터 사용)
                        hires_layer = hires_layer.resize((int(text_width), int(text_height)), Image.LANCZOS)
                        
                        # 원본 위치에 합성 (20px 허용 범위 내에서)
                        paste_x = max(text_rect[0] - tolerance, min(int(text_x), text_rect[2] - int(text_width) + tolerance))
                        paste_y = max(text_rect[1] - tolerance, min(int(text_y), text_rect[3] - int(text_height) + tolerance))
                        pil_image.paste(hires_layer, (paste_x, paste_y), hires_layer)
                    except Exception as e:
                        logger.error(f"고해상도 텍스트 그리기 실패: {e}")
                        # 대체 방법으로 텍스트 그리기 (20px 허용 범위 확인)
                        if text_x >= text_rect[0] - tolerance and text_x + text_width <= text_rect[2] + tolerance and text_y + font_size <= text_rect[3] + tolerance:
                            # 테두리 적용
                            stroke_color = getattr(region, 'stroke_color', None)
                            stroke_width = getattr(region, 'stroke_width', 0)
                            if stroke_color is not None and stroke_width > 0:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color,
                                         stroke_width=stroke_width, stroke_fill=stroke_color)
                            else:
                                draw.text((text_x, text_y), line_text, font=font, fill=text_color)
        
        # PIL 이미지를 OpenCV 형식으로 변환
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return result_image
    
    
    def load_font_for_overlay(self, font_family, font_size):
        """오버레이용 폰트 로드 (create_overlay_image에서 사용)"""
        # 사용자 추가 폰트 확인 (우선순위)
        if hasattr(self, 'custom_fonts') and font_family in self.custom_fonts:
            custom_font_path = self.custom_fonts[font_family]
            if os.path.exists(custom_font_path):
                try:
                    font = ImageFont.truetype(custom_font_path, font_size)
                    return font
                except Exception as e:
                    logger.error(f"사용자 추가 폰트 로딩 실패: {custom_font_path}, 오류: {e}")
                    # 실패 시 기본 폰트로 폴백
        
        # 사용자 설정 폰트가 시스템 폰트 목록에 있는지 확인
        system_fonts = ["Arial", "Times New Roman", "Courier New", "굴림", "맑은 고딕", "나눔고딕"]
        
        if font_family in system_fonts:
            font_paths = {
                "Arial": ["fonts/arial.ttf", "C:/Windows/Fonts/arial.ttf"],
                "Times New Roman": ["fonts/times.ttf", "C:/Windows/Fonts/times.ttf"],
                "Courier New": ["fonts/cour.ttf", "C:/Windows/Fonts/cour.ttf"],
                "굴림": [resource_path("fonts/gulim.ttc"), "C:/Windows/Fonts/gulim.ttc", "C:/Windows/Fonts/NGULIM.TTF"],
                "맑은 고딕": [resource_path("fonts/malgun.ttf"), "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgunsl.ttf"],
                "나눔고딕": [resource_path("fonts/NanumGothic.ttf"), "C:/Booxen/BooxenEBook/reader/fonts/epub/NanumGothic.ttf", "C:/Windows/Fonts/NanumGothic.ttf"]
            }
            
            if font_family in font_paths:
                for font_path in font_paths[font_family]:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(resource_path(font_path), font_size)
                            return font
                        except Exception as e:
                            logger.error(f"폰트 로딩 실패: {font_path}, 오류: {e}")
                            continue
        
        # 기본 한글 폰트들 시도
        default_font_paths = [
            resource_path("fonts/NanumGothic.ttf"),
            resource_path("fonts/malgun.ttf"),
            resource_path("fonts/gulim.ttc"),
            "C:/Windows/Fonts/NanumGothic.ttf",
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            "C:/Windows/Fonts/batang.ttc",
            "C:/Windows/Fonts/dotum.ttc",
        ]
        
        for font_path in default_font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(resource_path(font_path), font_size)
                    return font
                except Exception as e:
                    logger.error(f"기본 폰트 로딩 실패: {font_path}, 오류: {e}")
                    continue
        
        # 모든 시도가 실패하면 기본 폰트 사용
        logger.error("모든 폰트 로딩 실패, 기본 폰트 사용")
        return ImageFont.load_default()
    
    def wrap_text_for_overlay_safe(self, text, max_width, font_size, font_path="fonts/NanumGothic.ttf"):
        """PIL 충돌 없는 안전한 줄바꿈 (글자 단위, textbbox 미사용, textlength만 사용)"""
        try:
            if not text or not text.strip():
                return [""]

            max_width = max(20, int(max_width))
            font_size = max(6, int(font_size))

            # ⚠️ Dummy Image (항상 새로 생성)
            dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
            draw = ImageDraw.Draw(dummy_img)

            try:
                font = ImageFont.truetype(resource_path(font_path), font_size)
            except Exception:
                font = ImageFont.load_default()

            # 폭 계산 전용 (글자 단위 안전)
            lines = []
            current_line = ""
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = ""
                    continue

                test_line = current_line + char
                width = draw.textlength(test_line, font=font)
                if width > max_width and current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            del draw  # ⚠️ Pillow 객체 명시 해제
            return lines

        except Exception as e:
            logger.error(f"wrap_text_for_overlay_safe 오류: {e}")
            return [text]
    
    def on_table_selection_changed(self):
        """테이블 선택 변경 시"""
        current_row = self.text_table.currentRow()
        if current_row >= 0 and current_row < len(self.text_regions):
            region = self.text_regions[current_row]
            
            # 폰트 크기 동기화
            self.font_size_spin.blockSignals(True)
            self.font_size_slider.blockSignals(True)
            self.font_size_spin.setValue(region.font_size)
            self.font_size_slider.setValue(region.font_size)
            self.font_size_spin.blockSignals(False)
            self.font_size_slider.blockSignals(False)
            
            # 색상 버튼 동기화
            color = region.color
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            self.color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color_hex};
                    color: {'white' if sum(color) < 384 else 'black'};
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    width: 30px;
                    height: 25px;
                }}
            """)
            
            # 타겟 이미지 미리보기 업데이트
            if hasattr(self, 'jp_canvas'):
                # 현재 이미지의 텍스트 박스만 표시
                self.update_display_for_current_image()

    def wrap_text_for_overlay_safe_word(self, text, max_width, font_size, font):
        """PIL 충돌 없는 안전한 단어 단위 줄바꿈 (띄어쓰기 단위, 줄바꿈 문자 지원)"""
        try:
            if not text or not text.strip():
                return [""]

            max_width = max(20, int(max_width))
            font_size = max(6, int(font_size))

            # ⚠️ Dummy Image (항상 새로 생성)
            dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
            draw = ImageDraw.Draw(dummy_img)

            # 전달받은 폰트 사용
            if font is None:
                font = ImageFont.load_default()

            # 먼저 줄바꿈 문자로 분할 (사용자가 엔터키로 입력한 줄바꿈 보존)
            paragraphs = text.split('\n')
            lines = []
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    # 빈 줄은 빈 문자열로 추가
                    lines.append("")
                    continue
                
                # 각 단락을 띄어쓰기 단위로 단어 분할
                words = paragraph.split()
                current_line = ""
                
                for word in words:
                    # 현재 줄에 단어를 추가했을 때의 너비 계산
                    test_line = current_line + (" " if current_line else "") + word
                    try:
                        width = draw.textlength(test_line, font=font)
                    except Exception:
                        # textlength 실패 시 문자 수 기반 추정
                        width = len(test_line) * font_size * 0.6
                    
                    if width <= max_width:
                        current_line = test_line
                    else:
                        # 현재 줄이 너무 길면 새 줄로 이동
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # 단어 자체가 너무 긴 경우 강제로 줄바꿈
                            lines.append(word)
                            current_line = ""
                
                # 단락의 마지막 줄 추가
                if current_line:
                    lines.append(current_line)

            del draw  # ⚠️ Pillow 객체 명시 해제
            return lines if lines else [text]

        except Exception as e:
            logger.error(f"wrap_text_for_overlay_safe_word 오류: {e}")
            return [text]


def main():
    """
    Main entry point for the application
    애플리케이션의 메인 진입점
    
    Initializes the Qt application and shows the main window.
    Qt 애플리케이션을 초기화하고 메인 윈도우를 표시합니다.
    """
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # 애플리케이션 폰트 설정 (나눔고딕 등록)
    try:
        font_id = QFontDatabase.addApplicationFont(resource_path("fonts/NanumGothic.ttf"))
        if font_id != -1:
            font_name = QFontDatabase.applicationFontFamilies(font_id)[0]
            app.setFont(QtGui.QFont(font_name, 9))
        else:
            app.setFont(QtGui.QFont("맑은 고딕", 9))
            logger.warning("나눔고딕 폰트 등록 실패, 맑은 고딕 사용")
    except Exception as e:
        app.setFont(QtGui.QFont("맑은 고딕", 9))
        logger.error(f"애플리케이션 폰트 설정 오류: {e}")
    
    # 애플리케이션 정보 설정
    app.setApplicationName("텍스트 오버레이 툴 (클라우드 비전 OCR)")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("TextOverlayTool")
    
    # 메인 윈도우 생성 및 표시
    try:
        window = TextOverlayTool()
        window.show()
    except Exception as e:
        logger.error(f"메인 윈도우 생성 오류: {e}")
        sys.exit(1)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

