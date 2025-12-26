"""
Text region model classes
텍스트 영역 모델 클래스들
"""

from PyQt5 import QtWidgets


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
        """
        Initialize text region / 텍스트 영역 초기화
        
        Args / 인자:
            text (str): Text content / 텍스트 내용
            bbox (tuple): Bounding box (x1, y1, x2, y2) / 바운딩 박스
            font_size (int): Font size / 폰트 크기
            color (tuple): Text color (B, G, R) - Default: (0, 0, 0) = 검은색 (Black)
                          / 텍스트 색상 (B, G, R) - 기본값: (0, 0, 0) = 검은색
            font_family (str): Font family name / 폰트 패밀리 이름
            margin (int): Margin in pixels / 픽셀 단위 여백
            wrap_mode (str): Wrap mode ("char" or "word") / 줄바꿈 모드
            line_spacing (float): Line spacing multiplier / 줄간격 배율
            bold (bool): Bold text / 볼드 텍스트
            text_align (str): Text alignment ("left", "center", "right") / 텍스트 정렬
            bg_color (tuple): Background color (R, G, B, A) - Default: (255, 255, 255, 255) = 흰색 (White)
                             / 배경색 (R, G, B, A) - 기본값: (255, 255, 255, 255) = 흰색
        """
        self.text = text
        self.bbox = bbox if bbox is not None else (0, 0, 0, 0)  # (x1, y1, x2, y2)
        self.font_size = font_size
        self.font_family = font_family
        self.margin = margin  # 상하좌우 여백 / Margin in pixels
        self.color = color  # BGR format - Default: (0, 0, 0) = 검은색 / BGR format - Default: (0, 0, 0) = Black
        self.wrap_mode = wrap_mode  # "char" 또는 "word" - 줄바꿈 모드 / Wrap mode
        self.bold = bold  # 볼드 설정 / Bold setting
        # 폰트 굵기 레벨: 0=보통, 1=진하게, 2=더 진하게
        # Font weight level: 0=normal, 1=bold, 2=extra bold
        self.bold_level = 1 if bold else 0
        self.line_spacing = line_spacing  # 줄간격 배율 / Line spacing multiplier
        self.text_align = text_align  # 텍스트 정렬 / Text alignment
        # 배경색: (R, G, B, A) 형식, None이면 기본값 흰색 (255, 255, 255, 255) 사용
        # Background color: (R, G, B, A) format, if None then default white (255, 255, 255, 255)
        self.bg_color = bg_color if bg_color is not None else (255, 255, 255, 255)  # 기본값: 흰색 / Default: White
        # 텍스트 테두리: stroke_color는 (R, G, B) 형식, stroke_width는 픽셀 단위
        # Text border: stroke_color is (R, G, B) format, stroke_width is in pixels
        self.stroke_color = None  # None이면 테두리 없음 / None means no border
        self.stroke_width = 0  # 0이면 테두리 없음 / 0 means no border
        self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        self.target_bbox = None  # 타겟 이미지에서의 타겟 위치 / Target position on target image
        self.is_positioned = False  # 위치가 설정되었는지 여부 / Whether position is set
        self.image_filename = None  # 해당 텍스트 박스가 속한 이미지 파일명 / Image filename this text box belongs to
        self.is_manual = False  # 수동으로 추가된 텍스트인지 여부 / Whether manually added text
        self.visible = True  # 텍스트 박스 표시 여부 (기본값: 표시) / Text box visibility (default: visible)


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

