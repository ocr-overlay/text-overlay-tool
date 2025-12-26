"""
Main entry point for Text Overlay Tool
텍스트 오버레이 툴 메인 진입점

This module contains the main application window and entry point.
이 모듈은 메인 애플리케이션 윈도우와 진입점을 포함합니다.
"""

import sys
import os

# Add the parent directory to sys.path so we can import the package
# 부모 디렉토리를 sys.path에 추가하여 패키지를 import할 수 있도록 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFontDatabase

# Import modules / 모듈 가져오기
from text_overlay_tool.utils import logger, resource_path
from text_overlay_tool.ocr import CloudVisionOCR, CLOUD_VISION_AVAILABLE

# Import TextOverlayTool from original file for now
# 현재는 원본 파일에서 TextOverlayTool을 가져옴
# TODO: Complete migration of TextOverlayTool class to this package
# TODO: TextOverlayTool 클래스를 이 패키지로 완전히 마이그레이션 필요

# For development: import from original file
# 개발용: 원본 파일에서 import
import importlib.util
vision_file_path = os.path.join(os.path.dirname(__file__), "text_overlay_tool_vision.py")
if os.path.exists(vision_file_path):
    spec = importlib.util.spec_from_file_location("text_overlay_tool_vision", vision_file_path)
    vision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision_module)
    TextOverlayTool = vision_module.TextOverlayTool
else:
    raise ImportError(f"원본 파일을 찾을 수 없습니다: {vision_file_path}")


def main():
    """
    Main entry point for the application
    애플리케이션의 메인 진입점
    
    Initializes the Qt application and shows the main window.
    Qt 애플리케이션을 초기화하고 메인 윈도우를 표시합니다.
    """
    app = QtWidgets.QApplication(sys.argv)
    
    # 애플리케이션 폰트 설정 (나눔고딕 등록)
    # Application font setup (register Nanum Gothic)
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
    # Application information setup
    app.setApplicationName("텍스트 오버레이 툴 (클라우드 비전 OCR)")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("TextOverlayTool")
    
    # 메인 윈도우 생성 및 표시
    # Create and show main window
    try:
        window = TextOverlayTool()
        window.show()
    except Exception as e:
        logger.error(f"메인 윈도우 생성 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

