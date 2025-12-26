"""
Resource path utility module
리소스 경로 유틸리티 모듈
"""

import os
import sys


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

