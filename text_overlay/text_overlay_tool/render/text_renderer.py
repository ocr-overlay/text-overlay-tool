"""
Text rendering utilities for text overlay tool
텍스트 오버레이 툴 텍스트 렌더링 유틸리티

This module contains functions for text wrapping and font loading.
이 모듈은 텍스트 줄바꿈 및 폰트 로딩 함수를 포함합니다.
"""

import os
from PIL import Image, ImageDraw, ImageFont

from ..utils import logger, resource_path


def _is_korean(char):
    """
    Check if character is Korean
    한글 문자인지 확인
    
    Args / 인자:
        char (str): Character to check / 확인할 문자
        
    Returns / 반환값:
        bool: True if Korean, False otherwise / 한글이면 True, 아니면 False
    """
    return '\uAC00' <= char <= '\uD7AF' or '\u1100' <= char <= '\u11FF' or '\u3130' <= char <= '\u318F'


def wrap_text_for_box(text, max_width, font_size, font):
    """
    Wrap text to fit within box width (supports Korean)
    텍스트 박스에 맞는 줄바꿈 (한글 지원)
    
    Args / 인자:
        text (str): Text to wrap / 줄바꿈할 텍스트
        max_width (int): Maximum width in pixels / 최대 너비 (픽셀)
        font_size (int): Font size / 폰트 크기
        font: PIL ImageFont object / PIL ImageFont 객체
        
    Returns / 반환값:
        list[str]: List of wrapped text lines / 줄바꿈된 텍스트 라인 목록
    """
    try:
        if not text or not text.strip():
            return [""]
        
        # 한글과 영문을 구분하여 처리 / Process Korean and English separately
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
            
            # 한글, 영문, 숫자, 특수문자 구분 / Distinguish Korean, English, numbers, special characters
            if _is_korean(char):
                # 한글은 문자 단위로 처리 / Process Korean character by character
                test_line = current_line + char
            elif char.isspace():
                # 공백은 단어 경계로 처리 / Treat space as word boundary
                test_line = current_line + char
            else:
                # 영문/숫자는 단어 단위로 처리 / Process English/numbers word by word
                word = ""
                while i < len(text) and not text[i].isspace() and not _is_korean(text[i]):
                    word += text[i]
                    i += 1
                i -= 1  # 다음 루프에서 올바른 위치에서 시작 / Start at correct position in next loop
                test_line = current_line + word
            
            # 텍스트 너비 측정 / Measure text width
            try:
                temp_img = Image.new("L", (max_width * 2, font_size * 2), color=0)
                temp_draw = ImageDraw.Draw(temp_img)
                width = temp_draw.textlength(test_line, font=font)
                
                if width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = char if _is_korean(char) else word
                    else:
                        # 단어/문자가 너무 긴 경우 강제로 줄바꿈 / Force line break if word/char is too long
                        lines.append(char if _is_korean(char) else word)
                        current_line = ""
            except Exception:
                # textlength 실패 시 문자 수 기반 추정 / Estimate based on character count if textlength fails
                estimated_width = len(test_line) * font_size * 0.6
                if estimated_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = char if _is_korean(char) else word
                    else:
                        lines.append(char if _is_korean(char) else word)
                        current_line = ""
            
            i += 1
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
        
    except Exception as e:
        logger.error(f"줄바꿈 처리 오류: {e}")
        return [text]


def wrap_text_for_overlay_safe_word(text, max_width, font_size, font):
    """
    Safe word-based text wrapping for overlay (space-based, supports line breaks)
    PIL 충돌 없는 안전한 단어 단위 줄바꿈 (띄어쓰기 단위, 줄바꿈 문자 지원)
    
    Args / 인자:
        text (str): Text to wrap / 줄바꿈할 텍스트
        max_width (int): Maximum width in pixels / 최대 너비 (픽셀)
        font_size (int): Font size / 폰트 크기
        font: PIL ImageFont object / PIL ImageFont 객체
        
    Returns / 반환값:
        list[str]: List of wrapped text lines / 줄바꿈된 텍스트 라인 목록
    """
    try:
        if not text or not text.strip():
            return [""]

        max_width = max(20, int(max_width))
        font_size = max(6, int(font_size))

        # Dummy Image (always create new) / 더미 이미지 (항상 새로 생성)
        dummy_img = Image.new("L", (max_width * 2, font_size * 3), color=0)
        draw = ImageDraw.Draw(dummy_img)

        # Use provided font / 전달받은 폰트 사용
        if font is None:
            font = ImageFont.load_default()

        # First split by line breaks (preserve user-entered line breaks)
        # 먼저 줄바꿈 문자로 분할 (사용자가 엔터키로 입력한 줄바꿈 보존)
        paragraphs = text.split('\n')
        lines = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                # 빈 줄은 빈 문자열로 추가 / Add empty string for empty line
                lines.append("")
                continue
            
            # 각 단락을 띄어쓰기 단위로 단어 분할 / Split paragraph into words by space
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                # 현재 줄에 단어를 추가했을 때의 너비 계산 / Calculate width when adding word to current line
                test_line = current_line + (" " if current_line else "") + word
                try:
                    width = draw.textlength(test_line, font=font)
                except Exception:
                    # textlength 실패 시 문자 수 기반 추정 / Estimate based on character count if textlength fails
                    width = len(test_line) * font_size * 0.6
                
                if width <= max_width:
                    current_line = test_line
                else:
                    # 현재 줄이 너무 길면 새 줄로 이동 / Move to new line if current line is too long
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # 단어 자체가 너무 긴 경우 강제로 줄바꿈 / Force line break if word itself is too long
                        lines.append(word)
                        current_line = ""
            
            # 단락의 마지막 줄 추가 / Add last line of paragraph
            if current_line:
                lines.append(current_line)

        del draw  # Explicitly release Pillow object / Pillow 객체 명시 해제
        return lines if lines else [text]

    except Exception as e:
        logger.error(f"wrap_text_for_overlay_safe_word 오류: {e}")
        return [text]


def load_font_for_overlay(font_family, font_size, custom_fonts=None):
    """
    Load font for overlay rendering
    오버레이 렌더링용 폰트 로드
    
    Args / 인자:
        font_family (str): Font family name / 폰트 패밀리 이름
        font_size (int): Font size / 폰트 크기
        custom_fonts (dict): Custom fonts dictionary {name: path} / 사용자 정의 폰트 딕셔너리
        
    Returns / 반환값:
        ImageFont: PIL ImageFont object / PIL ImageFont 객체
    """
    # 사용자 추가 폰트 확인 (우선순위) / Check custom fonts first (priority)
    if custom_fonts and font_family in custom_fonts:
        custom_font_path = custom_fonts[font_family]
        if os.path.exists(custom_font_path):
            try:
                font = ImageFont.truetype(custom_font_path, font_size)
                return font
            except Exception as e:
                logger.error(f"사용자 추가 폰트 로딩 실패: {custom_font_path}, 오류: {e}")
                # 실패 시 기본 폰트로 폴백 / Fallback to default font on failure
    
    # 사용자 설정 폰트가 시스템 폰트 목록에 있는지 확인 / Check if font is in system font list
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
    
    # 기본 한글 폰트들 시도 / Try default Korean fonts
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
    
    # 모든 시도가 실패하면 기본 폰트 사용 / Use default font if all attempts fail
    logger.error("모든 폰트 로딩 실패, 기본 폰트 사용")
    return ImageFont.load_default()


# TextRenderer class for backward compatibility / 하위 호환성을 위한 TextRenderer 클래스
class TextRenderer:
    """
    Text renderer class (wrapper for utility functions)
    텍스트 렌더러 클래스 (유틸리티 함수 래퍼)
    """
    
    @staticmethod
    def wrap_text_for_box(text, max_width, font_size, font):
        """Wrapper for wrap_text_for_box function / wrap_text_for_box 함수 래퍼"""
        return wrap_text_for_box(text, max_width, font_size, font)
    
    @staticmethod
    def wrap_text_for_overlay_safe_word(text, max_width, font_size, font):
        """Wrapper for wrap_text_for_overlay_safe_word function / wrap_text_for_overlay_safe_word 함수 래퍼"""
        return wrap_text_for_overlay_safe_word(text, max_width, font_size, font)
    
    @staticmethod
    def load_font_for_overlay(font_family, font_size, custom_fonts=None):
        """Wrapper for load_font_for_overlay function / load_font_for_overlay 함수 래퍼"""
        return load_font_for_overlay(font_family, font_size, custom_fonts)

