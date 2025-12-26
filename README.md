# text-overlay-tool
A practical desktop tool for overlaying OCR text onto images using Google Cloud Vision. 만화 식질을 위한 텍스트 오버레이 데스크톱 툴

# Text Overlay Tool

A practical desktop tool for overlaying OCR text onto images using Google Cloud Vision.  
Google Cloud Vision OCR을 사용해 이미지 위에 텍스트를 오버레이하는 실용적인 데스크톱 툴입니다.

만화 식질을 위한 텍스트 오버레이 데스크톱 툴

---

## Overview

**Text Overlay Tool** is a desktop application designed to assist comic translation
and image text replacement workflows.

It extracts text from source images using OCR and allows users to interactively
place, resize, and render translated text onto target images with fine-grained
control over layout and typography.

**Text Overlay Tool**은 만화 번역(식질) 및 이미지 내 텍스트 교체 작업을 돕기 위해
제작된 데스크톱 애플리케이션입니다.

OCR을 통해 소스 이미지에서 텍스트를 추출하고, 번역된 텍스트를
타겟 이미지 위에 직접 배치·이동·크기 조절하며
레이아웃과 타이포그래피를 세밀하게 조정할 수 있습니다.

---

## Key Features

- Google Cloud Vision OCR for text extraction
- Interactive text box placement and resizing
- Optimized Korean text rendering (safe line wrapping)
- Support for Korean, Japanese, and English text
- Custom fonts, font size, alignment, spacing, and margins
- Desktop GUI built with PyQt5
- Designed with executable (exe) distribution in mind

주요 기능은 다음과 같습니다.

- Google Cloud Vision OCR을 이용한 텍스트 추출
- 텍스트 박스의 직접 배치 및 크기 조절
- 한글 렌더링에 최적화된 안전한 줄바꿈 처리
- 한국어, 일본어, 영어 텍스트 지원
- 사용자 정의 폰트, 폰트 크기, 정렬, 줄간격, 여백 설정
- PyQt5 기반 데스크톱 GUI
- exe 배포를 고려한 구조 설계

---

## Project Status

⚠ **This project is currently in the process of refactoring.**

This repository is a modular reorganization of a previously monolithic,
production-used tool.

- Core application logic and UI are currently implemented in  
  `text_overlay_tool_vision.py`
- The package structure under `text_overlay_tool/` represents the ongoing
  migration toward a cleaner modular architecture
- Temporary imports from the legacy file are intentional and documented

⚠ **이 프로젝트는 현재 리팩터링 진행 중입니다.**

본 저장소는 실제로 사용되던 단일 파일(monolithic) 기반 도구를
점진적으로 모듈화하는 과정의 결과물입니다.

- 핵심 애플리케이션 로직과 UI는 현재  
  `text_overlay_tool_vision.py`에 구현되어 있습니다.
- `text_overlay_tool/` 패키지 구조는
  점진적인 구조 개선을 위한 마이그레이션 단계입니다.
- 레거시 파일을 임시로 import하는 방식은 의도된 설계입니다.

---

## Directory Structure

```text
text-overlay-tool/
 ├─ main.py                         # Application entry point
 ├─ text_overlay_tool_vision.py     # Legacy core implementation (Vision OCR)
 ├─ text_overlay_tool/              # Modularized package (in progress)
 │   ├─ utils/
 │   │   ├─ logger.py
 │   │   └─ resource.py
 │   ├─ ocr/
 │   │   └─ cloud_vision.py
 │   ├─ models/
 │   │   └─ text_region.py
 │   ├─ render/
 │   │   └─ text_renderer.py
 │   └─ ui/
 │       └─ image_canvas.py
 ├─ fonts/
 │   └─ NanumGothic.ttf
 ├─ README.md
 ├─ .gitignore
 └─ LICENSE





본 도구는 개인 및 학습 목적을 위해 제공됩니다.
사용자는 각 지역의 저작권 및 관련 법규를 준수할 책임이 있습니다.
