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

---
## Requirements

- Python 3.8+
- PyQt5
- OpenCV (cv2)
- Pillow
- NumPy
- google-cloud-vision (optional, required for OCR)

실행을 위해 다음 환경이 필요합니다.

- Python 3.8 이상
- PyQt5
- OpenCV (cv2)
- Pillow
- NumPy
- google-cloud-vision (OCR 사용 시 필요)

---

## Google Cloud Vision Setup

1. Create a Google Cloud project
2. Enable **Cloud Vision API**
3. Create a service account and download the JSON key file
4. Set the credential file path inside the application UI

Google Cloud Vision OCR을 사용하려면 다음 과정을 거쳐야 합니다.

1. Google Cloud 프로젝트 생성
2. **Cloud Vision API** 활성화
3. 서비스 계정 생성 후 JSON 키 파일 다운로드
4. 애플리케이션 UI에서 키 파일 경로 설정

> ⚠ The service account key file must **not** be committed to this repository.  
> ⚠ 서비스 계정 키 파일은 절대 GitHub에 커밋하지 마세요.

---

## How to Run

Run the application **from the project root directory**:

```bash
python main.py


## Why Google Cloud Vision?

This project prioritizes **practical usability and executable distribution**.

While local OCR engines (e.g. PaddleOCR) were evaluated, they introduced
significant instability when packaged as executables.

Google Cloud Vision was chosen as the default OCR backend due to:

- Stable runtime behavior
- Smaller executable size
- Fewer platform-specific issues
- Better end-user accessibility

이 프로젝트는 **실제 사용성과 exe 배포 안정성**을 최우선으로 고려합니다.

PaddleOCR 등 로컬 OCR 엔진도 검토했지만,
exe 패키징 과정에서 잦은 오류와 불안정성이 발생했습니다.

Google Cloud Vision을 기본 OCR 엔진으로 선택한 이유는 다음과 같습니다.

- 안정적인 실행 환경
- 비교적 작은 실행 파일 크기
- 플랫폼 의존성 문제 감소
- 사용자 접근성 향상

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

본 프로젝트는 **MIT 라이선스**로 배포됩니다.  
자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## Disclaimer

This tool is intended for personal and educational use.  
Users are responsible for ensuring that their usage complies with
copyright laws and content policies in their region.

본 도구는 개인 및 학습 목적을 위해 제공됩니다.  
사용자는 각 지역의 저작권 및 관련 법규를 준수할 책임이 있습니다.
본 도구는 개인 및 학습 목적을 위해 제공됩니다.
사용자는 각 지역의 저작권 및 관련 법규를 준수할 책임이 있습니다.
