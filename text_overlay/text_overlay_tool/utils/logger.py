"""
Logger module for application logging
애플리케이션 로깅을 위한 로거 모듈
"""

import logging


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
        # 로그 포맷 설정 / Log format configuration
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러 설정 (오류 및 경고만 저장)
        # File handler configuration (only errors and warnings saved)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.WARNING)  # WARNING 이상만 저장 / Only WARNING and above
        
        # 로거 설정 / Logger configuration
        self.logger = logging.getLogger('TextOverlayTool')
        self.logger.setLevel(logging.WARNING)  # WARNING 이상만 처리 / Only WARNING and above
        self.logger.addHandler(file_handler)
        # 콘솔 핸들러 제거 (배포용) / Console handler removed (for deployment)
    
    def info(self, message):
        """
        Log info message / 정보 로그 기록
        Args / 인자:
            message (str): Info message / 정보 메시지
        """
        self.logger.info(message)
    
    def debug(self, message):
        """
        Log debug message / 디버그 로그 기록
        Args / 인자:
            message (str): Debug message / 디버그 메시지
        """
        self.logger.debug(message)
    
    def warning(self, message):
        """
        Log warning message / 경고 로그 기록
        Args / 인자:
            message (str): Warning message / 경고 메시지
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        Log error message / 에러 로그 기록
        Args / 인자:
            message (str): Error message / 에러 메시지
        """
        self.logger.error(message)


# 전역 로거 인스턴스 / Global logger instance
logger = Logger()

