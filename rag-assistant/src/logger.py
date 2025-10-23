import os
import threading
import traceback
from datetime import datetime, timezone, timedelta


class SessionLogger:
    """Minimal logger that separates system logs and chat transcripts."""

    def __init__(self):
        print("Welcome to RAG assistant...")
        print("Please hold on...")
        now = datetime.now()
        self.timestamp = now.strftime("%Y%m%d_%H%M_%S%f")[:-3]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(base_dir, "logs")
        self.chat_dir = os.path.join(base_dir, "chats")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chat_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, f"{self.timestamp}_log.txt")
        self.chat_file = os.path.join(self.chat_dir, f"{self.timestamp}_chat.txt")

        self._lock = threading.Lock()
        self.IST = timezone(timedelta(hours=5, minutes=30))

        self.log("Logger initialized.", level="INFO")

    def _timestamp(self):
        now = datetime.now(self.IST)
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write(self, filepath, text):
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def log(self, message, level="INFO"):
        timestamp = self._timestamp()
        line = f"[{timestamp}] [{level.upper()}] {message}"
        print(line)
        self._write(self.log_file, line)

    def log_chat(self, user_message=None, assistant_message=None):
        """Write user and assistant messages to chat transcript."""
        timestamp = self._timestamp()
        with self._lock:
            with open(self.chat_file, "a", encoding="utf-8") as f:
                if user_message:
                    f.write(f"\n[{timestamp}] USER:\n{user_message.strip()}\n")
                if assistant_message:
                    f.write(f"\n[{timestamp}] ASSISTANT:\n{assistant_message.strip()}\n")

    def log_exception(self, e: Exception):
        tb_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        msg = f"[{self._timestamp()}] [EXCEPTION] {e}\n{tb_text}"
        print(msg)
        self._write(self.log_file, msg)


LOGGER = SessionLogger()