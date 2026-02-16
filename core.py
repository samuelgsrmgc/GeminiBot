import json
import os
import google.generativeai as genai
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class GeminiChat:
    def __init__(self, gemini_token: str, chat_history: list = None):
        self.chat_history = chat_history
        genai.configure(api_key=gemini_token)
        with open("./safety_settings.json", "r") as fp:
            self.safety_settings = json.load(fp)
        logging.info("Initiated new chat model")

    def _get_model(self, generative_model: str = "gemini-flash-latest"):
        model_name = os.getenv("GEMINI_MODEL", generative_model)
        logging.info(f"Trying to get generative model: {model_name}")
        try:
            return genai.GenerativeModel(model_name, safety_settings=self.safety_settings)
        except Exception as e:
            logging.error(f"Failed to get model: {e}")
            raise

    def start_chat(self, image=None) -> None:
        model_name = "gemini-pro-vision" if image else "gemini-flash-latest"
        model = self._get_model(model_name)
        
        history = []
        if image:
            history.append({'role': 'user', 'parts': [image]})

        if self.chat_history:
            history.extend(self.chat_history)
            
        lang = os.getenv("LANGUAGE", "en")
        self.chat = model.start_chat(history=history)
        self.chat.send_message(f"You are a helpful assistant with a female persona. Please respond in {lang} language. Please use Telegram-compatible markdown. For example, use *bold* for bold text, _italic_ for italic, and `code` for code blocks. Do not use markdown features that are not supported by Telegram, such as headers or horizontal rules.")
        logging.info("Start new conversation")

    def send_message(self, message_text: str, image=None) -> str:
        try:
            if image:
                response = self.chat.send_message([message_text, image])
            else:
                response = self.chat.send_message(message_text)
            return response.text
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            return "Couldn't reach out to Google Gemini. Try Again..."

    def get_chat_title(self) -> str:
        return self.send_message("Write a one-line short title up to 10 words for this conversation in plain text.")

    def get_chat_history(self):
        return self.chat.history

    def close(self) -> None:
        logging.info("Closed model instance")
        self.chat = None
        self.chat_history = []