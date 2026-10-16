import os 
import certifi 
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from pydantic import Field



class Settings(BaseSettings):
    llm_provider: Literal['openai', 'gemini'] = 'openai'
    gemini_api_key: str = ''
    gemini_chat_model: str = 'gemini-2.5-flash'
    gemini_vision_model: str = 'gemini-2.5-flash'
    gemini_embed_model: str = 'gemini-embedding-001'
    gemini_embed_dimensions: int = Field(default=1536, ge=128, le=3072)
    openai_embed_dimensions: int = Field(default=1536, gt=0)
    gemini_transcribe_model: str = 'gemini-2.5-flash'
    gemini_tts_model: str = 'gemini-2.5-pro-preview-tts'
    gemini_tts_voice: str = 'Kore'
    langfuse_enabled: bool = False
    langfuse_public_key: str = ''
    langfuse_secret_key: str = ''
    langfuse_base_url: str = 'https://cloud.langfuse.com'
    openai_api_key: str = ''
    pinecone_api_key: str = ''
    tavily_api_key: str = ''
    pinecone_index: str = 'medical-agent-demo'
    openai_chat_model: str = 'gpt-4.1-mini'
    openai_embed_model: str = 'text-embedding-3-small'
    openai_vision_model: str = 'gpt-4.1-mini'
    api_token: str = ''
    allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider == 'gemini' else self.openai_api_key

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == 'gemini':
            return 'https://generativelanguage.googleapis.com/v1beta/openai/'
        return 'https://api.openai.com/v1/'

    @property
    def chat_model(self) -> str:
        return self.gemini_chat_model if self.llm_provider == 'gemini' else self.openai_chat_model

    @property
    def vision_model(self) -> str:
        return self.gemini_vision_model if self.llm_provider == 'gemini' else self.openai_vision_model

    @property
    def embed_model(self) -> str:
        return self.gemini_embed_model if self.llm_provider == 'gemini' else self.openai_embed_model

    @property
    def embed_dimensions(self) -> int:
        return self.gemini_embed_dimensions if self.llm_provider == 'gemini' else self.openai_embed_dimensions

    @property
    def active_pinecone_index(self) -> str:
        return f'{self.pinecone_index}-gemini' if self.llm_provider == 'gemini' else self.pinecone_index

    @property
    def embedding_namespace(self) -> str:
        return f'gemini-{self.embed_model}-{self.embed_dimensions}' if self.llm_provider == 'gemini' else ''

    @property
    def missing_chat_settings(self) -> list[str]:
        missing = []
        if not self.llm_api_key:
            missing.append('GEMINI_API_KEY' if self.llm_provider == 'gemini' else 'OPENAI_API_KEY')
        if not self.pinecone_api_key:
            missing.append('PINECONE_API_KEY')
        return missing

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)



settings = Settings()