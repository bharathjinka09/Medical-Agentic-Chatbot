import logging
from functools import lru_cache

from langchain_core.runnables import RunnableConfig

from app.config import settings


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tracing_client():
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_configured:
        logger.warning('Langfuse disabled: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.')
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
    except Exception:
        logger.warning('Langfuse initialization failed; continuing without tracing.')
        return None


def graph_config(thread_id: str) -> RunnableConfig:
    config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
    if get_tracing_client() is not None:
        try:
            from langfuse.langchain import CallbackHandler

            config.update(
                callbacks=[CallbackHandler(public_key=settings.langfuse_public_key)],
                run_name='medical-chat',
                metadata={
                    'langfuse_session_id': thread_id,
                    'llm_provider': settings.llm_provider,
                },
            )
        except Exception:
            logger.warning('Langfuse callback unavailable; continuing without tracing.')
    return config


def shutdown_tracing():
    if get_tracing_client.cache_info().currsize:
        client = get_tracing_client()
        if client is not None:
            try:
                client.shutdown()
            except Exception:
                logger.warning('Langfuse shutdown failed.')
        get_tracing_client.cache_clear()