from functools import lru_cache
import math
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config import settings

@lru_cache()
def clients():
    if settings.missing_chat_settings:
        raise ValueError('Set in .env: '+', '.join(settings.missing_chat_settings))
    if settings.llm_provider == 'gemini':
        from google import genai

        ai = genai.Client(api_key=settings.gemini_api_key, http_options={'timeout': 45000})
    else:
        ai = OpenAI(api_key=settings.openai_api_key)
    return ai, Pinecone(api_key=settings.pinecone_api_key)


def ensure_index():
    _, pc = clients()
    name = settings.active_pinecone_index
    if not pc.has_index(name):
        pc.create_index(name=name, dimension=settings.embed_dimensions, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    elif pc.describe_index(name).dimension != settings.embed_dimensions:
        raise ValueError(f'Pinecone index {name} has incompatible dimensions. Use a new PINECONE_INDEX and re-ingest your PDFs.')
    return pc.Index(name)


def embed(texts, task_type='RETRIEVAL_DOCUMENT'):
    if not texts:
        return []
    ai, _ = clients()
    if settings.llm_provider == 'gemini':
        config = {'output_dimensionality': settings.embed_dimensions}
        if settings.embed_model == 'gemini-embedding-001':
            contents = texts
            config['task_type'] = task_type
        else:
            from google.genai import types

            contents = [types.Content(parts=[types.Part.from_text(
                text=f'task: search result | query: {text}' if task_type == 'RETRIEVAL_QUERY' else f'title: none | text: {text}'
            )]) for text in texts]
        result = ai.models.embed_content(model=settings.embed_model, contents=contents, config=config)
        vectors = [embedding.values for embedding in result.embeddings or []]
        normalized = []
        for vector in vectors:
            magnitude = math.hypot(*vector)
            if not magnitude or not math.isfinite(magnitude):
                raise ValueError('Invalid embedding vector returned')
            normalized.append([value / magnitude for value in vector])
        vectors = normalized
    else:
        result = ai.embeddings.create(model=settings.embed_model, input=texts, dimensions=settings.embed_dimensions)
        vectors = [item.embedding for item in sorted(result.data, key=lambda item: item.index)]
    if len(vectors) != len(texts) or any(len(vector) != settings.embed_dimensions for vector in vectors):
        raise ValueError('Embedding response does not match the requested batch size or dimensions.')
    return vectors



def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query], task_type='RETRIEVAL_QUERY')[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=settings.embedding_namespace)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]
