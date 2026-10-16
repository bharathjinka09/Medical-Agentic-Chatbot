import os
import io
import wave
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fastapi.testclient import TestClient
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from app.config import Settings
from app import knowledge, tracing


test_connection = sqlite3.connect(':memory:', check_same_thread=False)
with patch('sqlite3.connect', return_value=test_connection):
    from app import agent, main
import ingest


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.settings = Settings(
            _env_file=None,
            openai_api_key='test-openai',
            pinecone_api_key='test-pinecone',
        )
        for module in (agent, main, tracing, knowledge, ingest):
            replacement = patch.object(module, 'settings', self.settings)
            replacement.start()
            self.addCleanup(replacement.stop)
        tracing.get_tracing_client.cache_clear()
        self.addCleanup(tracing.get_tracing_client.cache_clear)
        knowledge.clients.cache_clear()
        self.addCleanup(knowledge.clients.cache_clear)

    def use_gemini(self):
        self.settings.llm_provider = 'gemini'
        self.settings.gemini_api_key = 'test-gemini'
        self.settings.openai_api_key = ''

    def enable_tracing(self):
        self.settings.langfuse_enabled = True
        self.settings.langfuse_public_key = 'pk-lf-test'
        self.settings.langfuse_secret_key = 'sk-lf-test'

    def test_openai_defaults(self):
        self.assertEqual(self.settings.llm_provider, 'openai')
        self.assertEqual(self.settings.llm_api_key, 'test-openai')
        self.assertEqual(self.settings.chat_model, self.settings.openai_chat_model)
        self.assertEqual(self.settings.vision_model, self.settings.openai_vision_model)
        self.assertEqual(self.settings.llm_base_url, 'https://api.openai.com/v1/')
        self.assertEqual(self.settings.missing_chat_settings, [])

    def test_gemini_configuration(self):
        self.use_gemini()
        self.assertEqual(self.settings.llm_api_key, 'test-gemini')
        self.assertEqual(self.settings.chat_model, 'gemini-2.5-flash')
        self.assertEqual(self.settings.vision_model, 'gemini-2.5-flash')
        self.assertEqual(self.settings.llm_base_url, 'https://generativelanguage.googleapis.com/v1beta/openai/')
        self.assertEqual(self.settings.missing_chat_settings, [])

    def test_invalid_provider_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, llm_provider='unknown')

    def test_missing_gemini_key_returns_503(self):
        self.settings.llm_provider = 'gemini'
        with TestClient(main.app) as client, patch.object(main.graph, 'invoke') as invoke:
            response = client.post('/api/chat', json={'message': 'Hello'})
            self.assertEqual(response.status_code, 503)
            self.assertIn('GEMINI_API_KEY', response.json()['detail'])
            invoke.assert_not_called()
            self.assertFalse(client.get('/api/health').json()['configured'])

    def test_gemini_does_not_require_openai_key(self):
        self.use_gemini()
        self.settings.openai_api_key = ''
        self.assertEqual(self.settings.missing_chat_settings, [])

    def test_gemini_embeddings_use_retrieval_tasks(self):
        self.use_gemini()
        self.settings.openai_api_key = ''
        vector = [3.0, 4.0] + [0.0] * (self.settings.embed_dimensions - 2)
        client = Mock()
        client.models.embed_content.return_value.embeddings = [SimpleNamespace(values=vector)]
        with patch.object(knowledge, 'clients', return_value=(client, Mock())):
            for task in ('RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'):
                result = knowledge.embed(['medical text'], task_type=task)
                request = client.models.embed_content.call_args.kwargs
                self.assertEqual(request['model'], 'gemini-embedding-001')
                self.assertEqual(request['config']['task_type'], task)
                self.assertEqual(request['config']['output_dimensionality'], 1536)
                self.assertEqual(result[0][:2], [0.6, 0.8])

    def test_provider_indexes_are_separate(self):
        original_index = self.settings.active_pinecone_index
        self.assertEqual(self.settings.embedding_namespace, '')
        self.use_gemini()
        self.assertNotEqual(self.settings.active_pinecone_index, original_index)
        self.assertIn('gemini-embedding-001', self.settings.embedding_namespace)

    def test_explicit_gemini_index_is_not_suffixed_twice(self):
        self.use_gemini()
        self.settings.pinecone_index = 'medical-agent-demo-gemini'
        self.assertEqual(self.settings.active_pinecone_index, 'medical-agent-demo-gemini')
        client = Mock()
        client.describe_index.return_value.dimension = self.settings.embed_dimensions
        with patch.object(knowledge, 'clients', return_value=(Mock(), client)):
            knowledge.ensure_index()
        client.Index.assert_called_once_with('medical-agent-demo-gemini')
        client.create_index.assert_not_called()

    def test_index_creation_uses_selected_dimensions(self):
        self.use_gemini()
        client = Mock()
        client.has_index.return_value = False
        with patch.object(knowledge, 'clients', return_value=(Mock(), client)):
            knowledge.ensure_index()
        self.assertEqual(client.create_index.call_args.kwargs['dimension'], 1536)
        self.assertEqual(client.create_index.call_args.kwargs['name'], self.settings.active_pinecone_index)

    def test_incompatible_index_dimensions_are_rejected(self):
        client = Mock()
        client.describe_index.return_value.dimension = 3072
        with patch.object(knowledge, 'clients', return_value=(Mock(), client)):
            with self.assertRaisesRegex(ValueError, 'incompatible dimensions'):
                knowledge.ensure_index()

    def test_gemini_client_never_initializes_openai(self):
        self.use_gemini()
        with patch('google.genai.Client') as gemini, patch.object(knowledge, 'OpenAI') as openai, patch.object(knowledge, 'Pinecone'):
            knowledge.clients()
            self.assertEqual(gemini.call_args.kwargs['api_key'], 'test-gemini')
            openai.assert_not_called()

    def test_openai_embeddings_keep_input_order(self):
        client = Mock()
        first = [1.0] * self.settings.embed_dimensions
        second = [2.0] * self.settings.embed_dimensions
        client.embeddings.create.return_value.data = [
            SimpleNamespace(index=1, embedding=second), SimpleNamespace(index=0, embedding=first)
        ]
        with patch.object(knowledge, 'clients', return_value=(client, Mock())):
            self.assertEqual(knowledge.embed(['first', 'second']), [first, second])
        self.assertEqual(client.embeddings.create.call_args.kwargs['model'], 'text-embedding-3-small')

    def test_gemini_embedding_2_preserves_separate_documents(self):
        self.use_gemini()
        self.settings.gemini_embed_model = 'gemini-embedding-2'
        client = Mock()
        vector = [1.0] + [0.0] * (self.settings.embed_dimensions - 1)
        client.models.embed_content.return_value.embeddings = [SimpleNamespace(values=vector)] * 2
        with patch.object(knowledge, 'clients', return_value=(client, Mock())):
            self.assertEqual(len(knowledge.embed(['first', 'second'])), 2)
            request = client.models.embed_content.call_args.kwargs
            self.assertNotIn('task_type', request['config'])
            self.assertEqual(len(request['contents']), 2)
            self.assertEqual(request['contents'][0].parts[0].text, 'title: none | text: first')
            knowledge.embed(['first', 'second'], task_type='RETRIEVAL_QUERY')
            self.assertEqual(client.models.embed_content.call_args.kwargs['contents'][0].parts[0].text, 'task: search result | query: first')

    def test_ingestion_uses_gemini_namespace_for_all_batches(self):
        self.use_gemini()
        index = Mock()
        document = MagicMock()
        document.__enter__.return_value = [SimpleNamespace(get_text=lambda: 'medical text')]
        with patch('sys.argv', ['ingest.py', '--pdf', 'test.pdf']), patch.object(ingest.Path, 'exists', return_value=True), patch.object(ingest, 'ensure_index', return_value=index), patch.object(ingest.fitz, 'open', return_value=document), patch.object(ingest, 'chunks', return_value=['medical text'] * 55), patch.object(ingest, 'embed', side_effect=lambda texts: [[1.0]] * len(texts)), patch('builtins.print'):
            ingest.main()
        self.assertEqual(index.upsert.call_count, 2)
        for call in index.upsert.call_args_list:
            self.assertEqual(call.kwargs['namespace'], self.settings.embedding_namespace)
        self.assertEqual(len(index.upsert.call_args_list[0].kwargs['vectors']), 50)
        self.assertEqual(len(index.upsert.call_args_list[1].kwargs['vectors']), 5)

    def test_retrieval_uses_query_task_and_gemini_namespace(self):
        self.use_gemini()
        index = Mock()
        index.query.return_value.matches = []
        with patch.object(knowledge, 'ensure_index', return_value=index), patch.object(knowledge, 'embed', return_value=[[1.0]]) as embed:
            self.assertEqual(knowledge.search_kb('Question'), [])
            embed.assert_called_once_with(['Question'], task_type='RETRIEVAL_QUERY')
        self.assertEqual(index.query.call_args.kwargs['namespace'], self.settings.embedding_namespace)

    def test_gemini_rejects_incomplete_embedding_batch(self):
        self.use_gemini()
        client = Mock()
        vector = [1.0] + [0.0] * (self.settings.embed_dimensions - 1)
        client.models.embed_content.return_value.embeddings = [SimpleNamespace(values=vector)]
        with patch.object(knowledge, 'clients', return_value=(client, Mock())):
            with self.assertRaisesRegex(ValueError, 'batch size or dimensions'):
                knowledge.embed(['first', 'second'])

    def test_answer_selects_provider_and_forwards_callbacks(self):
        state = {'question': 'Hello', 'history': [], 'kb': [], 'web': []}
        config = {'callbacks': [BaseCallbackHandler()]}
        for provider in ('openai', 'gemini'):
            with self.subTest(provider=provider):
                self.settings.llm_provider = provider
                self.settings.gemini_api_key = 'test-gemini'
                with patch.object(agent, 'ChatOpenAI') as model:
                    model.return_value.invoke.return_value.content = 'Educational answer'
                    self.assertEqual(agent.answer(state, config), {'answer': 'Educational answer'})
                    self.assertEqual(model.call_args.kwargs['api_key'], self.settings.llm_api_key)
                    self.assertEqual(model.call_args.kwargs['base_url'], self.settings.llm_base_url)
                    self.assertEqual(model.call_args.kwargs['model'], self.settings.chat_model)
                    self.assertIs(model.return_value.invoke.call_args.kwargs['config'], config)

    def test_graph_propagates_generation_callbacks(self):
        class CaptureGenerations(BaseCallbackHandler):
            def __init__(self):
                self.generations = 0

            def on_llm_end(self, response, **kwargs):
                self.generations += 1

        callback = CaptureGenerations()
        config = {'configurable': {'thread_id': 'callback-test'}, 'callbacks': [callback]}
        state = {'question': 'Hello', 'history': [], 'image_description': '', 'kb': [], 'web': [], 'answer': '', 'route': ''}
        with patch.object(agent, 'search_kb', return_value=[]), patch.object(
            agent, 'ChatOpenAI', return_value=FakeListChatModel(responses=['Educational answer'])
        ):
            result = agent.graph.invoke(state, config=config)
        self.assertEqual(result['answer'], 'Educational answer')
        self.assertEqual(callback.generations, 1)

    def test_gemini_image_uses_selected_endpoint(self):
        self.use_gemini()
        image = 'data:image/png;base64,aGVsbG8='
        with patch.object(main, 'OpenAI') as client:
            client.return_value.chat.completions.create.return_value.choices = [
                SimpleNamespace(message=SimpleNamespace(content='Visible features'))
            ]
            self.assertEqual(main.inspect_image(image), 'Visible features')
            self.assertEqual(client.call_args.kwargs['api_key'], 'test-gemini')
            self.assertEqual(client.call_args.kwargs['base_url'], self.settings.llm_base_url)
            request = client.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request['model'], self.settings.gemini_vision_model)
            self.assertEqual(request['messages'][0]['content'][1]['image_url']['url'], image)

    def test_gemini_transcription_without_openai(self):
        self.use_gemini()
        with TestClient(main.app) as client, patch.object(main.genai, 'Client') as sdk, patch.object(main, 'OpenAI') as openai:
            generate = AsyncMock(return_value=SimpleNamespace(text='My symptoms'))
            sdk.return_value.aio.__aenter__.return_value.models.generate_content = generate
            response = client.post('/api/transcribe', files={'file': ('voice.webm', b'audio', 'audio/webm;codecs=opus')})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'text': 'My symptoms'})
            request = generate.call_args.kwargs
            self.assertEqual(request['model'], self.settings.gemini_transcribe_model)
            self.assertEqual(request['contents'][0].inline_data.mime_type, 'audio/webm')
            self.assertEqual(request['contents'][0].inline_data.data, b'audio')
            openai.assert_not_called()

    def test_gemini_speech_returns_playable_wav(self):
        self.use_gemini()
        audio = SimpleNamespace(mime_type='audio/L16;codec=pcm;rate=24000', data=b'\x00\x00' * 20)
        result = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(inline_data=audio)]))])
        with TestClient(main.app) as client, patch.object(main.genai, 'Client') as sdk, patch.object(main, 'OpenAI') as openai:
            generate = sdk.return_value.__enter__.return_value.models.generate_content
            generate.return_value = result
            response = client.post('/api/speak', json={'text': 'Educational answer'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['content-type'], 'audio/wav')
            with wave.open(io.BytesIO(response.content), 'rb') as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.readframes(20), audio.data)
            self.assertEqual(generate.call_args.kwargs['model'], self.settings.gemini_tts_model)
            openai.assert_not_called()

    def test_gemini_audio_failure_never_falls_back_to_openai(self):
        self.use_gemini()
        with TestClient(main.app) as client, patch.object(main.genai, 'Client', side_effect=RuntimeError('unavailable')), patch.object(main, 'OpenAI') as openai:
            self.assertEqual(client.post('/api/speak', json={'text': 'Hello'}).status_code, 502)
            self.assertEqual(client.post('/api/transcribe', files={'file': ('voice.webm', b'audio', 'audio/webm')}).status_code, 502)
            openai.assert_not_called()

    def test_openai_audio_is_preserved(self):
        with TestClient(main.app) as client, patch.object(main, 'OpenAI') as sdk, patch.object(main.genai, 'Client') as gemini:
            sdk.return_value.audio.transcriptions.create.return_value.text = 'Transcript'
            sdk.return_value.audio.speech.create.return_value.content = b'mp3-audio'
            response = client.post('/api/transcribe', files={'file': ('voice.webm', b'audio', 'audio/webm')})
            self.assertEqual(response.json(), {'text': 'Transcript'})
            response = client.post('/api/speak', json={'text': 'Hello'})
            self.assertEqual(response.content, b'mp3-audio')
            self.assertEqual(response.headers['content-type'], 'audio/mpeg')
            gemini.assert_not_called()

    def test_audio_requires_only_selected_key(self):
        self.settings.llm_provider = 'gemini'
        with TestClient(main.app) as client:
            self.assertEqual(client.post('/api/speak', json={'text': 'Hello'}).status_code, 503)
            self.assertEqual(client.post('/api/transcribe', files={'file': ('voice.webm', b'audio', 'audio/webm')}).status_code, 503)

    def test_tracing_disabled_does_not_initialize_sdk(self):
        with patch('langfuse.Langfuse') as sdk:
            self.assertEqual(tracing.graph_config('thread'), {'configurable': {'thread_id': 'thread'}})
            sdk.assert_not_called()

    def test_missing_tracing_keys_does_not_block_chat(self):
        self.settings.langfuse_enabled = True
        with patch('langfuse.Langfuse') as sdk:
            self.assertNotIn('callbacks', tracing.graph_config('thread'))
            sdk.assert_not_called()

    def test_tracing_session_and_shutdown(self):
        self.enable_tracing()
        with patch('langfuse.Langfuse') as sdk, patch('langfuse.langchain.CallbackHandler') as handler:
            first = tracing.graph_config('session-one')
            second = tracing.graph_config('session-two')
            sdk.assert_called_once_with(
                public_key='pk-lf-test', secret_key='sk-lf-test', base_url='https://cloud.langfuse.com'
            )
            self.assertEqual(handler.call_count, 2)
            self.assertEqual(first['metadata']['langfuse_session_id'], 'session-one')
            self.assertEqual(second['metadata']['langfuse_session_id'], 'session-two')
            self.assertEqual(first['callbacks'], [handler.return_value])
            tracing.shutdown_tracing()
            sdk.return_value.shutdown.assert_called_once()

    def test_tracing_failure_does_not_block_chat(self):
        self.enable_tracing()
        with patch('langfuse.Langfuse', side_effect=RuntimeError('unavailable')):
            self.assertNotIn('callbacks', tracing.graph_config('thread'))

    def test_chat_reuses_conversation_for_tracing(self):
        self.use_gemini()
        result = {'answer': 'Educational answer', 'kb': [], 'web': [], 'route': 'answer'}
        config = {'configurable': {'thread_id': 'existing-session'}}
        with TestClient(main.app) as client, patch.object(main, 'graph_config', return_value=config) as configure, patch.object(main.graph, 'invoke', return_value=result) as invoke:
            response = client.post('/api/chat', json={'message': 'Hello'}, headers={'X-Conversation-ID': 'existing-session'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['conversation_id'], 'existing-session')
            configure.assert_called_once_with('existing-session')
            self.assertIs(invoke.call_args.kwargs['config'], config)


if __name__ == '__main__':
    unittest.main()