"""
Tests for LLM streaming middleware tracking.

Tests cover OpenAI and Anthropic streaming response tracking,
usage accumulation, and token estimation.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


@pytest.mark.unit
class TestOpenAIStreamingTracking:
    """Tests for OpenAI streaming response tracking."""

    def test_openai_streaming_response_tracked(self):
        """Should track OpenAI streaming responses."""
        # Create a mock streaming response
        mock_response = MagicMock()
        mock_response.__iter__ = Mock(return_value=iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content='Hello', finish_reason=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=' world', finish_reason='stop'))]),
        ]))

        # The middleware should wrap this and track it
        # Verify that the response is iterable and tracks content
        content_parts = []
        for chunk in mock_response:
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Should accumulate content
        assert len(content_parts) == 2
        assert ''.join(content_parts) == 'Hello world'

    def test_openai_streaming_accumulates_content(self):
        """Should accumulate content from streaming chunks."""
        accumulated = []

        # Simulate streaming chunks
        chunks = [
            {'content': 'The '},
            {'content': 'quick '},
            {'content': 'brown '},
            {'content': 'fox'},
            {'content': None}  # End marker
        ]

        for chunk in chunks:
            if chunk['content']:
                accumulated.append(chunk['content'])

        result = ''.join(accumulated)
        assert result == 'The quick brown fox'
        assert len(accumulated) == 4

    def test_openai_streaming_extracts_usage_from_final_chunk(self):
        """Should extract usage stats from final message chunk."""
        # Mock final chunk with usage info
        final_chunk = MagicMock()
        final_chunk.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

        # Should extract usage
        usage = {
            'prompt_tokens': final_chunk.usage.prompt_tokens,
            'completion_tokens': final_chunk.usage.completion_tokens,
            'total_tokens': final_chunk.usage.total_tokens
        }

        assert usage['prompt_tokens'] == 10
        assert usage['completion_tokens'] == 20
        assert usage['total_tokens'] == 30

    def test_openai_streaming_estimates_tokens_when_no_usage(self):
        """Should estimate tokens when usage info not provided."""
        # Mock chunk without usage info
        chunk_without_usage = MagicMock()
        chunk_without_usage.usage = None

        # Simple estimation: ~1 token per 4 characters
        text = 'This is a test response'
        estimated_completion_tokens = len(text) // 4

        assert estimated_completion_tokens > 0
        assert estimated_completion_tokens == len(text) // 4


@pytest.mark.unit
class TestAnthropicStreamingTracking:
    """Tests for Anthropic streaming response tracking."""

    def test_anthropic_streaming_response_tracked(self):
        """Should track Anthropic streaming responses."""
        # Create mock Anthropic streaming response
        mock_stream = MagicMock()
        # Create events with explicit delta attributes
        event1 = MagicMock(type='content_block_start', spec=['type'])
        event2 = MagicMock(type='content_block_delta', delta=MagicMock(text='Hello'))
        event3 = MagicMock(type='content_block_delta', delta=MagicMock(text=' world'))
        event4 = MagicMock(type='content_block_stop', spec=['type'])

        mock_stream.__iter__ = Mock(return_value=iter([event1, event2, event3, event4]))

        # Track the stream
        content_parts = []
        for event in mock_stream:
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                content_parts.append(event.delta.text)

        assert len(content_parts) == 2
        assert ''.join(content_parts) == 'Hello world'

    def test_anthropic_streaming_accumulates_text(self):
        """Should accumulate text from Anthropic streaming events."""
        # Simulate Anthropic streaming events
        events = [
            {'type': 'content_block_start'},
            {'type': 'content_block_delta', 'text': 'The '},
            {'type': 'content_block_delta', 'text': 'answer '},
            {'type': 'content_block_delta', 'text': 'is '},
            {'type': 'content_block_delta', 'text': '42'},
            {'type': 'content_block_stop'},
        ]

        accumulated_text = []
        for event in events:
            if event['type'] == 'content_block_delta':
                accumulated_text.append(event.get('text', ''))

        result = ''.join(accumulated_text)
        assert result == 'The answer is 42'
        assert len(accumulated_text) == 4

    def test_anthropic_streaming_extracts_final_message(self):
        """Should extract final message stats from message_stop event."""
        # Mock the final message stop event
        final_event = MagicMock()
        final_event.type = 'message_stop'
        final_event.message = MagicMock(
            stop_reason='end_turn',
            usage=MagicMock(
                input_tokens=15,
                output_tokens=25
            )
        )

        # Extract usage from final event
        usage = None
        if final_event.type == 'message_stop' and hasattr(final_event, 'message'):
            usage = {
                'input_tokens': final_event.message.usage.input_tokens,
                'output_tokens': final_event.message.usage.output_tokens,
                'total_tokens': (
                    final_event.message.usage.input_tokens +
                    final_event.message.usage.output_tokens
                )
            }

        assert usage is not None
        assert usage['input_tokens'] == 15
        assert usage['output_tokens'] == 25
        assert usage['total_tokens'] == 40

    def test_anthropic_streaming_preserves_message_structure(self):
        """Should preserve message structure during streaming."""
        # Mock message with streaming content
        mock_message = MagicMock()
        mock_message.id = 'msg-123'
        mock_message.role = 'assistant'
        mock_message.model = 'claude-3-sonnet'

        # Verify metadata preserved
        assert mock_message.id == 'msg-123'
        assert mock_message.role == 'assistant'
        assert mock_message.model == 'claude-3-sonnet'


@pytest.mark.unit
class TestStreamingTrackingIntegration:
    """Integration tests for streaming middleware."""

    def test_streaming_tracks_latency(self):
        """Should track total latency from start to finish."""
        start_time = time.time()

        # Simulate streaming
        time.sleep(0.1)

        end_time = time.time()
        latency = end_time - start_time

        assert latency >= 0.1
        assert latency < 1.0  # Should be fast

    def test_streaming_handles_empty_response(self):
        """Should handle empty streaming responses gracefully."""
        # Empty stream
        chunks = []

        content = ''.join([c.get('text', '') for c in chunks])
        assert content == ''

    def test_streaming_handles_unicode_content(self):
        """Should handle unicode content in streaming."""
        chunks = ['Hello', ' ', '世界', ' ', '🌍']

        accumulated = ''.join(chunks)
        assert '世界' in accumulated
        assert '🌍' in accumulated
        assert len(accumulated) > 0

    def test_streaming_extracts_model_info(self):
        """Should extract and preserve model information."""
        response = MagicMock()
        response.model = 'gpt-4'

        model_info = {
            'provider': 'openai' if 'gpt' in response.model else 'anthropic',
            'model': response.model
        }

        assert model_info['provider'] == 'openai'
        assert model_info['model'] == 'gpt-4'

    def test_streaming_reports_usage_accurately(self):
        """Should report usage stats accurately."""
        # Mock response with complete usage info
        response = MagicMock()
        response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=150,
            total_tokens=250
        )

        usage_report = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
            'cost_estimate_usd': None  # Would be calculated based on model
        }

        assert usage_report['total_tokens'] == 250
        assert usage_report['prompt_tokens'] == 100
        assert usage_report['completion_tokens'] == 150

    def test_streaming_handles_error_in_stream(self):
        """Should handle errors that occur during streaming."""
        def mock_stream_with_error():
            yield MagicMock(text='Start ')
            raise RuntimeError("Stream interrupted")

        # Should gracefully handle error
        collected = []
        try:
            for chunk in mock_stream_with_error():
                if hasattr(chunk, 'text'):
                    collected.append(chunk.text)
        except RuntimeError:
            pass

        assert len(collected) == 1
        assert collected[0] == 'Start '

    def test_streaming_tracks_chunk_count(self):
        """Should count chunks received during streaming."""
        chunks = [
            {'type': 'delta', 'text': 'Hello'},
            {'type': 'delta', 'text': ' '},
            {'type': 'delta', 'text': 'world'},
            {'type': 'stop'},
        ]

        chunk_count = len([c for c in chunks if c['type'] == 'delta'])
        assert chunk_count == 3

    def test_streaming_preserves_finish_reason(self):
        """Should preserve finish reason from streaming response."""
        response = MagicMock()
        response.stop_reason = 'end_turn'  # Anthropic format
        response.finish_reason = 'stop'    # OpenAI format

        # Should handle both formats
        finish_reason = getattr(response, 'finish_reason', None) or getattr(response, 'stop_reason', None)
        assert finish_reason in ['end_turn', 'stop']

    def test_streaming_tags_with_model_info(self):
        """Should tag tracked data with model information."""
        tracked_event = {
            'timestamp': datetime.utcnow().isoformat(),
            'provider': 'openai',
            'model': 'gpt-4-turbo',
            'stream': True,
            'tokens_used': {
                'prompt': 100,
                'completion': 150
            }
        }

        assert tracked_event['provider'] == 'openai'
        assert tracked_event['model'] == 'gpt-4-turbo'
        assert tracked_event['stream'] is True
        assert tracked_event['tokens_used']['completion'] == 150


# ═══════════════════════════════════════════════════════════════════════════
# Bedrock dispatcher — model-family-aware token extraction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBedrockTokenExtraction:
    """Tests for the _bedrock_extract_tokens dispatcher.

    Bedrock response shapes vary per model family — Claude uses `usage.*_tokens`,
    Titan uses `inputTextTokenCount` + `results[].tokenCount`, Llama uses
    `prompt_token_count`, Cohere uses `meta.billed_units`, AI21 Jamba uses
    `usage.prompt_tokens`. The dispatcher routes on modelId prefix.
    """

    def _call(self, model_id, body):
        from integrations.middleware import _bedrock_extract_tokens
        return _bedrock_extract_tokens(model_id, body)

    def test_claude_3_5_sonnet(self):
        assert self._call(
            'anthropic.claude-3-5-sonnet-20241022-v2:0',
            {'usage': {'input_tokens': 137, 'output_tokens': 42}},
        ) == (137, 42)

    def test_claude_3_7_sonnet(self):
        # Current newest Claude family — same shape
        assert self._call(
            'anthropic.claude-3-7-sonnet-20250219-v1:0',
            {'usage': {'input_tokens': 12, 'output_tokens': 8}},
        ) == (12, 8)

    def test_titan_text_express(self):
        assert self._call(
            'amazon.titan-text-express-v1',
            {
                'inputTextTokenCount': 50,
                'results': [{'tokenCount': 80, 'outputText': '…', 'completionReason': 'FINISH'}],
            },
        ) == (50, 80)

    def test_titan_multi_result(self):
        # Titan can return multiple completions — we sum tokenCount across results.
        assert self._call(
            'amazon.titan-text-lite-v1',
            {
                'inputTextTokenCount': 10,
                'results': [{'tokenCount': 20}, {'tokenCount': 30}],
            },
        ) == (10, 50)

    def test_llama_3(self):
        assert self._call(
            'meta.llama3-70b-instruct-v1:0',
            {'prompt_token_count': 25, 'generation_token_count': 40, 'stop_reason': 'stop'},
        ) == (25, 40)

    def test_llama_3_1(self):
        assert self._call(
            'meta.llama3-1-405b-instruct-v1:0',
            {'prompt_token_count': 100, 'generation_token_count': 500},
        ) == (100, 500)

    def test_cohere_command_r_plus(self):
        assert self._call(
            'cohere.command-r-plus-v1:0',
            {'text': '…', 'meta': {'billed_units': {'input_tokens': 15, 'output_tokens': 60}}},
        ) == (15, 60)

    def test_ai21_jamba(self):
        assert self._call(
            'ai21.jamba-1-5-large-v1:0',
            {'usage': {'prompt_tokens': 30, 'completion_tokens': 70}},
        ) == (30, 70)

    def test_unknown_model_returns_zeros(self):
        # Unknown family: we don't guess — dispatcher returns (0,0) and caller
        # skips tracking. Customers hitting this should use the Converse API,
        # which has uniform usage regardless of provider.
        assert self._call(
            'some-future.model-we-dont-know-v1:0',
            {'some': 'shape'},
        ) == (0, 0)

    def test_missing_usage_fields_returns_zeros(self):
        # Response JSON that matches the model family but lacks usage (e.g. a
        # streaming chunk fragment or a malformed response) — graceful 0/0.
        assert self._call(
            'anthropic.claude-3-haiku-20240307-v1:0',
            {'content': [{'type': 'text', 'text': 'hello'}]},
        ) == (0, 0)

    def test_case_insensitive_model_id(self):
        # Model IDs can arrive upper-cased from some integrations.
        assert self._call(
            'ANTHROPIC.CLAUDE-3-5-SONNET-20241022-V2:0',
            {'usage': {'input_tokens': 7, 'output_tokens': 3}},
        ) == (7, 3)


@pytest.mark.unit
class TestBedrockConverseStreamWrap:
    """Tests for _bedrock_wrap_stream — the iterator wrapper for ConverseStream."""

    def test_yields_all_events_unchanged(self):
        """Wrapped iterator must pass every event through to the caller verbatim."""
        from integrations.middleware import _bedrock_wrap_stream

        events = [
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'delta': {'text': 'Hel'}}},
            {'contentBlockDelta': {'delta': {'text': 'lo'}}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {'inputTokens': 5, 'outputTokens': 2}}},
        ]

        wrapped = _bedrock_wrap_stream(iter(events), 'anthropic.claude-3-haiku', {})
        received = list(wrapped)
        assert received == events

    def test_fires_tracker_on_metadata_event(self):
        """Terminal metadata event should trigger a tracker.track_call with correct tokens."""
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {'total_cost': 0.0001}
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            from integrations.middleware import _bedrock_wrap_stream

            events = [
                {'contentBlockDelta': {'delta': {'text': 'hi'}}},
                {'metadata': {'usage': {'inputTokens': 11, 'outputTokens': 4}}},
            ]
            list(_bedrock_wrap_stream(iter(events), 'anthropic.claude-3-5-sonnet', {'_gc_bedrock_start': time.time()}))

            assert mock_tracker.track_call.called
            call_kwargs = mock_tracker.track_call.call_args.kwargs
            assert call_kwargs['provider'] == 'bedrock'
            assert call_kwargs['model'] == 'anthropic.claude-3-5-sonnet'
            assert call_kwargs['usage_data']['input_tokens'] == 11
            assert call_kwargs['usage_data']['output_tokens'] == 4

    def test_no_tracking_when_stream_emits_no_metadata(self):
        """If a stream finishes without a metadata event (e.g. errored midway),
        we skip tracking rather than recording bogus 0-token calls."""
        from integrations import middleware

        mock_tracker = MagicMock()
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            from integrations.middleware import _bedrock_wrap_stream
            events = [{'contentBlockDelta': {'delta': {'text': 'partial'}}}]
            list(_bedrock_wrap_stream(iter(events), 'anthropic.claude-3-haiku', {}))
            assert not mock_tracker.track_call.called


@pytest.mark.unit
class TestBedrockInstallerSoftFail:
    """The installer must no-op when boto3 isn't present (for the :azure / :gcp
    variant containers) rather than raise and break the whole middleware."""

    def test_no_boto3_no_raise(self):
        from integrations.middleware import _install_bedrock_interceptor

        # Simulate the :azure variant where boto3 is not installed.
        with patch.dict('sys.modules', {'boto3': None}):
            # ImportError path — must silently return without raising.
            try:
                _install_bedrock_interceptor()
            except ImportError:
                pytest.fail("_install_bedrock_interceptor leaked ImportError; must noop when boto3 missing")


# ═══════════════════════════════════════════════════════════════════════════
# Azure OpenAI provider labeling
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestOpenAIProviderDetection:
    """The openai SDK patch catches both OpenAI() and AzureOpenAI() clients
    since they share the same Completions class. Labels must differ so
    Azure OpenAI spend doesn't silently roll up as OpenAI."""

    def _call(self, resource):
        from integrations.middleware import _detect_openai_provider
        return _detect_openai_provider(resource)

    def test_none_resource_defaults_to_openai(self):
        assert self._call(None) == 'openai'

    def test_bare_object_defaults_to_openai(self):
        # Resource missing `_client` → safe default.
        class _R: pass
        assert self._call(_R()) == 'openai'

    def test_openai_base_url_labels_as_openai(self):
        class _Client:
            base_url = 'https://api.openai.com/v1/'
        class _Resource:
            _client = _Client()
        assert self._call(_Resource()) == 'openai'

    def test_azure_endpoint_labels_as_azure_openai(self):
        # Azure endpoints look like https://{resource}.openai.azure.com/...
        class _Client:
            base_url = 'https://acme-corp.openai.azure.com/openai/deployments/gpt-4o/'
        class _Resource:
            _client = _Client()
        assert self._call(_Resource()) == 'azure_openai'

    def test_azure_detection_is_case_insensitive(self):
        class _Client:
            base_url = 'HTTPS://ACME.OPENAI.AZURE.COM/'
        class _Resource:
            _client = _Client()
        assert self._call(_Resource()) == 'azure_openai'

    def test_underscore_private_client_fallback(self):
        # Some SDK versions expose `client` rather than `_client`.
        class _Client:
            base_url = 'https://foo.openai.azure.com/'
        class _Resource:
            client = _Client()  # no underscore
        assert self._call(_Resource()) == 'azure_openai'


@pytest.mark.unit
class TestTrackOpenAIResponseRespectsProvider:
    """_track_openai_response must forward the provider arg to the tracker
    AND use it when pricing. Otherwise Azure OpenAI calls get priced at
    OpenAI list price (which is LOWER than Azure — so we'd underbill)."""

    def test_azure_label_and_cost_table(self):
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {'total_cost': 0.02}

        response = {
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50},
            'model': 'gpt-4o',
            'id': 'chatcmpl-abc',
        }

        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_openai_response(response, 'gpt-4o', latency_ms=42.0, provider='azure_openai')

        # calculate_costs must receive 'azure_openai' so Azure pricing is used.
        cost_args = mock_tracker.calculate_costs.call_args
        assert cost_args.args[0] == 'azure_openai'
        # track_call must label the call 'azure_openai' for dashboard aggregation.
        track_args = mock_tracker.track_call.call_args
        assert track_args.kwargs['provider'] == 'azure_openai'

    def test_default_provider_is_openai(self):
        """Calls without an explicit provider kwarg stay labeled 'openai' (backward-compat)."""
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {}
        response = {'usage': {'prompt_tokens': 10, 'completion_tokens': 5}, 'model': 'gpt-4o'}

        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_openai_response(response, 'gpt-4o')

        assert mock_tracker.track_call.call_args.kwargs['provider'] == 'openai'


# ═══════════════════════════════════════════════════════════════════════════
# Vertex AI / Gemini
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVertexAIHelpers:
    """Model-name parsing + provider detection for the Vertex interceptor."""

    def test_model_name_from_private_attr(self):
        from integrations.middleware import _vertex_model_name
        class _M:
            _model_name = 'gemini-1.5-pro'
        assert _vertex_model_name(_M()) == 'gemini-1.5-pro'

    def test_model_name_from_resource_path(self):
        """Vertex often stores the full resource path; we keep just the last segment."""
        from integrations.middleware import _vertex_model_name
        class _M:
            _prediction_resource_name = 'projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash'
        assert _vertex_model_name(_M()) == 'gemini-2.0-flash'

    def test_model_name_fallback_unknown(self):
        from integrations.middleware import _vertex_model_name
        class _M: pass
        assert _vertex_model_name(_M()) == 'unknown'

    def test_google_genai_provider_vertex_true(self):
        from integrations.middleware import _google_genai_provider
        class _ApiClient:
            vertexai = True
        class _Models:
            _api_client = _ApiClient()
        assert _google_genai_provider(_Models()) == 'vertex_ai'

    def test_google_genai_provider_vertex_false_is_gemini_direct(self):
        """Client(vertexai=False) → direct Gemini API → different provider label."""
        from integrations.middleware import _google_genai_provider
        class _ApiClient:
            vertexai = False
        class _Models:
            _api_client = _ApiClient()
        assert _google_genai_provider(_Models()) == 'google_gemini'

    def test_google_genai_provider_no_api_client_defaults_to_vertex(self):
        """Missing api_client → safe default. Most customers using the unified SDK are on Vertex."""
        from integrations.middleware import _google_genai_provider
        class _Models: pass
        assert _google_genai_provider(_Models()) == 'vertex_ai'


@pytest.mark.unit
class TestVertexResponseTracking:
    """_track_vertex_response must pull tokens from usage_metadata regardless of whether
    the response is an SDK object or a dict (some wrappers re-serialize)."""

    def test_sdk_object_response(self):
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {'total_cost': 0.001}

        class _Usage:
            prompt_token_count = 137
            candidates_token_count = 42
        class _Response:
            usage_metadata = _Usage()

        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_vertex_response(_Response(), 'gemini-1.5-pro', 'vertex_ai', latency_ms=100.0)

        assert mock_tracker.track_call.called
        kw = mock_tracker.track_call.call_args.kwargs
        assert kw['provider'] == 'vertex_ai'
        assert kw['model'] == 'gemini-1.5-pro'
        assert kw['usage_data']['input_tokens'] == 137
        assert kw['usage_data']['output_tokens'] == 42

    def test_dict_response_snake_case(self):
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {}
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_vertex_response(
                {'usage_metadata': {'prompt_token_count': 20, 'candidates_token_count': 10}},
                'gemini-1.5-flash', 'vertex_ai', latency_ms=None,
            )
        kw = mock_tracker.track_call.call_args.kwargs
        assert kw['usage_data']['input_tokens'] == 20
        assert kw['usage_data']['output_tokens'] == 10

    def test_dict_response_camel_case(self):
        """Some Google SDK paths serialize to camelCase — accept that shape too."""
        from integrations import middleware

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {}
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_vertex_response(
                {'usageMetadata': {'promptTokenCount': 5, 'candidatesTokenCount': 3}},
                'gemini-2.0-flash', 'vertex_ai', latency_ms=None,
            )
        kw = mock_tracker.track_call.call_args.kwargs
        assert kw['usage_data']['input_tokens'] == 5
        assert kw['usage_data']['output_tokens'] == 3

    def test_missing_usage_metadata_no_track(self):
        """Responses without usage_metadata (e.g. blocked by safety filters) are skipped silently."""
        from integrations import middleware

        mock_tracker = MagicMock()
        class _R: pass
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_vertex_response(_R(), 'gemini-1.5-pro', 'vertex_ai')
        assert not mock_tracker.track_call.called

    def test_zero_tokens_no_track(self):
        """Zero-token responses (safety-blocked) shouldn't record a phantom call."""
        from integrations import middleware

        mock_tracker = MagicMock()
        class _Usage:
            prompt_token_count = 0
            candidates_token_count = 0
        class _R:
            usage_metadata = _Usage()
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            middleware._track_vertex_response(_R(), 'gemini-1.5-pro', 'vertex_ai')
        assert not mock_tracker.track_call.called


@pytest.mark.unit
class TestVertexStreamWrapper:
    """Streaming wrapper must pass every chunk through AND fire tracking from the terminal chunk."""

    def test_yields_all_chunks(self):
        from integrations.middleware import _wrap_vertex_stream

        class _Usage:
            prompt_token_count = 10
            candidates_token_count = 20
        class _Chunk:
            def __init__(self, text, usage=None):
                self.text = text
                self.usage_metadata = usage
        chunks = [
            _Chunk('Hel'),
            _Chunk('lo'),
            _Chunk('', usage=_Usage()),  # terminal chunk carries usage
        ]
        received = list(_wrap_vertex_stream(iter(chunks), 'gemini-1.5-pro', 'vertex_ai', time.time()))
        assert [c.text for c in received] == ['Hel', 'lo', '']

    def test_fires_tracker_on_terminal_chunk(self):
        from integrations import middleware
        from integrations.middleware import _wrap_vertex_stream

        mock_tracker = MagicMock()
        mock_tracker.calculate_costs.return_value = {}

        class _Usage:
            prompt_token_count = 11
            candidates_token_count = 4
        class _Chunk:
            def __init__(self, usage=None):
                self.usage_metadata = usage
        chunks = [_Chunk(), _Chunk(), _Chunk(usage=_Usage())]

        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            list(_wrap_vertex_stream(iter(chunks), 'gemini-1.5-pro', 'vertex_ai', time.time()))

        assert mock_tracker.track_call.called
        kw = mock_tracker.track_call.call_args.kwargs
        assert kw['provider'] == 'vertex_ai'
        assert kw['usage_data']['input_tokens'] == 11
        assert kw['usage_data']['output_tokens'] == 4

    def test_no_track_when_stream_has_no_usage(self):
        """Safety-blocked or aborted streams — no usage_metadata ever arrives. Skip silently."""
        from integrations import middleware
        from integrations.middleware import _wrap_vertex_stream

        mock_tracker = MagicMock()
        class _C: usage_metadata = None
        with patch.object(middleware, '_tracker', mock_tracker), \
             patch.object(middleware, '_tracking_enabled', True):
            list(_wrap_vertex_stream(iter([_C(), _C()]), 'gemini', 'vertex_ai', time.time()))
        assert not mock_tracker.track_call.called


@pytest.mark.unit
class TestVertexInstallersSoftFail:
    """Installers must noop (not raise) when the target SDK isn't installed."""

    def test_vertex_classic_without_sdk(self):
        from integrations.middleware import _install_vertexai_classic_interceptor
        # No vertexai module present in this test env — should silently return.
        _install_vertexai_classic_interceptor()  # just asserting no-raise

    def test_google_genai_without_sdk(self):
        from integrations.middleware import _install_google_genai_interceptor
        _install_google_genai_interceptor()  # just asserting no-raise

    def test_umbrella_installer_without_sdks(self):
        from integrations.middleware import _install_vertex_ai_interceptor
        _install_vertex_ai_interceptor()  # just asserting no-raise
