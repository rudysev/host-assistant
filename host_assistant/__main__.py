"""Assemble and run the local voice host.

Pipeline (app mic → app speaker):
    transport.input → Whisper STT → InputTranscriptEmitter → context(user)
                    → Qwen3 (Ollama) → Kokoro TTS → ControlBridge → transport.output
                    → context(assistant)

The WebSocket transport uses PortalSerializer (our wire); turn-taking VAD (Silero) is configured on the
user context aggregator. The app streams PCM and gets PCM + control frames back — the Gemini-shaped
contract, served locally.

Uses Pipecat's PipelineWorker + LLMContextAggregatorPair. If a Pipecat upgrade breaks startup, check
frame class imports, aggregator params, and LLM tool-registration APIs first.
"""

from __future__ import annotations

import asyncio
import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InterruptionFrame, OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transports.websocket.server import SingleClientWebsocketServerParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.workers.runner import WorkerRunner

from host_assistant import protocol
from host_assistant.config import CONFIG, ConfigError, require_openssl, startup_warnings, web_search_available
from host_assistant.logging_config import configure_logging
from host_assistant.pipeline.bridge import DEFAULT_TURN_ERROR, ControlBridge
from host_assistant.pipeline.input_transcript import InputTranscriptEmitter
from host_assistant.pipeline.serializer import PortalSerializer
from host_assistant.pipeline.session import Session
from host_assistant.text.filters import EmojiTextFilter
from host_assistant.tools.host_tools import warm_web_search
from host_assistant.transport.tls import load_server_ssl_context
from host_assistant.transport.websocket import TlsWebsocketServerTransport
from host_assistant.warmup.kokoro import warm_kokoro
from host_assistant.warmup.ollama import warm_ollama
from host_assistant.warmup.whisper import warm_whisper

log = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    try:
        require_openssl()
    except ConfigError as exc:
        log.error("config error: %s", exc)
        raise SystemExit(1) from exc

    for warning in startup_warnings():
        log.warning("%s", warning)

    session = Session()
    bridge = ControlBridge()

    transport = TlsWebsocketServerTransport(
        host=CONFIG.host,
        port=CONFIG.port,
        ssl_context=load_server_ssl_context(),
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=CONFIG.input_sample_rate,
            audio_out_sample_rate=CONFIG.output_sample_rate,
            serializer=PortalSerializer(session),
        ),
    )

    llm = OpenAILLMService(
        base_url=CONFIG.ollama_base_url,
        api_key="ollama",
        settings=OpenAILLMService.Settings(
            model=CONFIG.ollama_model,
            max_tokens=CONFIG.ollama_max_tokens,
        ),
    )
    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(model=CONFIG.whisper_model),
        ttfs_p99_latency=1.5,
    )
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=CONFIG.kokoro_voice),
        sample_rate=CONFIG.output_sample_rate,
        text_filters=[MarkdownTextFilter(), EmojiTextFilter()],
    )

    async def push_message(message: str) -> None:
        # Urgent control (ready, tool_call, error, input_transcript) must reach the Portal
        # immediately — do not queue at the pipeline input while the LLM/TTS path is busy.
        await transport.output().send_message(OutputTransportMessageUrgentFrame(message=message))

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=CONFIG.input_sample_rate,
                params=VADParams(stop_secs=CONFIG.vad_stop_secs),
            ),
        ),
    )
    input_transcript = InputTranscriptEmitter(push_message)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            input_transcript,
            aggregators.user(),
            llm,
            tts,
            bridge,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=CONFIG.input_sample_rate,
            audio_out_sample_rate=CONFIG.output_sample_rate,
            enable_metrics=True,
            report_only_initial_ttfb=True,
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,
    )

    session.configure(llm=llm, context=context, push_message=push_message)

    @worker.event_handler("on_pipeline_error")
    async def _on_pipeline_error(_worker, frame):  # noqa: ANN001 - Pipecat event signature
        message = frame.error or DEFAULT_TURN_ERROR
        await push_message(protocol.error(message))
        await push_message(protocol.turn_complete())
        bridge.sync_turn_completed()

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client):  # noqa: ANN001 - Pipecat event signature
        session.reset()
        bridge.reset_turn_state()

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client):  # noqa: ANN001 - Pipecat event signature
        # Drop in-flight LLM/TTS so a late reply cannot "speak" into a dead socket
        # (e.g. Portal stall-timeout closed the session mid host-tool turn).
        await worker.queue_frame(InterruptionFrame())
        session.reset()
        bridge.reset_turn_state()

    warm_task: asyncio.Task | None = None
    if (
        CONFIG.warm_ollama_at_startup
        or CONFIG.warm_kokoro_at_startup
        or CONFIG.warm_tavily_at_startup
        or CONFIG.warm_whisper_at_startup
    ):

        async def _warm() -> None:
            tasks: list[tuple[str, asyncio.Task]] = []
            if CONFIG.warm_ollama_at_startup:
                tasks.append(("Ollama", asyncio.create_task(warm_ollama(CONFIG.ollama_base_url, CONFIG.ollama_model))))
            if CONFIG.warm_kokoro_at_startup:
                tasks.append(("Kokoro", asyncio.create_task(warm_kokoro(tts))))
            if CONFIG.warm_tavily_at_startup and web_search_available():
                tasks.append(("web_search", asyncio.create_task(warm_web_search())))
            if CONFIG.warm_whisper_at_startup:
                tasks.append(
                    (
                        "Whisper",
                        asyncio.create_task(
                            warm_whisper(stt, sample_rate=CONFIG.input_sample_rate)
                        ),
                    )
                )
            for label, task in tasks:
                try:
                    await task
                    if label == "Ollama":
                        log.info("Ollama warm: %s", CONFIG.ollama_model)
                    elif label == "Kokoro":
                        log.info("Kokoro warm: %s", CONFIG.kokoro_voice)
                    elif label == "Whisper":
                        log.info("Whisper warm: %s", CONFIG.whisper_model)
                    else:
                        log.info("web_search warm: %s", CONFIG.web_search_backend)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - warm-up is best-effort
                    log.warning("%s warm-up failed: %s", label, e)

        warm_task = asyncio.create_task(_warm())

    log.info(
        "Local voice host listening on wss://%s:%s  model=%s  max_tokens=%d  (/no_think prompt)",
        CONFIG.host,
        CONFIG.port,
        CONFIG.ollama_model,
        CONFIG.ollama_max_tokens,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if warm_task is not None and not warm_task.done():
            warm_task.cancel()
            try:
                await warm_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
