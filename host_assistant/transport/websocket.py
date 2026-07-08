"""Pipecat WebSocket server transport with auto TLS (wss://)."""

from __future__ import annotations

import logging
import ssl

from websockets.asyncio.server import serve as websocket_serve

log = logging.getLogger(__name__)

from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerInputTransport,
    SingleClientWebsocketServerTransport,
)


class TlsWebsocketServerInputTransport(SingleClientWebsocketServerInputTransport):
    """[SingleClientWebsocketServerInputTransport] that serves ``wss://`` with a supplied SSL context."""

    def __init__(self, *args, ssl_context: ssl.SSLContext, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ssl_context = ssl_context

    async def _server_task_handler(self) -> None:
        log.info("Starting TLS websocket server on %s:%s", self._host, self._port)
        origins = self._params.allowed_origins or None
        async with websocket_serve(
            self._client_handler,
            self._host,
            self._port,
            origins=origins,
            ssl=self._ssl_context,
        ) as server:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _stop_tasks(self) -> None:
        """Exit ``websocket_serve`` cleanly on cancel (Pipecat's default cancel_task hangs)."""
        self._stop_server_event.set()
        if self._server_task and not self._server_task.done():
            await self._server_task
        self._server_task = None
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None


class TlsWebsocketServerTransport(SingleClientWebsocketServerTransport):
    """WebSocket server transport that accepts one client at a time over ``wss://``."""

    def __init__(self, *args, ssl_context: ssl.SSLContext, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ssl_context = ssl_context

    def input(self) -> TlsWebsocketServerInputTransport:
        if not self._input:
            self._input = TlsWebsocketServerInputTransport(
                self,
                self._host,
                self._port,
                self._params,
                self._callbacks,
                ssl_context=self._ssl_context,
                name=self._input_name,
            )
        return self._input
