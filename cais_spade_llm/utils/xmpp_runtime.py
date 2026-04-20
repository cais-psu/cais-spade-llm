"""Runtime patches for the embedded pyjabber XMPP server."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import ssl
from typing import Any

_INSTALLED = False
_log = logging.getLogger("xmpp.runtime")


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


async def _safe_tls_worker() -> None:
    """Handle STARTTLS upgrades without crashing the whole embedded server."""
    from pyjabber import metadata
    from pyjabber.network import CertGenerator
    from pyjabber.network.ConnectionManager import ConnectionManager
    from pyjabber.network.XMLProtocol import TransportProxy

    try:
        if not CertGenerator.check_hostname_cert_exists(metadata.HOST, metadata.CERT_PATH):
            CertGenerator.generate_hostname_cert(metadata.HOST, metadata.CERT_PATH)
    except FileNotFoundError as exc:
        _log.error(
            "Embedded XMPP cert path is invalid: %s. Set an existing directory for certificates.",
            exc,
        )
        raise SystemExit from exc

    connection_manager = ConnectionManager()
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.load_cert_chain(
        certfile=os.path.join(metadata.CERT_PATH, f"{metadata.HOST}_cert.pem"),
        keyfile=os.path.join(metadata.CERT_PATH, f"{metadata.HOST}_key.pem"),
    )
    loop = asyncio.get_running_loop()
    tls_queue = metadata.TLS_QUEUE
    try:
        while True:
            transport, protocol, parser = await tls_queue.get()
            peer = transport.get_extra_info("peername")
            try:
                upgraded = await loop.start_tls(
                    transport=transport.originalTransport,
                    protocol=protocol,
                    sslcontext=ssl_context,
                    server_side=True,
                )
                upgraded_proxy = TransportProxy(upgraded, peer)
                protocol.transport = upgraded_proxy
                parser.transport = upgraded_proxy
            except (ConnectionResetError, ssl.SSLError, OSError, RuntimeError) as exc:
                _log.debug("Ignoring failed XMPP TLS upgrade from %s: %s", peer, exc)
                try:
                    if not transport.is_closing():
                        transport.close()
                except Exception:
                    pass
                try:
                    connection_manager.disconnection(peer)
                except Exception:
                    pass
    except asyncio.CancelledError:
        pass


async def _patched_server_start(self: Any) -> None:
    """Start pyjabber with a safer TLS worker and optional admin UI."""
    from pyjabber import metadata
    from pyjabber.workers import queue_worker

    metadata.TLS_QUEUE = asyncio.Queue()
    metadata.CONNECTION_QUEUE = asyncio.Queue()
    metadata.MESSAGE_QUEUE = asyncio.Queue()

    for sig in (signal.SIGINT, signal.SIGABRT, signal.SIGTERM):
        try:
            signal.signal(sig, self.raise_exit)
        except ValueError:
            # Signal handlers can only be installed from the main thread.
            break

    tasks: list[asyncio.Task[Any]] = []
    try:
        if _env_flag("CAIS_XMPP_ENABLE_ADMIN", False):
            tasks.append(asyncio.create_task(self._adminServer.start(), name="xmpp_admin"))
        tasks.extend(
            [
                asyncio.create_task(_safe_tls_worker(), name="xmpp_tls_worker"),
                asyncio.create_task(queue_worker(), name="xmpp_queue_worker"),
                asyncio.create_task(self.run_server(), name="xmpp_server"),
            ]
        )
        await asyncio.gather(*tasks)
    except (SystemExit, KeyboardInterrupt, asyncio.CancelledError) as exc:
        _log.debug("Embedded XMPP shutdown via %s", exc.__class__.__name__)
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def install_xmpp_runtime_patches() -> None:
    """Patch pyjabber for local embedded use inside this project."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        import pyjabber.init_utils as pyjabber_init_utils
        import pyjabber.server as pyjabber_server
        import pyjabber.workers as pyjabber_workers
    except ImportError:
        _log.debug("pyjabber not available; skipping embedded XMPP runtime patch")
        return

    if _env_flag("CAIS_XMPP_LOCAL_ONLY", True):

        def _local_only_ip() -> None:
            return None

        pyjabber_init_utils.setup_query_local_ip = _local_only_ip
        pyjabber_server.init_utils.setup_query_local_ip = _local_only_ip

    pyjabber_workers.tls_worker = _safe_tls_worker
    pyjabber_server.tls_worker = _safe_tls_worker
    pyjabber_server.Server.start = _patched_server_start
