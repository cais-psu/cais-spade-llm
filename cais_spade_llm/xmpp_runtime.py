"""Runtime patches for the embedded pyjabber XMPP server."""

from __future__ import annotations

import asyncio
import importlib
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


def _local_xmpp_only() -> bool:
    """Return whether embedded localhost XMPP should avoid TLS entirely."""
    return _env_flag("CAIS_XMPP_LOCAL_ONLY", True)


def _is_loopback_xmpp_host(host: Any) -> bool:
    text = str(host or "").strip().lower().strip("[]")
    return text in {"localhost", "127.0.0.1", "::1"} or text.startswith("127.")


def _jid_host(jid: Any) -> str:
    host = getattr(jid, "host", "")
    if host:
        return str(host)
    text = str(jid or "").strip()
    if "@" in text:
        text = text.split("@", 1)[1]
    if "/" in text:
        text = text.split("/", 1)[0]
    return text


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
        if not _local_xmpp_only():
            tasks.append(asyncio.create_task(_safe_tls_worker(), name="xmpp_tls_worker"))
        tasks.extend(
            [
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


def _patch_pyjabber_stream_handler(*, local_only: bool) -> None:
    """Advertise plaintext local features instead of STARTTLS when requested."""
    stream_handler_module = importlib.import_module("pyjabber.stream.StreamHandler")
    stream_handler_cls = stream_handler_module.StreamHandler

    original_attr = "_cais_original_handle_init"
    if not hasattr(stream_handler_cls, original_attr):
        setattr(stream_handler_cls, original_attr, stream_handler_cls._handle_init)

    if not local_only:
        stream_handler_cls._handle_init = getattr(stream_handler_cls, original_attr)
        return

    def _handle_init_plaintext(self: Any, _: Any) -> None:
        self._streamFeature.reset()
        self._streamFeature.register(stream_handler_module.IBR.InBandRegistration())
        self._streamFeature.register(stream_handler_module.SASLFeature())
        self._transport.write(self._streamFeature.to_bytes())
        self._stage = stream_handler_module.Stage.SASL

    stream_handler_cls._handle_init = _handle_init_plaintext


def _patch_spade_xmpp_client(*, local_only: bool) -> None:
    """Configure SPADE/slixmpp clients to use plaintext for local embedded XMPP."""
    try:
        import spade.xmpp_client as spade_xmpp_client
    except ImportError:
        _log.debug("SPADE not available; skipping XMPP client plaintext patch")
        return

    client_cls = spade_xmpp_client.XMPPClient
    original_attr = "_cais_original_init"
    if not hasattr(client_cls, original_attr):
        setattr(client_cls, original_attr, client_cls.__init__)
    original_init = getattr(client_cls, original_attr)

    if not local_only:
        client_cls.__init__ = original_init
        return

    def _init_local_plaintext(self: Any, jid: Any, password: str, verify_security: bool, auto_register: bool) -> None:
        loopback_client = _is_loopback_xmpp_host(_jid_host(jid))
        # Local pyjabber auto-creates loopback credentials on first auth, so
        # skip XEP-0077 registration and avoid its slow timeout path.
        original_init(self, jid, password, verify_security, False if loopback_client else auto_register)
        host = getattr(getattr(self, "boundjid", None), "host", "")
        if _is_loopback_xmpp_host(host):
            self.enable_starttls = False
            self.enable_direct_tls = False
            self.enable_plaintext = True
            self.starttls_services = set()
            self.tls_services = set()
            self._cais_skipped_local_registration = bool(auto_register)
            try:
                self["feature_mechanisms"].unencrypted_plain = True
            except Exception:
                _log.debug("Could not enable local plaintext SASL PLAIN", exc_info=True)

    client_cls.__init__ = _init_local_plaintext


def _patch_pyjabber_sasl(*, local_only: bool) -> None:
    """Auto-create credentials for trusted loopback plaintext clients."""
    sasl_module = importlib.import_module("pyjabber.features.SASLFeature")
    sasl_cls = sasl_module.SASL

    original_attr = "_cais_original_handle_auth"
    if not hasattr(sasl_cls, original_attr):
        setattr(sasl_cls, original_attr, sasl_cls.handleAuth)
    original_handle_auth = getattr(sasl_cls, original_attr)

    if not local_only:
        sasl_cls.handleAuth = original_handle_auth
        return

    def _handle_auth_local_autocreate(self: Any, element: Any) -> Any:
        peername = getattr(self, "_peername", None)
        peer_host = peername[0] if isinstance(peername, tuple) and peername else ""
        if not _is_loopback_xmpp_host(peer_host):
            return original_handle_auth(self, element)

        try:
            data = sasl_module.base64.b64decode(element.text or "").split(b"\x00")
            jid = data[1].decode()
            pwd = data[2]
            if not jid or not pwd:
                return sasl_module.SE.not_authorized()

            with sasl_module.DB.connection() as con:
                query = sasl_module.select(sasl_module.Model.Credentials.c.hash_pwd).where(
                    sasl_module.Model.Credentials.c.jid == jid
                )
                credentials = con.execute(query).fetchone()
                if credentials:
                    if not sasl_module.bcrypt.checkpw(pwd, credentials[0]):
                        return sasl_module.SE.not_authorized()
                else:
                    hashed_pwd = sasl_module.bcrypt.hashpw(pwd, sasl_module.bcrypt.gensalt())
                    query_insert = sasl_module.insert(sasl_module.Model.Credentials).values(
                        {"jid": jid, "hash_pwd": hashed_pwd}
                    )
                    con.execute(query_insert)
                    con.commit()

            self._connection_manager.set_jid(
                peername,
                sasl_module.JID(user=jid, domain=sasl_module.metadata.HOST),
            )
            return (
                sasl_module.Signal.RESET,
                b"<success xmlns='urn:ietf:params:xml:ns:xmpp-sasl'/>",
            )
        except Exception as exc:
            _log.debug("Local XMPP auth failed for peer %s: %s", peername, exc)
            return sasl_module.SE.not_authorized()

    sasl_cls.handleAuth = _handle_auth_local_autocreate


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

    local_only = _local_xmpp_only()
    if local_only:
        def _local_only_ip() -> None:
            return None

        pyjabber_init_utils.setup_query_local_ip = _local_only_ip
        pyjabber_server.init_utils.setup_query_local_ip = _local_only_ip

    _patch_pyjabber_stream_handler(local_only=local_only)
    _patch_pyjabber_sasl(local_only=local_only)
    _patch_spade_xmpp_client(local_only=local_only)
    pyjabber_workers.tls_worker = _safe_tls_worker
    pyjabber_server.tls_worker = _safe_tls_worker
    pyjabber_server.Server.start = _patched_server_start
