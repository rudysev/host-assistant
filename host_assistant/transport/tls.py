"""Auto-generated self-signed TLS for the LAN voice host — zero manual cert setup.

Requires the system ``openssl`` binary on first boot (to mint the cert into ``~/.host-assistant/tls/``).
Install via your OS package manager if it is not already on ``PATH``.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
from pathlib import Path

DEFAULT_CERT_DIR = Path.home() / ".host-assistant" / "tls"
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"


def ensure_self_signed_cert(cert_dir: Path = DEFAULT_CERT_DIR) -> tuple[Path, Path]:
    """Return (cert, key) paths, generating a 10-year self-signed pair on first use."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / CERT_FILE
    key_path = cert_dir / KEY_FILE
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl not found — required to auto-generate the LAN TLS certificate")

    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "3650",
            "-nodes",
            "-subj",
            "/CN=portal-local-voice-host",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cert_path, key_path


def load_server_ssl_context(cert_dir: Path | None = None) -> ssl.SSLContext:
    """TLS server context backed by the auto-generated self-signed cert."""
    cert_path, key_path = ensure_self_signed_cert(cert_dir or DEFAULT_CERT_DIR)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return ctx
