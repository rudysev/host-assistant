"""Unit tests for auto TLS certificate generation."""

from __future__ import annotations

import shutil
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from host_assistant.transport import tls


class TlsTests(unittest.TestCase):
    def test_ensure_self_signed_cert_generates_once(self) -> None:
        if shutil.which("openssl") is None:
            self.skipTest("openssl not installed")

        with tempfile.TemporaryDirectory() as tmp:
            cert_dir = Path(tmp)
            cert1, key1 = tls.ensure_self_signed_cert(cert_dir)
            cert2, key2 = tls.ensure_self_signed_cert(cert_dir)
            self.assertEqual(cert1, cert2)
            self.assertEqual(key1, key2)
            self.assertTrue(cert1.exists())
            self.assertTrue(key1.exists())

    def test_load_server_ssl_context(self) -> None:
        if shutil.which("openssl") is None:
            self.skipTest("openssl not installed")

        with tempfile.TemporaryDirectory() as tmp:
            ctx = tls.load_server_ssl_context(Path(tmp))
            self.assertIsInstance(ctx, ssl.SSLContext)

    def test_missing_openssl_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("host_assistant.transport.tls.shutil.which", return_value=None):
                with self.assertRaises(RuntimeError):
                    tls.ensure_self_signed_cert(Path(tmp))


if __name__ == "__main__":
    unittest.main()
