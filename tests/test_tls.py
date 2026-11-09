"""Verified local HTTPS fixtures for company CA and optional native trust support."""
from contextlib import contextmanager
import hashlib
import http.server
import importlib.util
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tls


@contextmanager
def local_https(certificate, private_key):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'verified local HTTPS fixture'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate), str(private_key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def open_local(context, url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                        urllib.request.HTTPSHandler(context=context))
    with opener.open(url, timeout=5) as response:
        return response.read()


class TLSContextTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {'SSL_CERT_FILE': ''})
        self.environment.start()
        self.no_native = mock.patch.object(tls.importlib, 'import_module', side_effect=ImportError('fixture: optional module absent'))
        self.no_native.start()

    def tearDown(self):
        self.no_native.stop()
        self.environment.stop()

    def test_default_context_enforces_certificate_and_hostname_checks(self):
        context = tls.create_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_optional_truststore_is_selected_without_global_ssl_patch(self):
        self.no_native.stop()
        original = ssl.SSLContext
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        constructor = mock.Mock(return_value=context)
        native = types.SimpleNamespace(SSLContext=constructor)
        with mock.patch.object(tls.importlib, 'import_module', return_value=native):
            actual = tls.create_ssl_context()
        self.assertIs(actual, context)
        self.assertIs(ssl.SSLContext, original)
        constructor.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_missing_and_invalid_ca_are_clear_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'CA bundle does not exist'):
                tls.create_ssl_context(root / 'absent.pem')
            invalid = root / 'invalid.pem'
            invalid.write_text('not a certificate')
            with self.assertRaisesRegex(ValueError, 'Unable to load CA bundle'):
                tls.create_ssl_context(invalid)

    def test_nested_certificate_error_is_recognized(self):
        failure = ssl.SSLCertVerificationError(1, 'certificate verify failed')
        self.assertTrue(tls.certificate_error(urllib.error.URLError(failure)))
        self.assertFalse(tls.certificate_error(urllib.error.URLError('connection reset')))
        wrapped = RuntimeError('wrapper')
        wrapped.__cause__ = failure
        self.assertTrue(tls.certificate_error(wrapped))

    def test_error_explains_native_store_custom_ca_and_safe_resume(self):
        message = tls.certificate_error_message(ssl.SSLCertVerificationError(1, 'fixture'))
        self.assertIn('truststore', message)
        self.assertIn('--ca-bundle', message)
        self.assertIn('SSL_CERT_FILE', message)
        self.assertIn('Existing verified files', message)
        self.assertIn('remain enabled', message)


@unittest.skipUnless(shutil.which('openssl'), 'openssl is needed to generate local HTTPS test certificates')
class LocalHTTPSCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='model-tls-fixture-')
        cls.root = Path(cls.temporary.name)
        cls.ca = cls.root / 'company root.pem'
        cls.key = cls.root / 'server.key'
        cls.certificate = cls.root / 'server.pem'
        config = cls.root / 'ca.cnf'
        config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=v3_ca\n'
                          '[dn]\nCN=Ephemeral fixture company root\n[v3_ca]\n'
                          'basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\n'
                          'subjectKeyIdentifier=hash\n')
        extensions = cls.root / 'server.ext'
        extensions.write_text('basicConstraints=critical,CA:FALSE\n'
                              'keyUsage=critical,digitalSignature,keyEncipherment\n'
                              'extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\n')
        def command(*arguments):
            subprocess.run(['openssl', *map(str, arguments)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', cls.root/'ca.key',
                '-out', cls.ca, '-days', '1', '-config', config)
        command('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', cls.key,
                '-out', cls.root/'server.csr', '-subj', '/CN=localhost')
        command('x509', '-req', '-in', cls.root/'server.csr', '-CA', cls.ca,
                '-CAkey', cls.root/'ca.key', '-CAcreateserial', '-out', cls.certificate,
                '-days', '1', '-extfile', extensions)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {'SSL_CERT_FILE': ''})
        self.environment.start()
        self.no_native = mock.patch.object(tls.importlib, 'import_module', side_effect=ImportError('fixture'))
        self.no_native.start()

    def tearDown(self):
        self.no_native.stop()
        self.environment.stop()

    def test_company_self_signed_root_is_rejected_by_default(self):
        with local_https(self.certificate, self.key) as port:
            with self.assertRaises(urllib.error.URLError) as caught:
                open_local(tls.create_ssl_context(), 'https://localhost:{}/'.format(port))
        self.assertTrue(tls.certificate_error(caught.exception))

    def test_explicit_company_ca_allows_verified_https(self):
        context = tls.create_ssl_context(self.ca)
        with local_https(self.certificate, self.key) as port:
            self.assertEqual(open_local(context, 'https://localhost:{}/'.format(port)),
                             b'verified local HTTPS fixture')
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_ssl_cert_file_environment_allows_verified_https(self):
        with mock.patch.dict(os.environ, {'SSL_CERT_FILE': str(self.ca)}):
            context = tls.create_ssl_context()
        with local_https(self.certificate, self.key) as port:
            self.assertEqual(open_local(context, 'https://localhost:{}/'.format(port)),
                             b'verified local HTTPS fixture')

    def test_custom_ca_does_not_disable_hostname_verification(self):
        with local_https(self.certificate, self.key) as port:
            with self.assertRaises(urllib.error.URLError) as caught:
                open_local(tls.create_ssl_context(self.ca), 'https://127.0.0.1:{}/'.format(port))
        self.assertTrue(tls.certificate_error(caught.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
