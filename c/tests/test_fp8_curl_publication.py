import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import download_fp8


class CurlPublicationTests(unittest.TestCase):
    def test_curl_only_publishes_successful_expected_size_transfer(self):
        for status, payload, expected_size, complete in (
                (22, b'oops', 4, False), (18, b'data', 4, False),
                (0, b'longer', 4, False), (0, b'da', 4, False),
                (0, b'data', 4, True), (0, b'data', 0, True)):
            with self.subTest(status=status, size=len(payload), expected=expected_size), \
                    tempfile.TemporaryDirectory() as destination, \
                    mock.patch.object(download_fp8, 'DEST', destination):
                def transfer(command, **kwargs):
                    Path(destination, 'model.safetensors.part').write_bytes(payload)
                    return subprocess.CompletedProcess(command, status)
                with mock.patch.object(download_fp8.subprocess, 'run', side_effect=transfer):
                    result = download_fp8.download_file_curl('model.safetensors', 'https://example.invalid', expected_size)
                self.assertEqual(result, complete)
                final = Path(destination, 'model.safetensors')
                self.assertEqual(final.exists(), complete)
                self.assertEqual((final if complete else Path(str(final) + '.part')).read_bytes(), payload)
