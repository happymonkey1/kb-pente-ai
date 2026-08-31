from __future__ import annotations

import unittest

from src.monitoring.__main__ import _is_loopback


class MonitoringMainTest(unittest.TestCase):
    def test_loopback_detection_rejects_remote_addresses(self) -> None:
        self.assertTrue(_is_loopback("127.0.0.1"))
        self.assertTrue(_is_loopback("::1"))
        self.assertTrue(_is_loopback("localhost"))
        self.assertFalse(_is_loopback("0.0.0.0"))
        self.assertFalse(_is_loopback("192.0.2.1"))
        self.assertFalse(_is_loopback("monitor.example.com"))


if __name__ == "__main__":
    unittest.main()
