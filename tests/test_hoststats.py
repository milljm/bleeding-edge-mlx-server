import unittest

from mlx_edge.hoststats import parse_ioreg_gpu, parse_meminfo, parse_vm_stat, snapshot


VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                1000.
Pages active:                              2000.
Pages inactive:                            3000.
Pages speculative:                          100.
Pages wired down:                          4000.
Pages occupied by compressor:              500.
"""

MEMINFO = """\
MemTotal:       32768000 kB
MemFree:         4096000 kB
MemAvailable:   12288000 kB
Buffers:          102400 kB
Cached:          2048000 kB
"""

IOREG = """\
+-o AGXAccelerator  <class AGXAccelerator, id 0x1>
  | {
  |   "PerformanceStatistics" = {"Device Utilization %"=37,"Renderer Utilization %"=12}
  | }
"""


class HostStatsTests(unittest.TestCase):
    def test_vm_stat_used_is_active_wired_compressor(self):
        total = 128 * 1024**3
        row = parse_vm_stat(VM_STAT, total)
        page = 16384
        used = (2000 + 4000 + 500) * page
        self.assertEqual(row["used_bytes"], used)
        self.assertEqual(row["total_bytes"], total)
        self.assertGreater(row["ratio"], 0)

    def test_meminfo_uses_available(self):
        row = parse_meminfo(MEMINFO)
        self.assertEqual(row["total_bytes"], 32768000 * 1024)
        self.assertEqual(row["used_bytes"], (32768000 - 12288000) * 1024)

    def test_ioreg_picks_device_utilization(self):
        gpu = parse_ioreg_gpu(IOREG)
        self.assertIsNotNone(gpu)
        assert gpu is not None
        self.assertEqual(gpu["percent"], 37)
        self.assertEqual(gpu["source"], "ioreg")

    def test_ioreg_empty(self):
        self.assertIsNone(parse_ioreg_gpu("no stats here"))

    def test_snapshot_shape(self):
        snap = snapshot()
        self.assertEqual(snap["object"], "edge.host")
        self.assertIn("used_bytes", snap["memory"])
        self.assertIn("total_bytes", snap["memory"])
        self.assertGreater(snap["memory"]["total_bytes"], 0)
        gpu = snap["gpu"]
        if gpu is not None:
            self.assertIn("percent", gpu)
            self.assertGreaterEqual(gpu["percent"], 0)
            self.assertLessEqual(gpu["percent"], 100)
