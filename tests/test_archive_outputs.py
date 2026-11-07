"""归档辅助脚本：归档目录必须带时间戳后缀，不能与活跃的 `output/<配置名>/` 同名。

同名旧目录留在归档里是上一轮产物写错位置的前提（NFS 目录项缓存把写入解析到已移动的
旧 dentry），所以判别点是：归档名永远不是裸配置名，重复归档会加序号而不覆盖。
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from archive_outputs import archive_one  # noqa: E402


class ArchiveOneTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name) / "output"
        self.output.mkdir(parents=True)
        self.run = self.output / "appsolver_v2_random_seed0_isoflops"
        self.run.mkdir()
        (self.run / "metrics_best.json").write_text("{}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_archived_name_has_timestamp_suffix(self):
        source, target = archive_one("appsolver_v2_random_seed0_isoflops", "stale",
                                     self.output, "20261106_090000")
        self.assertEqual(source, self.run)
        self.assertEqual(target, self.output / "archive" / "stale"
                         / "appsolver_v2_random_seed0_isoflops__archived_20261106_090000")
        self.assertFalse(self.run.exists())
        self.assertTrue((target / "metrics_best.json").is_file())

    def test_repeated_archive_does_not_overwrite(self):
        self.run.mkdir(exist_ok=True)
        first, _ = archive_one(self.run, "stale", self.output, "20261106_090000")
        self.run.mkdir(parents=True)
        (self.run / "metrics_best.json").write_text("{}")
        second, target = archive_one(self.run, "stale", self.output, "20261106_090000")
        self.assertEqual(second, self.run)
        self.assertEqual(target.name, "appsolver_v2_random_seed0_isoflops__archived_20261106_090000_2")
        self.assertTrue(target.is_dir())

    def test_dry_run_keeps_source(self):
        source, target = archive_one("appsolver_v2_random_seed0_isoflops", "stale",
                                     self.output, "20261106_090000", dry_run=True)
        self.assertTrue(source.is_dir())
        self.assertFalse(target.exists())

    def test_missing_run_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            archive_one("no_such_run", "stale", self.output, "20261106_090000")


if __name__ == "__main__":
    unittest.main()
