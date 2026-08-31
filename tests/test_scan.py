import json
import tempfile
import unittest
from pathlib import Path

from mlx_edge.scan import list_models, scan_dirs, slug_model_id


def _write_model(path: Path, *, vision: bool = False, bits: int | None = 4, nbytes: int = 2048) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "model_type": "qwen2_vl" if vision else "qwen3",
        "architectures": ["Qwen2VLForConditionalGeneration"] if vision else ["Qwen3ForCausalLM"],
    }
    if vision:
        cfg["vision_config"] = {"hidden_size": 16}
    if bits:
        cfg["quantization"] = {"bits": bits}
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"w" * nbytes)


class ScanTests(unittest.TestCase):
    def test_org_name_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_model(root / "mlx-community" / "Qwen3-8B-4bit", nbytes=4096)
            _write_model(
                root / "mlx-community" / "Qwen2.5-VL-7B-Instruct-4bit",
                vision=True,
                nbytes=8192,
            )
            (root / "mlx-community" / "not-a-model").mkdir()
            (root / "mlx-community" / "not-a-model" / "config.json").write_text("{}", encoding="utf-8")
            models = list_models(str(root))
            repos = {m["repo"]: m for m in models}
            self.assertEqual(set(repos), {
                "mlx-community/Qwen3-8B-4bit",
                "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            })
            self.assertEqual(repos["mlx-community/Qwen3-8B-4bit"]["engine"], "lm")
            self.assertEqual(repos["mlx-community/Qwen2.5-VL-7B-Instruct-4bit"]["engine"], "vlm")
            self.assertEqual(repos["mlx-community/Qwen3-8B-4bit"]["quant"], "4-bit")
            self.assertEqual(repos["mlx-community/Qwen3-8B-4bit"]["source"], "scan")
            self.assertTrue(repos["mlx-community/Qwen3-8B-4bit"]["path"].endswith("Qwen3-8B-4bit"))
            self.assertEqual(repos["mlx-community/Qwen3-8B-4bit"]["id"], "lm-mlx-community-qwen3-8b-4bit")

    def test_hub_cache_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "models--mlx-community--Llama-3.2-3B-Instruct-4bit"
            snap = hub / "snapshots" / "abc123def"
            _write_model(snap, bits=4)
            (hub / "refs").mkdir()
            (hub / "refs" / "main").write_text("abc123def", encoding="utf-8")
            (hub / "blobs").mkdir()
            models = list_models(str(Path(tmp)))
            self.assertEqual(len(models), 1)
            rec = models[0]
            self.assertEqual(rec["repo"], "mlx-community/Llama-3.2-3B-Instruct-4bit")
            self.assertEqual(rec["path"], str(snap))
            self.assertEqual(rec["engine"], "lm")

    def test_single_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Qwen3-8B-4bit"
            _write_model(path)
            models = list_models(str(path))
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["repo"], "Qwen3-8B-4bit")

    def test_missing_dir(self):
        result = scan_dirs(["/definitely/missing/mlx-models"])
        self.assertEqual(result["models"], [])
        self.assertEqual(result["errors"][0]["message"], "not found")

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = scan_dirs([tmp])
            self.assertEqual(result["models"], [])
            self.assertEqual(result["errors"], [])

    def test_slug(self):
        self.assertEqual(slug_model_id("lm", "mlx-community/Qwen3-8B-4bit"), "lm-mlx-community-qwen3-8b-4bit")


if __name__ == "__main__":
    unittest.main()
