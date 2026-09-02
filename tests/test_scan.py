import json
import tempfile
import unittest
from pathlib import Path

from mlx_edge.scan import context_window, list_models, scan_dirs, slug_model_id


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

    def test_embedding_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_model(root / "mlx-community" / "Qwen3-Embedding-0.6B-4bit")
            path = root / "BAAI" / "bge-small-en-v1.5-mlx"
            path.mkdir(parents=True)
            (path / "config.json").write_text(
                json.dumps({"model_type": "bert", "architectures": ["BertModel"]}),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"w")
            _write_model(root / "mlx-community" / "Qwen3-8B-4bit")
            models = {m["repo"]: m for m in list_models(str(root))}
            self.assertEqual(models["mlx-community/Qwen3-Embedding-0.6B-4bit"]["engine"], "embed")
            self.assertEqual(models["BAAI/bge-small-en-v1.5-mlx"]["engine"], "embed")
            self.assertEqual(models["mlx-community/Qwen3-8B-4bit"]["engine"], "lm")

    def test_tts_and_stt_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tts = root / "mlx-community" / "Kokoro-82M"
            tts.mkdir(parents=True)
            (tts / "config.json").write_text(
                json.dumps({"model_type": "kokoro", "architectures": ["KokoroModel"]}),
                encoding="utf-8",
            )
            (tts / "model.safetensors").write_bytes(b"w")
            stt = root / "mlx-community" / "whisper-tiny-mlx"
            stt.mkdir(parents=True)
            (stt / "config.json").write_text(
                json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}),
                encoding="utf-8",
            )
            (stt / "model.safetensors").write_bytes(b"w")
            omni = root / "mlx-community" / "Qwen2-Audio-7B"
            omni.mkdir(parents=True)
            (omni / "config.json").write_text(
                json.dumps({
                    "model_type": "qwen2_audio",
                    "audio_config": {"hidden_size": 16},
                    "architectures": ["Qwen2AudioForConditionalGeneration"],
                }),
                encoding="utf-8",
            )
            (omni / "model.safetensors").write_bytes(b"w")
            models = {m["repo"]: m for m in list_models(str(root))}
            self.assertEqual(models["mlx-community/Kokoro-82M"]["engine"], "tts")
            self.assertEqual(models["mlx-community/whisper-tiny-mlx"]["engine"], "stt")
            self.assertEqual(models["mlx-community/Qwen2-Audio-7B"]["engine"], "vlm")

    def test_reads_context_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "org" / "Big-32k"
            path.mkdir(parents=True)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "max_position_embeddings": 32768}),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"w")
            models = list_models(str(root))
            self.assertEqual(models[0]["context"], 32768)
            self.assertFalse(models[0]["hasChatTemplate"])
            self.assertEqual(context_window(path), 32768)
            self.assertIsNone(context_window(path / "missing"))

    def test_detects_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "org" / "MiniMax-M2.7-ConfigI-MLX"
            _write_model(path)
            models = list_models(str(root))
            self.assertFalse(models[0]["hasChatTemplate"])
            (path / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "<|start|>{{ message }}<|end|>"}),
                encoding="utf-8",
            )
            models = list_models(str(root))
            self.assertTrue(models[0]["hasChatTemplate"])

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

    def test_minimax_m3_vl_scans_as_lm(self):
        """minimax_m3_vl looks VL but runs text-only on patched mlx-lm."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "thetom-ai" / "MiniMax-M3-ConfigI-MLX"
            path.mkdir(parents=True)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "minimax_m3_vl",
                        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"w")
            models = list_models(str(root))
            self.assertEqual(models[0]["engine"], "lm")
            self.assertEqual(models[0]["repo"], "thetom-ai/MiniMax-M3-ConfigI-MLX")

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
