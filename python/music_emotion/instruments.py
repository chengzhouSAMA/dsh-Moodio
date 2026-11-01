# -*- coding: utf-8 -*-
"""本地 CLAP 多标签乐器候选识别；相似度不冒充分类概率。"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .instrument_catalog import INSTRUMENTS, OTHER_SOUNDS, AST_DIRECT_LABELS, AST_FAMILY_LABELS, catalog, prompts

MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
WEIGHTS_SHA256 = "1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb"
MODEL_SR = 48000
WINDOW_SEC = 10.0
MIN_SIMILARITY = 0.22
MIN_BACKGROUND_MARGIN = 0.01
AMBIGUITY_MARGIN = 0.025
AST_SUPPORT = 0.2
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
AST_SHA256 = "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d"
AST_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
MODEL_FILES = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.json", "merges.txt", "pytorch_model.bin"]


def model_directory(model_dir: Optional[str] = None) -> Path:
    return Path(model_dir or os.environ.get("MOODIO_INSTRUMENT_MODEL_DIR") or
                Path.home() / ".cache" / "moodio" / "clap-htsat-unfused").expanduser().resolve()


def model_status(model_dir: Optional[str] = None) -> Dict[str, Any]:
    directory = model_directory(model_dir)
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    return {"available": not missing, "model_dir": str(directory), "missing_files": missing,
            "verifier_available": all((directory / "ast" / name).is_file() for name in AST_FILES),
            "verifier_model_id": AST_MODEL_ID, "verifier_revision": AST_REVISION,
            "verifier_license": "BSD-3-Clause",
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0",
            "target_count": len(INSTRUMENTS), "catalog": catalog()}


def setup_model(model_dir: Optional[str] = None, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """只在显式 setup 时下载；固定版本并校验权重摘要，分析时禁止自动联网下载。"""
    from huggingface_hub import snapshot_download

    directory = model_directory(model_dir)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(directory),
                      allow_patterns=MODEL_FILES + ["README.md"], endpoint=endpoint,
                      max_workers=2, token=False)
    digest = hashlib.sha256()
    with (directory / "pytorch_model.bin").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("instrument model checksum mismatch; do not load these weights")
    snapshot_download(AST_MODEL_ID, revision=AST_REVISION, local_dir=str(directory / "ast"),
                      allow_patterns=AST_FILES + ["README.md"], endpoint=endpoint, max_workers=2, token=False)
    verifier_digest = hashlib.sha256()
    with (directory / "ast" / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            verifier_digest.update(block)
    if verifier_digest.hexdigest() != AST_SHA256:
        raise ValueError("instrument verifier checksum mismatch; do not load these weights")
    return {**model_status(str(directory)), "weights_sha256": digest.hexdigest(),
            "verifier_sha256": verifier_digest.hexdigest(), "verified": True}


class ClapRecognizer:
    def __init__(self, model_dir: str, device: str):
        import torch
        from transformers import ClapModel, ClapProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use cpu or auto")
        self.device = device
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        self.processor = ClapProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = ClapModel.from_pretrained(model_dir, local_files_only=True,
                                               weights_only=True).to(device).eval()
        with torch.inference_mode():
            tokens = self.processor(text=prompts() + OTHER_SOUNDS, return_tensors="pt", padding=True)
            embeddings = self.model.get_text_features(**{k: v.to(device) for k, v in tokens.items()})
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            targets = embeddings[:len(INSTRUMENTS) * 2].reshape(len(INSTRUMENTS), 2, -1).mean(dim=1)
            self.text_features = torch.cat([torch.nn.functional.normalize(targets, dim=-1),
                                            embeddings[len(INSTRUMENTS) * 2:]], dim=0)
        self.verifier = None
        verifier_dir = Path(model_dir) / "ast"
        if all((verifier_dir / name).is_file() for name in AST_FILES):
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            self.verifier_processor = ASTFeatureExtractor.from_pretrained(str(verifier_dir), local_files_only=True)
            self.verifier = ASTForAudioClassification.from_pretrained(str(verifier_dir), local_files_only=True,
                                                                      use_safetensors=True).to(device).eval()

    def recognize(self, waveform: np.ndarray) -> Dict[str, Any]:
        import torch

        with torch.inference_mode():
            inputs = self.processor(audio=waveform, sampling_rate=MODEL_SR, return_tensors="pt")
            features = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
            features = torch.nn.functional.normalize(features, dim=-1)
            scores = (features @ self.text_features.T)[0].detach().cpu().numpy()
            tags = None
            if self.verifier is not None:
                import librosa
                audio16 = librosa.resample(waveform, orig_sr=MODEL_SR, target_sr=16000)
                features16 = self.verifier_processor(audio16, sampling_rate=16000, return_tensors="pt")
                values = torch.sigmoid(self.verifier(**{k: v.to(self.device) for k, v in features16.items()}).logits)[0]
                tags = {self.verifier.config.id2label[i]: float(value) for i, value in enumerate(values.cpu())}
            return fuse_instrument_scores(scores, tags)


def fuse_instrument_scores(scores: np.ndarray, tags: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if tags is not None and any(not np.isfinite(value) or not 0 <= value <= 1 for value in tags.values()):
        raise ValueError("invalid verifier tag scores")
    decision = classify_similarities(scores)
    if not decision["candidates"]:
        return decision

    def support(identifier):
        labels = AST_DIRECT_LABELS.get(identifier, AST_FAMILY_LABELS.get(identifier, []))
        scope = "direct_label" if identifier in AST_DIRECT_LABELS else "family_only" if labels else "unsupported"
        return {"scope": scope if tags is not None else "unavailable", "labels": labels,
                "tag_score": round(max((tags.get(label, 0.0) for label in labels), default=0.0), 4) if tags is not None and labels else None}

    primary = decision["candidates"][0]
    primary["verification"] = support(primary["id"])
    if primary["verification"]["scope"] == "direct_label" and primary["verification"]["tag_score"] < AST_SUPPORT:
        primary["status"] = "uncorroborated"
        decision["status"] = "ambiguous"
    # 伴奏需要独立的直接类别支持；音色家族标签不能证明某件中国乐器同时存在。
    if tags is not None:
        for index in np.argsort(-scores[:len(INSTRUMENTS)]):
            identifier, name, family, _ = INSTRUMENTS[int(index)]
            if identifier == primary["id"] or identifier not in AST_DIRECT_LABELS:
                continue
            if family == primary["family"] and primary["verification"]["scope"] != "direct_label":
                continue
            verified = support(identifier)
            if (verified["tag_score"] >= AST_SUPPORT and scores[index] >= MIN_SIMILARITY and
                    scores[index] - decision["background_similarity"] >= MIN_BACKGROUND_MARGIN and
                    decision["best_target_similarity"] - scores[index] <= 0.15):
                decision["candidates"].append({"id": identifier, "name": name, "family": family,
                    "similarity": round(float(scores[index]), 4), "status": "candidate", "role": "cooccurring_candidate",
                    "verification": verified, "confusable_with": []})
            if len(decision["candidates"]) >= 4:
                break
    return decision


@lru_cache(maxsize=2)
def _recognizer(model_dir: str, device: str):
    return ClapRecognizer(model_dir, device)


def classify_similarities(scores: np.ndarray) -> Dict[str, Any]:
    """保留多标签候选与近似音色歧义；未知声音不能被强行塞进 21 类。"""
    if len(scores) != len(INSTRUMENTS) + len(OTHER_SOUNDS) or not np.all(np.isfinite(scores)):
        raise ValueError("invalid instrument similarity vector")
    background = float(np.max(scores[len(INSTRUMENTS):]))
    ranking = np.argsort(-scores[:len(INSTRUMENTS)])
    best = float(scores[ranking[0]])
    selected = [int(ranking[0])] if best >= MIN_SIMILARITY and best - background >= MIN_BACKGROUND_MARGIN else []
    alternatives = [int(i) for i in ranking[1:3] if scores[i] >= MIN_SIMILARITY and
                    scores[i] - background >= MIN_BACKGROUND_MARGIN and best - scores[i] < AMBIGUITY_MARGIN]
    candidates = []
    for index in selected:
        identifier, name, family, _ = INSTRUMENTS[index]
        close = [INSTRUMENTS[j][0] for j in alternatives]
        candidates.append({"id": identifier, "name": name, "family": family,
                           "similarity": round(float(scores[index]), 4),
                           "status": "ambiguous" if close else "candidate", "role": "dominant_candidate",
                           "confusable_with": close})
    return {"status": ("ambiguous" if any(c["confusable_with"] for c in candidates) else "candidate")
                      if candidates else "unknown",
            "candidates": candidates, "alternatives": [{"id": INSTRUMENTS[i][0], "name": INSTRUMENTS[i][1],
                "similarity": round(float(scores[i]), 4)} for i in alternatives] if candidates else [],
            "background_similarity": round(background, 4),
            "best_target_similarity": round(best, 4)}


def analyze_instruments(y: np.ndarray, sr: int, model_dir: Optional[str] = None,
                        device: str = "auto") -> Dict[str, Any]:
    duration = len(y) / sr
    status = model_status(model_dir)
    base = {"status": "unavailable", "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "target_count": len(INSTRUMENTS), "catalog": catalog(), "seconds_analyzed": 0.0,
            "audio_duration_sec": round(duration, 3), "candidates": [], "segments": [],
            "score_type": "cosine_similarity_not_probability",
            "thresholds": {"min_similarity": MIN_SIMILARITY, "background_margin": MIN_BACKGROUND_MARGIN,
                           "ast_support": AST_SUPPORT, "calibrated": False},
            "verifier_model_id": AST_MODEL_ID,
            "verifier_revision": AST_REVISION,
            "verifier_available": status.get("verifier_available", False),
            "window_sec": WINDOW_SEC,
            "support_definition": "window_support_not_exact_instrument_presence_duration",
            "limitation": "21 类为零样本候选范围，尚未逐类验证准确率；合奏、相似音色与合成音色可能混淆。"}
    if not status["available"]:
        return {**base, "reason": "model_missing", "setup_command": "moodio instruments setup",
                "model_dir": status["model_dir"]}
    try:
        import librosa

        engine = _recognizer(status["model_dir"], device)
        window_samples = int(WINDOW_SEC * sr)
        aggregate: Dict[str, Dict[str, Any]] = {}
        known_seconds = 0.0
        for offset in range(0, len(y), window_samples):
            chunk = y[offset:offset + window_samples]
            start, end = offset / sr, min(len(y), offset + window_samples) / sr
            if len(chunk) < sr * 0.25 or float(np.sqrt(np.mean(chunk ** 2))) < 1e-5:
                decision = {"status": "insufficient_audio", "candidates": []}
            else:
                audio = librosa.resample(chunk, orig_sr=sr, target_sr=MODEL_SR) if sr != MODEL_SR else chunk
                decision = engine.recognize(audio)
            base["segments"].append({"start_sec": round(start, 3), "end_sec": round(end, 3), **decision})
            if decision["candidates"]:
                known_seconds += end - start
            for candidate in decision["candidates"]:
                item = aggregate.setdefault(candidate["id"], {
                    "id": candidate["id"], "name": candidate["name"], "family": candidate["family"],
                    "supported_seconds": 0.0, "weighted_similarity": 0.0, "max_similarity": -1.0,
                    "corroborated_seconds": 0.0, "uncorroborated_seconds": 0.0, "ambiguous_seconds": 0.0,
                    "weighted_tag_score": 0.0, "tag_seconds": 0.0,
                    "status": "candidate", "evidence": [],
                })
                item["supported_seconds"] += end - start
                item["weighted_similarity"] += candidate["similarity"] * (end - start)
                item["max_similarity"] = max(item["max_similarity"], candidate["similarity"])
                if candidate["status"] in ("ambiguous", "uncorroborated"):
                    item[candidate["status"] + "_seconds"] += end - start
                item["verification_scope"] = candidate.get("verification", {}).get("scope", "unavailable")
                tag_score = candidate.get("verification", {}).get("tag_score")
                if tag_score is not None:
                    item["tag_seconds"] += end - start
                    item["weighted_tag_score"] += tag_score * (end - start)
                    if tag_score >= AST_SUPPORT:
                        item["corroborated_seconds"] += end - start
                item["evidence"].append({"start_sec": round(start, 3), "end_sec": round(end, 3),
                                         "similarity": candidate["similarity"], "status": candidate["status"],
                                         "verifier_tag_score": tag_score})
        for item in aggregate.values():
            item["mean_similarity"] = round(item.pop("weighted_similarity") / item["supported_seconds"], 4)
            item["supported_ratio"] = round(item["supported_seconds"] / max(duration, 1e-9), 4)
            item["corroborated_ratio"] = round(item["corroborated_seconds"] / item["supported_seconds"], 4)
            tag_seconds = item.pop("tag_seconds")
            weighted_tag = item.pop("weighted_tag_score")
            item["mean_verifier_tag_score"] = round(weighted_tag / tag_seconds, 4) if tag_seconds else None
            if item["uncorroborated_seconds"]:
                item["status"] = "mixed_evidence" if item["corroborated_seconds"] else "uncorroborated"
            elif item["ambiguous_seconds"]:
                item["status"] = "ambiguous"
            for key in ("corroborated_seconds", "uncorroborated_seconds", "ambiguous_seconds"):
                item[key] = round(item[key], 3)
            item["supported_seconds"] = round(item["supported_seconds"], 3)
        base["candidates"] = sorted(aggregate.values(), key=lambda item: (-item["supported_ratio"], -item["mean_similarity"]))
        base.update({"status": "candidate" if aggregate else "unknown", "seconds_analyzed": round(duration, 3),
                     "candidate_coverage": round(known_seconds / max(duration, 1e-9), 4), "device": engine.device})
        return base
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        return {**base, "status": "unavailable", "reason": "inference_failed", "segments": [], "candidates": [],
                "error": f"{type(exc).__name__}: {exc}", "setup_command": "moodio instruments setup"}
