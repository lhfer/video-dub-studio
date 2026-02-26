#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Mapping
from urllib.parse import urlparse

import numpy as np
import requests
import dashscope
from dashscope.audio.asr import Recognition, Transcription
from dashscope import MultiModalConversation
from dashscope.utils.oss_utils import upload_file
from openai import OpenAI

try:
    import certifi
except Exception:  # pragma: no cover - optional runtime fallback
    certifi = None

try:
    import truststore
except Exception:  # pragma: no cover - optional runtime fallback
    truststore = None


@dataclass
class Segment:
    idx: int
    start: float
    end: float
    speaker: str
    text_src: str
    text_zh: str = ""
    voice: str = ""
    audio_path: str = ""


@dataclass
class SpeakerProfile:
    gender: str = "unknown"
    style: str = "neutral"


DEFAULT_TTS_VOICES = ["Cherry", "Ethan", "Serena", "Chelsie", "Sunny", "Dylan"]
# Keep TTS chunks conservative to reduce model-side length rejections on
# different languages/encodings and runtime environments.
MAX_TTS_INPUT_CHARS = 220
MAX_TTS_INPUT_BYTES = 300
TLS_MODE_AUTO = "auto"
TLS_MODE_SYSTEM = "system"
TLS_MODE_CERTIFI = "certifi"
TLS_MODE_CUSTOM_CA = "custom_ca"
TLS_MODE_DEFAULT = "default"
VOICE_GENDER_HINT = {
    "Ethan": "male",
    "Dylan": "male",
    "Sunny": "male",
    "Cherry": "female",
    "Serena": "female",
    "Chelsie": "female",
    "Bella": "female",
}
OUTPUT_LANGUAGE_MAP = {
    "zh-CN": ("Simplified Chinese", "Chinese"),
    "en-US": ("English", "English"),
    "ja-JP": ("Japanese", "Japanese"),
    "es-ES": ("Spanish", "Spanish"),
    "fr-FR": ("French", "French"),
}

_TLS_MODE = "default"
_TLS_EFFECTIVE_CA_FILE = ""
_TLS_CONFIGURED = False
_TLS_ACTIVE_KEY: tuple[str, str] = ("", "")
_DASHSCOPE_CERTIFI_ORIG_MODULE = None


class _DashscopeCertifiProxy:
    def __init__(self, real_module, where_func):
        self._real = real_module
        self.where = where_func

    def __getattr__(self, item):
        return getattr(self._real, item)


def _patch_dashscope_certifi_where(mode_norm: str, ca_file: str = "") -> None:
    global _DASHSCOPE_CERTIFI_ORIG_MODULE
    try:
        from dashscope.api_entities import http_request as dashscope_http_request
    except Exception:
        return
    current_mod = getattr(dashscope_http_request, "certifi", None)
    if current_mod is None:
        return

    if _DASHSCOPE_CERTIFI_ORIG_MODULE is None:
        _DASHSCOPE_CERTIFI_ORIG_MODULE = current_mod
    real_mod = _DASHSCOPE_CERTIFI_ORIG_MODULE

    if mode_norm in {TLS_MODE_CERTIFI, TLS_MODE_CUSTOM_CA} and ca_file:
        dashscope_http_request.certifi = _DashscopeCertifiProxy(real_mod, lambda p=ca_file: p)
        return

    if mode_norm in {TLS_MODE_SYSTEM, TLS_MODE_DEFAULT}:
        # Let ssl.create_default_context(cafile=None) use system trust store.
        dashscope_http_request.certifi = _DashscopeCertifiProxy(real_mod, lambda: None)
        return

    dashscope_http_request.certifi = real_mod


def _norm_tls_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {TLS_MODE_AUTO, TLS_MODE_SYSTEM, TLS_MODE_CERTIFI, TLS_MODE_CUSTOM_CA, TLS_MODE_DEFAULT}:
        return m
    return TLS_MODE_AUTO


def _clear_tls_bundle_env() -> None:
    for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.pop(k, None)


def _build_custom_ca_bundle(ca_bundle_file: str) -> str:
    src = Path(ca_bundle_file).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise RuntimeError(f"TLS 自定义 CA 文件不存在: {src}")
    try:
        src_text = src.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"TLS 自定义 CA 文件无法读取为 UTF-8 文本（请使用 PEM 格式）: {src}") from e

    # Merge custom CA with certifi roots so both enterprise MITM certs and public roots can work.
    if certifi is None:
        return str(src)
    certifi_file = Path(certifi.where())
    if not certifi_file.exists():
        return str(src)
    merged = Path(tempfile.gettempdir()) / "video_dub_studio_merged_ca.pem"
    merged.write_text(certifi_file.read_text(encoding="utf-8") + "\n" + src_text, encoding="utf-8")
    return str(merged)


def _apply_ca_bundle(ca_file: str) -> None:
    os.environ["SSL_CERT_FILE"] = ca_file
    os.environ["REQUESTS_CA_BUNDLE"] = ca_file
    os.environ["CURL_CA_BUNDLE"] = ca_file


def configure_tls_trust(mode: Optional[str] = None, ca_bundle_file: Optional[str] = None) -> str:
    global _TLS_MODE, _TLS_EFFECTIVE_CA_FILE, _TLS_CONFIGURED, _TLS_ACTIVE_KEY

    if mode is None:
        if _TLS_CONFIGURED:
            return _TLS_MODE
        mode_norm = TLS_MODE_AUTO
    else:
        mode_norm = _norm_tls_mode(mode)
    ca_norm = str(ca_bundle_file or "").strip()

    key = (mode_norm, ca_norm)
    if _TLS_CONFIGURED and key == _TLS_ACTIVE_KEY:
        return _TLS_MODE

    _clear_tls_bundle_env()
    _TLS_EFFECTIVE_CA_FILE = ""
    _TLS_ACTIVE_KEY = key

    if mode_norm == TLS_MODE_SYSTEM:
        if truststore is None:
            raise RuntimeError("TLS 模式 system 需要 truststore 依赖，但当前不可用")
        truststore.inject_into_ssl()
        _patch_dashscope_certifi_where(TLS_MODE_SYSTEM)
        _TLS_MODE = "system-truststore"
        _TLS_CONFIGURED = True
        return _TLS_MODE

    if mode_norm == TLS_MODE_CERTIFI:
        if certifi is None:
            raise RuntimeError("TLS 模式 certifi 需要 certifi 依赖，但当前不可用")
        ca_file = certifi.where()
        _apply_ca_bundle(ca_file)
        _TLS_EFFECTIVE_CA_FILE = ca_file
        _patch_dashscope_certifi_where(TLS_MODE_CERTIFI, ca_file)
        _TLS_MODE = "certifi-bundle"
        _TLS_CONFIGURED = True
        return _TLS_MODE

    if mode_norm == TLS_MODE_CUSTOM_CA:
        if not ca_norm:
            raise RuntimeError("TLS 模式 custom_ca 需要指定 CA 证书文件")
        ca_file = _build_custom_ca_bundle(ca_norm)
        _apply_ca_bundle(ca_file)
        _TLS_EFFECTIVE_CA_FILE = ca_file
        _patch_dashscope_certifi_where(TLS_MODE_CUSTOM_CA, ca_file)
        _TLS_MODE = f"custom-ca:{ca_file}"
        _TLS_CONFIGURED = True
        return _TLS_MODE

    if mode_norm == TLS_MODE_DEFAULT:
        _patch_dashscope_certifi_where(TLS_MODE_DEFAULT)
        _TLS_MODE = "python-default"
        _TLS_CONFIGURED = True
        return _TLS_MODE

    # AUTO: system trust store first, then certifi, then python default.
    if truststore is not None:
        try:
            truststore.inject_into_ssl()
            _patch_dashscope_certifi_where(TLS_MODE_SYSTEM)
            _TLS_MODE = "system-truststore"
            _TLS_CONFIGURED = True
            return _TLS_MODE
        except Exception:
            pass
    if certifi is not None:
        try:
            ca_file = certifi.where()
            _apply_ca_bundle(ca_file)
            _TLS_EFFECTIVE_CA_FILE = ca_file
            _patch_dashscope_certifi_where(TLS_MODE_CERTIFI, ca_file)
            _TLS_MODE = "certifi-bundle"
            _TLS_CONFIGURED = True
            return _TLS_MODE
        except Exception:
            pass

    _patch_dashscope_certifi_where(TLS_MODE_DEFAULT)
    _TLS_MODE = "python-default"
    _TLS_CONFIGURED = True
    return _TLS_MODE


def current_tls_mode() -> str:
    if not _TLS_CONFIGURED:
        return configure_tls_trust()
    return _TLS_MODE


def current_tls_verify_arg() -> object:
    if _TLS_EFFECTIVE_CA_FILE:
        return _TLS_EFFECTIVE_CA_FILE
    return True


def _is_ssl_cert_error(err: Exception) -> bool:
    text = str(err).lower()
    return (
        ("certificate verify failed" in text)
        or ("cert_verify_failed" in text)
        or ("sslcertverificationerror" in text)
        or ("unable to get local issuer certificate" in text)
        or ("self signed certificate" in text)
        or ("tlsv1 alert unknown ca" in text)
    )


def _compact_error(err: Exception, max_len: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(err or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def websocket_base_url_from_http_api(api_url: str) -> str:
    src = str(api_url or "").strip() or "https://dashscope.aliyuncs.com/api/v1"
    parsed = urlparse(src)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.netloc or "dashscope.aliyuncs.com"
    path = parsed.path or "/api/v1"
    m = re.search(r"/api/([^/]+)", path)
    version = m.group(1) if m else "v1"
    return f"{scheme}://{host}/api-ws/{version}/inference"


def _configure_dashscope_runtime(api_key: str, api_url: str) -> None:
    http_base = str(api_url or "").strip().rstrip("/")
    if not http_base:
        http_base = "https://dashscope.aliyuncs.com/api/v1"
    ws_base = websocket_base_url_from_http_api(http_base)

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key
    os.environ["DASHSCOPE_HTTP_BASE_URL"] = http_base
    os.environ["DASHSCOPE_WEBSOCKET_BASE_URL"] = ws_base
    # dashscope module loads these at import time; assign explicitly at runtime.
    try:
        dashscope.base_http_api_url = http_base
    except Exception:
        pass
    try:
        dashscope.base_websocket_api_url = ws_base
    except Exception:
        pass


def _speaker_label(raw: object) -> str:
    if isinstance(raw, int):
        return f"S{raw + 1}" if raw >= 0 else "S1"
    txt = str(raw or "").strip()
    if not txt:
        return "S1"
    if txt.upper().startswith("S"):
        return txt.upper()
    if txt.isdigit():
        try:
            return f"S{int(txt) + 1}"
        except Exception:
            return "S1"
    return "S1"


_TEXT_KEYS = ("text", "transcript", "content", "sentence")
_START_KEYS = ("begin_time", "start_time", "start", "begin", "offset", "begin_ms", "start_ms")
_END_KEYS = ("end_time", "stop_time", "end", "current_time", "end_ms", "stop_ms")


def _first_non_none(mapping: Mapping[str, object], keys: Sequence[str]) -> object:
    for k in keys:
        if k in mapping and mapping.get(k) is not None:
            return mapping.get(k)
    return None


def _first_text(mapping: Mapping[str, object]) -> str:
    for k in _TEXT_KEYS:
        if k in mapping:
            v = mapping.get(k)
            if isinstance(v, str):
                t = v.strip()
                if t:
                    return t
    return ""


def _extract_sentence_entries(payload: object) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _append(text: str, begin: object, end: object, speaker: object, has_time: bool) -> None:
        t = str(text or "").strip()
        if not t:
            return
        b_key = "" if begin is None else str(begin)
        e_key = "" if end is None else str(end)
        s_key = "" if speaker is None else str(speaker)
        key = (t, b_key, e_key, s_key)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "begin": begin,
                "end": end,
                "speaker": speaker,
                "text": t,
                "has_time": bool(has_time and begin is not None and end is not None),
            }
        )

    def _ingest_dict(d: Mapping[str, object], parent_speaker: object = None) -> None:
        spk = d.get("speaker_id", d.get("speaker", parent_speaker))
        text = _first_text(d)
        b = _first_non_none(d, _START_KEYS)
        e = _first_non_none(d, _END_KEYS)
        if text:
            _append(text, b, e, spk, has_time=(b is not None and e is not None))

    if isinstance(payload, dict):
        for container_key in ("transcripts", "sentences", "paragraphs", "utterances", "segments", "items", "chunks"):
            container = payload.get(container_key)
            if not isinstance(container, list):
                continue
            for item in container:
                if not isinstance(item, dict):
                    continue
                parent_speaker = item.get("speaker_id", item.get("speaker", None))
                _ingest_dict(item, parent_speaker=parent_speaker)
                nested_sentences = item.get("sentences")
                if isinstance(nested_sentences, list):
                    for s in nested_sentences:
                        if isinstance(s, dict):
                            _ingest_dict(s, parent_speaker=parent_speaker)

    def _scan(node: object, parent_speaker: object = None) -> None:
        if isinstance(node, dict):
            spk = node.get("speaker_id", node.get("speaker", parent_speaker))
            _ingest_dict(node, parent_speaker=spk)
            for v in node.values():
                _scan(v, parent_speaker=spk)
        elif isinstance(node, list):
            for item in node:
                _scan(item, parent_speaker=parent_speaker)

    _scan(payload)

    # Last-resort: preserve plain text even if no timestamps are returned.
    # Downstream normalize_segments() will rebuild timeline by text length.
    if not rows:
        text_blocks: List[str] = []

        def _collect_text(node: object) -> None:
            if isinstance(node, dict):
                txt = _first_text(node)
                if txt and txt not in text_blocks:
                    text_blocks.append(txt)
                for v in node.values():
                    _collect_text(v)
            elif isinstance(node, list):
                for item in node:
                    _collect_text(item)

        _collect_text(payload)
        for t in text_blocks:
            _append(t, 0.0, 0.0, "S1", has_time=False)
    return rows


def _entries_to_segments(entries: List[Dict[str, object]]) -> List[dict]:
    if not entries:
        return []
    raw_pairs: List[tuple[float, float]] = []
    norm_entries: List[Dict[str, object]] = []
    for item in entries:
        has_time = bool(item.get("has_time", True))
        if has_time:
            try:
                b = float(item.get("begin", 0.0))
                e = float(item.get("end", b))
            except Exception:
                b = 0.0
                e = 0.0
                has_time = False
        else:
            b = 0.0
            e = 0.0
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if has_time and e <= b:
            e = b + 0.2
        if has_time:
            raw_pairs.append((b, e))
        norm_entries.append({"begin": b, "end": e, "speaker": item.get("speaker"), "text": text, "has_time": has_time})
    if not norm_entries:
        return []

    diffs = [max(0.0, e - b) for b, e in raw_pairs]
    ends = [max(0.0, e) for _, e in raw_pairs]
    median_diff = float(np.median(diffs)) if diffs else 0.0
    median_end = float(np.median(ends)) if ends else 0.0
    use_ms = (median_diff > 80.0) or (median_end > 10000.0)
    scale = 1000.0 if use_ms else 1.0

    segments: List[dict] = []
    for item in norm_entries:
        has_time = bool(item.get("has_time", True))
        if has_time:
            start = float(item["begin"]) / scale
            end = float(item["end"]) / scale
            if end <= start:
                end = start + 0.2
        else:
            start = 0.0
            end = 0.0
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": _speaker_label(item.get("speaker")),
                "text": str(item["text"]),
            }
        )
    segments.sort(key=lambda x: float(x["start"]))
    return segments


def _fetch_transcription_payloads(output: object) -> List[object]:
    out = output if isinstance(output, dict) else {}
    payloads: List[object] = []
    top_level_url = str(
        out.get("transcription_url", out.get("result_url", out.get("url", out.get("file_url", ""))))
    ).strip()
    if top_level_url:
        resp = requests.get(top_level_url, timeout=120, verify=current_tls_verify_arg())
        resp.raise_for_status()
        payloads.append(resp.json())
        return payloads
    result_nodes: List[object] = []
    for key in ("results", "result", "task_results", "subtasks", "items"):
        v = out.get(key)
        if isinstance(v, list):
            result_nodes.extend(v)
        elif isinstance(v, dict):
            result_nodes.append(v)
    if result_nodes:
        for item in result_nodes:
            if not isinstance(item, dict):
                continue
            sub_status = str(item.get("subtask_status", item.get("task_status", ""))).strip().upper()
            if sub_status and sub_status not in {"SUCCEEDED", "SUCCESS", "DONE"}:
                continue
            t_url = str(
                item.get(
                    "transcription_url",
                    item.get("result_url", item.get("url", item.get("file_url", ""))),
                )
            ).strip()
            if t_url:
                resp = requests.get(t_url, timeout=120, verify=current_tls_verify_arg())
                resp.raise_for_status()
                payloads.append(resp.json())
            else:
                payloads.append(item)
    if not payloads:
        payloads.append(out)
    return payloads


def _asr_with_qwen_http_fallback(
    api_key: str,
    model: str,
    audio_path: str,
    api_url: str,
) -> dict:
    # Fallback path for machines where websocket TLS fails:
    # upload local file to OSS then call HTTP batch transcription.
    candidates: List[str] = []
    for m in [model, "fun-asr", "paraformer-v1"]:
        mm = str(m or "").strip()
        if mm and mm not in candidates:
            candidates.append(mm)

    file_uri = Path(audio_path).resolve().as_uri()
    last_error: Optional[Exception] = None
    for trans_model in candidates:
        try:
            oss_url = upload_file(model=trans_model, upload_path=file_uri, api_key=api_key)
            if not oss_url:
                raise RuntimeError(f"文件上传失败（model={trans_model}）")
            req_headers: Dict[str, str] = {}
            if str(oss_url).startswith("oss://"):
                # Official docs require explicit resolver header for oss:// URL in HTTP calls.
                req_headers["X-DashScope-OssResourceResolve"] = "enable"
            task = Transcription.async_call(
                model=trans_model,
                file_urls=[str(oss_url)],
                api_key=api_key,
                diarization_enabled=True,
                timestamp_alignment_enabled=True,
                headers=req_headers if req_headers else None,
            )
            if int(getattr(task, "status_code", 0)) != 200:
                raise RuntimeError(f"提交识别任务失败: {getattr(task, 'code', '')} {getattr(task, 'message', '')}")

            task_output = task.output if isinstance(task.output, dict) else {}
            task_id = str(task_output.get("task_id", "")).strip()
            if not task_id:
                raise RuntimeError("识别任务缺少 task_id")

            done = Transcription.wait(
                task=task_id,
                api_key=api_key,
                headers=req_headers if req_headers else None,
            )
            if int(getattr(done, "status_code", 0)) != 200:
                raise RuntimeError(f"查询识别结果失败: {getattr(done, 'code', '')} {getattr(done, 'message', '')}")

            output = done.output if isinstance(done.output, dict) else {}
            task_status = str(output.get("task_status", "")).strip().upper()
            if task_status in {"FAILED", "CANCELED", "UNKNOWN"}:
                err_code = str(output.get("code", getattr(done, "code", ""))).strip()
                err_msg = str(output.get("message", getattr(done, "message", ""))).strip()
                if not err_code or not err_msg:
                    results = output.get("results")
                    if isinstance(results, list) and results and isinstance(results[0], dict):
                        err_code = err_code or str(results[0].get("code", "")).strip()
                        err_msg = err_msg or str(results[0].get("message", "")).strip()
                raise RuntimeError(
                    f"文件识别任务失败（model={trans_model} task_id={task_id} status={task_status} "
                    f"code={err_code or '-'} message={err_msg or '-'})"
                )
            payloads = _fetch_transcription_payloads(output)
            segments: List[dict] = []
            for payload in payloads:
                entries = _extract_sentence_entries(payload)
                segments.extend(_entries_to_segments(entries))
            if segments:
                return {"speaker_profiles": {}, "segments": segments, "asr_backend": "http_transcription"}
            output_keys = list(output.keys())[:12] if isinstance(output, dict) else []
            payload_preview = ""
            try:
                payload_preview = json.dumps(payloads[0], ensure_ascii=False)[:600] if payloads else ""
            except Exception:
                payload_preview = str(payloads[0])[:600] if payloads else ""
            raise RuntimeError(
                f"文件识别返回为空（model={trans_model} task_id={task_id} output_keys={output_keys} preview={payload_preview}）"
            )
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError(f"HTTP 文件识别兜底失败: {last_error}") from last_error
    raise RuntimeError("HTTP 文件识别兜底失败: 未知错误")


def run(cmd: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required binary: {name}")


def ffprobe_duration(path: str) -> float:
    p = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    return float(p.stdout.strip())


def strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    return m.group(1).strip() if m else text


def download_youtube_audio(url: str, out_dir: Path) -> str:
    out_tmpl = str(out_dir / "source.%(ext)s")
    run(["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "-o", out_tmpl, url])
    candidates = sorted(out_dir.glob("source.*"))
    if not candidates:
        raise RuntimeError("yt-dlp did not produce an audio file")
    return str(candidates[-1])


def normalize_audio(src_path: str, dst_path: str, sample_rate: int = 16000) -> str:
    run(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", str(sample_rate), dst_path])
    return dst_path


def trim_audio(src_path: str, dst_path: str, max_seconds: float) -> str:
    run(["ffmpeg", "-y", "-i", src_path, "-t", f"{max_seconds:.3f}", dst_path])
    return dst_path


def asr_with_qwen(
    api_key: str,
    model: str,
    audio_path: str,
    api_url: str,
) -> dict:
    # Use dedicated file recognition API (paraformer) for robust long audio ASR.
    configure_tls_trust()
    _configure_dashscope_runtime(api_key=api_key, api_url=api_url)
    recog = Recognition(
        model=model,
        callback=None,
        format=Path(audio_path).suffix.lstrip(".") or "wav",
        sample_rate=16000,
    )
    try:
        result = recog.call(
            audio_path,
            diarization_enabled=True,
            timestamp_alignment_enabled=True,
        )
    except Exception as ws_err:
        try:
            return _asr_with_qwen_http_fallback(
                api_key=api_key,
                model=model,
                audio_path=audio_path,
                api_url=api_url,
            )
        except Exception as fallback_err:
            ws_tls = _is_ssl_cert_error(ws_err)
            fb_tls = _is_ssl_cert_error(fallback_err)
            ws_msg = _compact_error(ws_err)
            fb_msg = _compact_error(fallback_err)
            if ws_tls and fb_tls:
                raise RuntimeError(
                    "ASR failed: websocket 与 HTTP 兜底均发生 TLS 证书异常。"
                    f"websocket={ws_msg}; fallback={fb_msg}"
                ) from fallback_err
            if ws_tls:
                raise RuntimeError(
                    "ASR failed: websocket TLS 通道不可用，且 HTTP 文件识别兜底失败。"
                    f"fallback={fb_msg}"
                ) from fallback_err
            if fb_tls:
                raise RuntimeError(
                    "ASR failed: websocket 调用失败，且 HTTP 兜底发生 TLS 证书异常。"
                    f"websocket={ws_msg}; fallback={fb_msg}"
                ) from fallback_err
            raise RuntimeError(
                "ASR failed: websocket 调用失败，且 HTTP 文件识别兜底失败。"
                f"websocket={ws_msg}; fallback={fb_msg}"
            ) from fallback_err
    if result.status_code != 200:
        ws_err = RuntimeError(f"ASR failed: {result.code} {result.message}")
        try:
            return _asr_with_qwen_http_fallback(
                api_key=api_key,
                model=model,
                audio_path=audio_path,
                api_url=api_url,
            )
        except Exception as fallback_err:
            ws_msg = _compact_error(ws_err)
            fb_msg = _compact_error(fallback_err)
            raise RuntimeError(
                "ASR failed: websocket 返回非 200，且 HTTP 文件识别兜底失败。"
                f"websocket={ws_msg}; fallback={fb_msg}"
            ) from fallback_err
    sentences = result.get_sentence()
    if not isinstance(sentences, list) or not sentences:
        raise RuntimeError("ASR returned empty sentences")

    segments: List[dict] = []
    for s in sentences:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        begin_ms = float(s.get("begin_time", 0))
        end_ms = float(s.get("end_time", begin_ms + 300))
        spk_id = s.get("speaker_id", None)
        speaker = f"S{int(spk_id) + 1}" if isinstance(spk_id, int) and spk_id >= 0 else "S1"
        segments.append(
            {
                "start": begin_ms / 1000.0,
                "end": end_ms / 1000.0,
                "speaker": speaker,
                "text": text,
            }
        )
    if not segments:
        raise RuntimeError("ASR parsed no valid segments")
    return {"speaker_profiles": {}, "segments": segments, "asr_backend": "websocket_realtime"}


def normalize_segments(raw: List[dict], total_duration: float) -> List[Segment]:
    out: List[Segment] = []
    for i, s in enumerate(raw):
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        spk = str(s.get("speaker", "S1")).strip() or "S1"
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 0.2))
        except Exception:
            start, end = 0.0, 0.0
        out.append(Segment(idx=i, start=start, end=end, speaker=spk, text_src=text))

    if not out:
        raise RuntimeError("No valid speech segments")

    has_timeline = any(seg.end > seg.start for seg in out)
    if has_timeline:
        for seg in out:
            if seg.end <= seg.start:
                seg.end = seg.start + 0.2
        out.sort(key=lambda x: x.start)
        return out

    total_chars = sum(max(1, len(seg.text_src)) for seg in out)
    cursor = 0.0
    for seg in out:
        dur = total_duration * (max(1, len(seg.text_src)) / total_chars)
        seg.start = cursor
        seg.end = min(total_duration, cursor + max(0.3, dur))
        cursor = seg.end
    return out


def normalize_speaker_profiles(raw: dict) -> Dict[str, SpeakerProfile]:
    profiles: Dict[str, SpeakerProfile] = {}
    if not isinstance(raw, dict):
        return profiles
    src = raw.get("speaker_profiles", {})
    if not isinstance(src, dict):
        return profiles
    for spk, v in src.items():
        if not isinstance(v, dict):
            continue
        gender = str(v.get("gender", "unknown")).strip().lower()
        style = str(v.get("style", "neutral")).strip().lower()
        if gender not in {"male", "female"}:
            gender = "unknown"
        if style not in {"calm", "energetic", "serious", "neutral"}:
            style = "neutral"
        profiles[str(spk)] = SpeakerProfile(gender=gender, style=style)
    return profiles


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        buf = wf.readframes(n_frames)
    if sampwidth != 2:
        raise RuntimeError("Expected 16-bit PCM wav for speaker clustering")
    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, sr


def estimate_pitch_hz(x: np.ndarray, sr: int) -> float:
    if x.size < int(sr * 0.15):
        return 0.0
    x = x - float(np.mean(x))
    if np.max(np.abs(x)) < 1e-4:
        return 0.0
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    min_lag = max(1, int(sr / 300))
    max_lag = max(min_lag + 1, int(sr / 70))
    if max_lag >= ac.size:
        return 0.0
    ac[:min_lag] = 0
    lag = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
    peak = float(ac[lag])
    if peak <= 0:
        return 0.0
    return float(sr / lag)


def kmeans_fit_numpy(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 80,
    n_init: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32), np.empty((0, X.shape[1]), dtype=np.float32), 0.0
    n_clusters = max(1, min(n_clusters, n_samples))
    rng = np.random.default_rng(42)
    best_labels = np.zeros(n_samples, dtype=np.int32)
    best_centroids = np.zeros((n_clusters, X.shape[1]), dtype=np.float32)
    best_inertia = float("inf")
    for _ in range(max(1, n_init)):
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = X[init_idx].copy()
        labels = np.zeros(n_samples, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                pts = X[labels == k]
                if pts.size > 0:
                    centroids[k] = pts.mean(axis=0)
        dists = np.sum((X - centroids[labels]) ** 2, axis=1)
        inertia = float(np.sum(dists))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
    return best_labels, best_centroids, best_inertia


def kmeans_cluster_numpy(X: np.ndarray, n_clusters: int, max_iter: int = 80) -> np.ndarray:
    labels, _, _ = kmeans_fit_numpy(X, n_clusters=n_clusters, max_iter=max_iter, n_init=1)
    return labels


def _min_centroid_distance(centroids: np.ndarray) -> float:
    if centroids.shape[0] <= 1:
        return 0.0
    min_dist = float("inf")
    for i in range(centroids.shape[0]):
        for j in range(i + 1, centroids.shape[0]):
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            min_dist = min(min_dist, d)
    return 0.0 if min_dist == float("inf") else min_dist


def _cluster_duration_ratios(labels: np.ndarray, durations: np.ndarray, n_clusters: int) -> np.ndarray:
    total = float(np.sum(durations) + 1e-9)
    ratios = np.zeros(n_clusters, dtype=np.float32)
    for k in range(n_clusters):
        ratios[k] = float(np.sum(durations[labels == k]) / total)
    return ratios


def _silhouette_score(X: np.ndarray, labels: np.ndarray, max_samples: int = 500) -> float:
    n = X.shape[0]
    if n < 3:
        return 0.0
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    if n > max_samples:
        # Evenly subsample to control O(n^2) cost for long videos.
        idx = np.linspace(0, n - 1, max_samples).astype(np.int32)
        X = X[idx]
        labels = labels[idx]
        n = X.shape[0]
        uniq = np.unique(labels)
        if uniq.size < 2:
            return 0.0

    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2) + 1e-9).astype(np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same = labels == labels[i]
        same_count = int(np.sum(same))
        if same_count <= 1:
            continue
        a = float(np.sum(D[i, same]) / max(1, same_count - 1))
        b = float("inf")
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            if not np.any(other):
                continue
            b = min(b, float(np.mean(D[i, other])))
        if b == float("inf"):
            continue
        denom = max(a, b, 1e-6)
        scores[i] = float((b - a) / denom)
    return float(np.mean(scores))


def _choose_cluster_count(
    X: np.ndarray,
    durations: np.ndarray,
    max_speakers: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    max_k = max(1, min(max_speakers, 3, n_samples))
    fit_map: Dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    bic_map: Dict[int, float] = {}
    sil_map: Dict[int, float] = {}
    dim = X.shape[1]
    for k in range(1, max_k + 1):
        labels_k, centroids_k, inertia_k = kmeans_fit_numpy(X, n_clusters=k, max_iter=80, n_init=6 if k > 1 else 1)
        fit_map[k] = (labels_k, centroids_k, inertia_k)
        variance = max(inertia_k / max(1, n_samples), 1e-9)
        params = float(k * (dim + 1))
        complexity_penalty = 3.2
        bic_map[k] = float(n_samples * np.log(variance) + complexity_penalty * params * np.log(max(2, n_samples)))
        sil_map[k] = _silhouette_score(X, labels_k) if k > 1 else 0.0

    best_k = min(bic_map, key=bic_map.get)
    inertia_1 = fit_map[1][2]
    if best_k > 1:
        labels_k, centroids_k, inertia_k = fit_map[best_k]
        improve = (inertia_1 - inertia_k) / max(inertia_1, 1e-9)
        ratios = _cluster_duration_ratios(labels_k, durations, best_k)
        min_ratio = float(np.min(ratios))
        sep = _min_centroid_distance(centroids_k)
        compact = float(np.sqrt(max(inertia_k / max(1, n_samples), 1e-9)))
        sep_score = sep / max(compact, 1e-6)
        sil = sil_map.get(best_k, 0.0)
        min_improve = 0.34 if best_k == 2 else 0.45
        min_sep = 1.35 if best_k == 2 else 1.50
        min_ratio_req = 0.18 if best_k == 2 else 0.14
        min_sil = 0.20 if best_k == 2 else 0.25
        if improve < min_improve or min_ratio < min_ratio_req or sep_score < min_sep or sil < min_sil:
            best_k = 1
        elif best_k == 3 and 2 in bic_map and bic_map[3] > bic_map[2] - 0.35:
            best_k = 2

    best_labels, best_centroids, _ = fit_map[best_k]
    if best_k > 1:
        ratios = _cluster_duration_ratios(best_labels, durations, best_k)
        if float(np.max(ratios)) >= 0.90:
            labels1, centroids1, _ = fit_map[1]
            return 1, labels1, centroids1
    return best_k, best_labels, best_centroids


def _viterbi_smooth_labels(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    k = centroids.shape[0]
    if k <= 1 or X.shape[0] <= 2:
        return labels
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    sep = _min_centroid_distance(centroids)
    compact = float(np.sqrt(max(np.median(dists), 1e-9)))
    sep_score = sep / max(compact, 1e-6)
    base = float(max(np.median(dists), 0.05))
    if sep_score >= 2.0:
        transition_penalty = base * 0.08
    elif sep_score >= 1.4:
        transition_penalty = base * 0.18
    else:
        transition_penalty = base * 0.35
    n = X.shape[0]
    dp = np.zeros((n, k), dtype=np.float32)
    bt = np.zeros((n, k), dtype=np.int32)
    dp[0] = dists[0]
    for t in range(1, n):
        prev = dp[t - 1]
        for j in range(k):
            costs = prev + transition_penalty
            costs[j] = prev[j]
            best_prev = int(np.argmin(costs))
            bt[t, j] = best_prev
            dp[t, j] = float(dists[t, j] + costs[best_prev])
    out = np.zeros(n, dtype=np.int32)
    out[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        out[t - 1] = bt[t, out[t]]
    return out


def _run_duration(segments: List[Segment], idxs: List[int], start_pos: int, end_pos: int) -> float:
    if start_pos < 0 or end_pos < start_pos:
        return 0.0
    start_idx = idxs[start_pos]
    end_idx = idxs[end_pos]
    return float(max(0.0, segments[end_idx].end - segments[start_idx].start))


def _merge_short_label_runs(labels: np.ndarray, segments: List[Segment], idxs: List[int]) -> np.ndarray:
    if labels.size <= 1:
        return labels
    out = labels.copy()
    for _ in range(3):
        changed = False
        run_start = 0
        while run_start < out.size:
            run_label = int(out[run_start])
            run_end = run_start
            while run_end + 1 < out.size and int(out[run_end + 1]) == run_label:
                run_end += 1
            dur = _run_duration(segments, idxs, run_start, run_end)
            if out.size > 2:
                prev_label = int(out[run_start - 1]) if run_start > 0 else -1
                next_label = int(out[run_end + 1]) if run_end + 1 < out.size else -1
                target = -1
                # Primary smoothing rule: merge very short A-B-A spikes only.
                if dur < 0.55 and prev_label >= 0 and next_label >= 0 and prev_label == next_label:
                    target = prev_label
                # Edge-case: ultra-short head/tail blip.
                elif dur < 0.35 and prev_label >= 0 and next_label < 0:
                    target = prev_label
                elif dur < 0.35 and next_label >= 0 and prev_label < 0:
                    target = next_label
                if target >= 0 and target != run_label:
                    out[run_start : run_end + 1] = target
                    changed = True
            run_start = run_end + 1
        if not changed:
            break
    return out


def maybe_collapse_to_single_speaker(segments: List[Segment]) -> str:
    speakers = sorted({seg.speaker for seg in segments if seg.speaker})
    if len(speakers) <= 1:
        return ""
    total = 0.0
    per_speaker: Dict[str, float] = {}
    runs: List[tuple[str, float]] = []
    cur_spk = ""
    run_start = 0.0
    last_end = 0.0
    for seg in segments:
        dur = max(0.0, seg.end - seg.start)
        total += dur
        per_speaker[seg.speaker] = per_speaker.get(seg.speaker, 0.0) + dur
        if not cur_spk:
            cur_spk = seg.speaker
            run_start = seg.start
            last_end = seg.end
            continue
        if seg.speaker == cur_spk:
            last_end = max(last_end, seg.end)
            continue
        runs.append((cur_spk, max(0.0, last_end - run_start)))
        cur_spk = seg.speaker
        run_start = seg.start
        last_end = seg.end
    if cur_spk:
        runs.append((cur_spk, max(0.0, last_end - run_start)))
    if total <= 1e-6 or not runs:
        return ""

    dom_speaker = max(per_speaker.items(), key=lambda x: x[1])[0]
    dom_ratio = per_speaker[dom_speaker] / total
    switches = max(0, len(runs) - 1)
    switches_per_min = switches / max(total / 60.0, 1e-6)
    median_run = float(np.median([r[1] for r in runs]))

    if dom_ratio >= 0.82 and (median_run <= 1.2 or switches_per_min >= 14):
        for seg in segments:
            seg.speaker = dom_speaker
        return dom_speaker
    return ""


def infer_speakers_from_audio(
    segments: List[Segment],
    wav_path: str,
    speaker_count: int,
) -> tuple[Dict[int, str], Dict[str, SpeakerProfile]]:
    signal, sr = load_wav_mono(wav_path)
    feats: List[np.ndarray] = []
    idxs: List[int] = []
    pitches: List[float] = []
    energies: List[float] = []
    rates: List[float] = []
    durations: List[float] = []
    for i, seg in enumerate(segments):
        s = max(0, int(seg.start * sr))
        e = min(signal.size, int(seg.end * sr))
        if e - s < int(sr * 0.2):
            continue
        clip = signal[s:e]
        pitch = estimate_pitch_hz(clip[: min(clip.size, sr * 3)], sr)
        energy = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(clip)).astype(np.float32))))
        duration = max(0.2, seg.end - seg.start)
        words = max(1, len(re.findall(r"[A-Za-z]+", seg.text_src)))
        wps = float(words / duration)
        feats.append(np.array([pitch / 300.0, energy, zcr], dtype=np.float32))
        idxs.append(i)
        pitches.append(pitch)
        energies.append(energy)
        rates.append(wps)
        durations.append(duration)
    if len(feats) < 2:
        return {}, {}

    X = np.stack(feats, axis=0).astype(np.float32)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-5] = 1.0
    X_norm = (X - mu) / sigma
    duration_arr = np.array(durations, dtype=np.float32)
    max_speakers = max(1, int(speaker_count))
    n_clusters, labels, centroids = _choose_cluster_count(X_norm, duration_arr, max_speakers=max_speakers)

    if n_clusters > 1:
        labels = _viterbi_smooth_labels(X_norm, labels, centroids)
        labels = _merge_short_label_runs(labels, segments, idxs)
        ratios = _cluster_duration_ratios(labels, duration_arr, n_clusters)
        if float(np.max(ratios)) >= 0.90:
            n_clusters = 1
            labels = np.zeros_like(labels)

    seg_to_spk: Dict[int, str] = {}
    cluster_pitch: Dict[int, List[float]] = {k: [] for k in range(n_clusters)}
    cluster_energy: Dict[int, List[float]] = {k: [] for k in range(n_clusters)}
    cluster_rate: Dict[int, List[float]] = {k: [] for k in range(n_clusters)}
    for arr_idx, seg_idx in enumerate(idxs):
        label = int(labels[arr_idx]) if n_clusters > 1 else 0
        seg_to_spk[seg_idx] = f"S{label + 1}"
        if pitches[arr_idx] > 0:
            cluster_pitch[label].append(pitches[arr_idx])
        cluster_energy[label].append(energies[arr_idx])
        cluster_rate[label].append(rates[arr_idx])

    profiles: Dict[str, SpeakerProfile] = {}
    for c in range(n_clusters):
        p = cluster_pitch.get(c, [])
        median_pitch = float(np.median(p)) if p else 0.0
        if median_pitch > 0 and median_pitch <= 155:
            gender = "male"
        elif median_pitch >= 190:
            gender = "female"
        else:
            gender = "unknown"
        avg_energy = float(np.mean(cluster_energy.get(c, [0.0])))
        avg_rate = float(np.mean(cluster_rate.get(c, [0.0])))
        if avg_rate >= 2.8 or avg_energy >= 0.08:
            style = "energetic"
        elif avg_rate <= 1.8 and avg_energy <= 0.035:
            style = "calm"
        else:
            style = "serious"
        profiles[f"S{c + 1}"] = SpeakerProfile(gender=gender, style=style)
    return seg_to_spk, profiles


def list_supported_voices(
    api_key: str,
    api_url: str,
    model: str,
    voices: Sequence[str],
) -> List[str]:
    configure_tls_trust()
    _configure_dashscope_runtime(api_key=api_key, api_url=api_url)
    supported: List[str] = []
    for voice in voices:
        rsp = MultiModalConversation.call(
            model=model,
            text="测试",
            voice=voice,
            language_type="Chinese",
            stream=False,
        )
        url = parse_tts_url(rsp)
        if url:
            supported.append(voice)
    return supported


def chunked(items: Sequence[Segment], size: int) -> List[List[Segment]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _extract_json_array_candidate(text: str) -> str:
    s = strip_json_fence(text or "").strip()
    if not s:
        return s
    start = s.find("[")
    end = s.rfind("]")
    if 0 <= start < end:
        return s[start : end + 1]
    return s


def _parse_translation_json(content: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    candidates = [_extract_json_array_candidate(content), strip_json_fence(content or "").strip()]
    seen: set[str] = set()
    for cand in candidates:
        c = cand.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            loaded = json.loads(c)
        except Exception:
            loaded = None
        rows: object = loaded
        if isinstance(rows, dict):
            rows = rows.get("items", rows.get("data", []))
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                raw_id = item.get("id")
                text = item.get("zh", item.get("text", item.get("translation", "")))
                try:
                    idx = int(raw_id)
                except Exception:
                    continue
                txt = str(text or "").strip()
                if txt:
                    mapping[idx] = txt
        if mapping:
            return mapping

    # Regex salvage for malformed JSON output.
    s = _extract_json_array_candidate(content or "")
    for m in re.finditer(
        r'"id"\s*:\s*(\d+)\s*,\s*"(?:zh|text|translation)"\s*:\s*"((?:[^"\\\\]|\\\\.)*)"',
        s,
        flags=re.S,
    ):
        idx = int(m.group(1))
        raw_txt = m.group(2)
        try:
            txt = json.loads(f"\"{raw_txt}\"")
        except Exception:
            txt = raw_txt.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
        txt = str(txt).strip()
        if txt:
            mapping[idx] = txt
    return mapping


def _translate_single_line(
    client: OpenAI,
    model: str,
    text: str,
    target_language: str,
) -> str:
    prompt = (
        f"Translate the line into natural spoken {target_language}. "
        "Output translation text only, no JSON, no explanation."
    )
    rsp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    out = (rsp.choices[0].message.content or "").strip()
    return out or text


def translate_segments_qwen(
    segments: List[Segment],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    target_language: str = "Simplified Chinese",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)
    batches = chunked(segments, batch_size)
    for i, batch in enumerate(batches, start=1):
        payload = [{"id": s.idx, "speaker": s.speaker, "text": s.text_src} for s in batch]
        prompt = (
            f"Translate interview lines to natural spoken {target_language}.\n"
            "Keep meaning and speaking style.\n"
            "Return STRICT valid JSON array only, with this schema:\n"
            "[{\"id\": number, \"zh\": string}]"
        )
        mapping: Dict[int, str] = {}
        last_error: Optional[Exception] = None
        for _attempt in range(3):
            try:
                rsp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                content = rsp.choices[0].message.content or ""
                mapping = _parse_translation_json(content)
                if mapping:
                    break
                last_error = RuntimeError("empty mapping from translation json")
            except Exception as e:
                last_error = e
                continue

        # Degrade gracefully: backfill missing lines with single-line translation
        # so one malformed batch never fails the whole task.
        if len(mapping) < len(batch):
            for seg in batch:
                if seg.idx in mapping and mapping[seg.idx].strip():
                    continue
                try:
                    mapping[seg.idx] = _translate_single_line(
                        client=client,
                        model=model,
                        text=seg.text_src,
                        target_language=target_language,
                    ).strip()
                except Exception:
                    mapping[seg.idx] = seg.text_src

        if not mapping and last_error is not None:
            raise last_error

        for seg in batch:
            seg.text_zh = mapping.get(seg.idx, seg.text_src)
        if progress_callback:
            progress_callback(i, len(batches))


def assign_voices(
    segments: List[Segment],
    voices: Sequence[str],
    speaker_profiles: Optional[Dict[str, SpeakerProfile]] = None,
) -> Dict[str, str]:
    speakers = sorted({s.speaker for s in segments}) or ["S1"]
    mapping: Dict[str, str] = {}
    female_pref = ["Serena", "Cherry", "Chelsie", "Bella"]
    male_pref = ["Ethan", "Dylan", "Sunny"]
    female_energetic = ["Chelsie", "Bella", "Cherry", "Serena"]
    female_calm = ["Serena", "Cherry", "Bella", "Chelsie"]
    male_energetic = ["Sunny", "Dylan", "Ethan"]
    male_calm = ["Ethan", "Dylan", "Sunny"]
    neutral_pref = list(voices)
    all_unknown = all((speaker_profiles or {}).get(spk, SpeakerProfile()).gender == "unknown" for spk in speakers)
    for i, spk in enumerate(speakers):
        profile = (speaker_profiles or {}).get(spk, SpeakerProfile())
        if profile.gender == "female":
            if profile.style == "energetic":
                pool = [v for v in voices if v in female_energetic]
            elif profile.style == "calm":
                pool = [v for v in voices if v in female_calm]
            else:
                pool = [v for v in voices if v in female_pref]
            if not pool:
                pool = [v for v in voices if VOICE_GENDER_HINT.get(v) == "female"] or neutral_pref
        elif profile.gender == "male":
            if profile.style == "energetic":
                pool = [v for v in voices if v in male_energetic]
            elif profile.style == "calm":
                pool = [v for v in voices if v in male_calm]
            else:
                pool = [v for v in voices if v in male_pref]
            if not pool:
                pool = [v for v in voices if VOICE_GENDER_HINT.get(v) == "male"] or neutral_pref
        else:
            pool = [v for v in voices if v in male_pref] if (all_unknown and len(speakers) >= 2) else neutral_pref
            if not pool:
                pool = neutral_pref
        mapping[spk] = pool[i % len(pool)]
    for seg in segments:
        seg.voice = mapping[seg.speaker]
    return mapping


def parse_tts_url(response: object) -> str:
    if isinstance(response, dict):
        output = response.get("output")
        if not isinstance(output, dict):
            return ""
        audio = output.get("audio")
        if not isinstance(audio, dict):
            return ""
        return str(audio.get("url", ""))
    out = getattr(response, "output", None)
    if out is None:
        return ""
    audio = out.get("audio") if isinstance(out, dict) else getattr(out, "audio", None)
    if isinstance(audio, dict):
        return str(audio.get("url", ""))
    return str(getattr(audio, "url", ""))


def utf8_len(text: str) -> int:
    return len((text or "").encode("utf-8"))


def split_text_hard_limit(text: str, max_chars: int, max_bytes: int) -> List[str]:
    out: List[str] = []
    cur_chars: List[str] = []
    cur_chars_len = 0
    cur_bytes_len = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))
        if cur_chars and (cur_chars_len + 1 > max_chars or cur_bytes_len + ch_bytes > max_bytes):
            out.append("".join(cur_chars).strip())
            cur_chars = []
            cur_chars_len = 0
            cur_bytes_len = 0
        cur_chars.append(ch)
        cur_chars_len += 1
        cur_bytes_len += ch_bytes
    if cur_chars:
        out.append("".join(cur_chars).strip())
    return [x for x in out if x]


def split_tts_text(
    text: str,
    max_chars: int = MAX_TTS_INPUT_CHARS,
    max_bytes: int = MAX_TTS_INPUT_BYTES,
) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars and utf8_len(text) <= max_bytes:
        return [text]

    tokens = re.split(r"([。！？!?；;，,\n])", text)
    units: List[str] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i]
        sep = tokens[i + 1] if i + 1 < len(tokens) else ""
        unit = f"{chunk}{sep}".strip()
        if unit:
            units.append(unit)
        i += 2
    if not units:
        units = [text]

    parts: List[str] = []
    cur = ""
    for unit in units:
        if len(unit) > max_chars or utf8_len(unit) > max_bytes:
            if cur:
                parts.append(cur)
                cur = ""
            for sub in split_text_hard_limit(unit, max_chars=max_chars, max_bytes=max_bytes):
                if sub:
                    parts.append(sub)
            continue
        if not cur:
            cur = unit
            continue
        candidate = f"{cur} {unit}" if not cur.endswith((" ", "\n")) else f"{cur}{unit}"
        if len(candidate) <= max_chars and utf8_len(candidate) <= max_bytes:
            cur = candidate
        else:
            parts.append(cur)
            cur = unit
    if cur:
        parts.append(cur)
    return parts


def is_tts_input_too_long_error(err_msg: str) -> bool:
    msg = (err_msg or "").lower()
    return (
        ("invalidparameter" in msg or "invalid parameter" in msg)
        and ("input length" in msg or "length should be" in msg or "out of range" in msg)
        and ("[0, 600]" in msg or "range of input length" in msg or "600" in msg)
    )


def concat_audio_clips(paths: Sequence[str], out_path: str) -> None:
    clips = [p for p in paths if p]
    if not clips:
        raise RuntimeError("No audio clips to concatenate")
    if len(clips) == 1:
        shutil.copyfile(clips[0], out_path)
        return

    args: List[str] = ["ffmpeg", "-y"]
    for p in clips:
        args += ["-i", p]
    nodes = "".join(f"[{i}:a]" for i in range(len(clips)))
    args += [
        "-filter_complex",
        f"{nodes}concat=n={len(clips)}:v=0:a=1[aout]",
        "-map",
        "[aout]",
        out_path,
    ]
    run(args)


def qwen_tts_to_file(
    api_key: str,
    model: str,
    text: str,
    voice: str,
    out_path: str,
    api_url: str,
    language_type: str = "Chinese",
) -> None:
    configure_tls_trust()
    _configure_dashscope_runtime(api_key=api_key, api_url=api_url)
    last_err = "unknown"
    for _ in range(3):
        rsp = MultiModalConversation.call(
            model=model,
            text=text,
            voice=voice,
            language_type=language_type,
            stream=False,
        )
        url = parse_tts_url(rsp)
        if url:
            with requests.get(url, timeout=120, verify=current_tls_verify_arg()) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
            return
        code = getattr(rsp, "code", "") if not isinstance(rsp, dict) else str(rsp.get("code", ""))
        msg = getattr(rsp, "message", "") if not isinstance(rsp, dict) else str(rsp.get("message", ""))
        status = getattr(rsp, "status_code", "") if not isinstance(rsp, dict) else str(rsp.get("status_code", ""))
        last_err = f"status={status} code={code} message={msg}"
        if is_tts_input_too_long_error(last_err):
            break
    raise RuntimeError(f"Qwen TTS failed: {last_err}")


def qwen_tts_to_file_with_retry_split(
    api_key: str,
    model: str,
    text: str,
    voice: str,
    out_path: str,
    api_url: str,
    language_type: str,
    temp_dir: Path,
    key_prefix: str,
    depth: int = 0,
) -> None:
    try:
        qwen_tts_to_file(
            api_key=api_key,
            model=model,
            text=text,
            voice=voice,
            out_path=out_path,
            api_url=api_url,
            language_type=language_type,
        )
        return
    except RuntimeError as e:
        if depth >= 8 or (not is_tts_input_too_long_error(str(e))) or len(text.strip()) <= 1:
            raise

    # Input length still rejected by remote side; split harder and retry recursively.
    # Keep each retry chunk much smaller to cover multi-byte languages (ja/zh/ko).
    level_chars = [180, 140, 110, 90, 70, 55, 45, 35]
    level_bytes = [240, 200, 170, 150, 130, 110, 95, 80]
    idx = min(depth, len(level_chars) - 1)
    next_max_chars = level_chars[idx]
    next_max_bytes = level_bytes[idx]
    sub_texts = split_tts_text(text, max_chars=next_max_chars, max_bytes=next_max_bytes)
    if len(sub_texts) <= 1:
        mid = max(1, len(text) // 2)
        sub_texts = [text[:mid].strip(), text[mid:].strip()]
        sub_texts = [x for x in sub_texts if x]
    if len(sub_texts) <= 1:
        raise RuntimeError("Qwen TTS failed: text too long but cannot split further")

    part_paths: List[str] = []
    for idx, part in enumerate(sub_texts, start=1):
        part_raw = temp_dir / f"{key_prefix}_autosplit_{depth}_{idx:03d}.wav"
        qwen_tts_to_file_with_retry_split(
            api_key=api_key,
            model=model,
            text=part,
            voice=voice,
            out_path=str(part_raw),
            api_url=api_url,
            language_type=language_type,
            temp_dir=temp_dir,
            key_prefix=f"{key_prefix}_{idx:03d}",
            depth=depth + 1,
        )
        part_norm = temp_dir / f"{key_prefix}_autosplit_{depth}_{idx:03d}_norm.wav"
        run(["ffmpeg", "-y", "-i", str(part_raw), "-ac", "1", "-ar", "44100", str(part_norm)])
        part_paths.append(str(part_norm))
    concat_audio_clips(part_paths, out_path)


def atempo_chain(ratio: float) -> List[str]:
    ratio = max(0.1, ratio)
    filters: List[str] = []
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    filters.append(f"atempo={ratio:.6f}")
    return filters


def retime_clip(in_path: str, out_path: str, target_duration: float) -> None:
    target = max(0.03, float(target_duration))
    current = ffprobe_duration(in_path)
    if current <= 0.03:
        # If generated clip is empty/corrupted, synthesize silence to preserve timeline.
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-t",
                f"{target:.6f}",
                "-i",
                "anullsrc=channel_layout=mono:sample_rate=44100",
                out_path,
            ]
        )
        return

    # Match target duration exactly:
    # 1) atempo adjusts speaking speed
    # 2) apad+atrim guarantees exact output length to avoid overlaps/gaps.
    speed = max(0.05, current / target)
    filters = atempo_chain(speed) + [f"apad=pad_dur={target:.6f}", f"atrim=0:{target:.6f}"]
    run(["ffmpeg", "-y", "-i", in_path, "-filter:a", ",".join(filters), out_path])


def synthesize_all(
    segments: List[Segment],
    temp_dir: Path,
    api_key: str,
    tts_model: str,
    api_url: str,
    language_type: str = "Chinese",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    total = len(segments)
    for i, seg in enumerate(segments, start=1):
        raw = temp_dir / f"seg_{seg.idx:05d}.wav"
        text = (seg.text_zh or seg.text_src or "").strip()
        chunks = split_tts_text(text, max_chars=MAX_TTS_INPUT_CHARS, max_bytes=MAX_TTS_INPUT_BYTES)
        if not chunks:
            chunks = ["..."]

        if len(chunks) == 1:
            qwen_tts_to_file_with_retry_split(
                api_key=api_key,
                model=tts_model,
                text=chunks[0],
                voice=seg.voice,
                out_path=str(raw),
                api_url=api_url,
                language_type=language_type,
                temp_dir=temp_dir,
                key_prefix=f"seg_{seg.idx:05d}",
            )
        else:
            part_paths: List[str] = []
            for j, part in enumerate(chunks, start=1):
                part_raw = temp_dir / f"seg_{seg.idx:05d}_part_{j:03d}.wav"
                qwen_tts_to_file_with_retry_split(
                    api_key=api_key,
                    model=tts_model,
                    text=part,
                    voice=seg.voice,
                    out_path=str(part_raw),
                    api_url=api_url,
                    language_type=language_type,
                    temp_dir=temp_dir,
                    key_prefix=f"seg_{seg.idx:05d}_part_{j:03d}",
                )
                part_norm = temp_dir / f"seg_{seg.idx:05d}_part_{j:03d}_norm.wav"
                run(["ffmpeg", "-y", "-i", str(part_raw), "-ac", "1", "-ar", "44100", str(part_norm)])
                part_paths.append(str(part_norm))
            concat_audio_clips(part_paths, str(raw))

        # Fit each segment into its slot to avoid cross-segment overlap.
        base_target = max(0.08, seg.end - seg.start)
        seg_pos = i - 1
        if seg_pos < total - 1:
            next_start = max(seg.start, segments[seg_pos + 1].start)
            slot_target = max(0.03, next_start - seg.start - 0.01)
            target = min(base_target, slot_target)
        else:
            target = base_target

        retimed = temp_dir / f"seg_{seg.idx:05d}_retime.wav"
        retime_clip(str(raw), str(retimed), target)
        seg.audio_path = str(retimed)
        if progress_callback:
            progress_callback(i, total)


def mix_segments(segments: List[Segment], duration: float, out_path: str) -> None:
    args: List[str] = ["ffmpeg", "-y"]
    args += ["-f", "lavfi", "-t", f"{duration + 0.5:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    for seg in segments:
        args += ["-i", seg.audio_path]

    filters: List[str] = []
    mix_nodes: List[str] = ["[0:a]"]
    for i, seg in enumerate(segments, start=1):
        delay_ms = max(0, int(seg.start * 1000))
        node = f"s{i}"
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[{node}]")
        mix_nodes.append(f"[{node}]")
    filters.append(f"{''.join(mix_nodes)}amix=inputs={len(mix_nodes)}:normalize=0[aout]")
    args += ["-filter_complex", ";".join(filters), "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", out_path]
    run(args)


def mux_video_with_audio(video_path: str, dubbed_audio_path: str, out_video_path: str) -> str:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            dubbed_audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_video_path,
        ]
    )
    return out_video_path


def save_artifacts(out_dir: Path, source_audio: str, dubbed_audio: str, speaker_voice_map: Dict[str, str], segments: List[Segment]) -> None:
    manifest = {
        "source_audio": source_audio,
        "dubbed_audio": dubbed_audio,
        "speaker_voice_map": speaker_voice_map,
        "segments": len(segments),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        {
            "idx": s.idx,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "speaker": s.speaker,
            "voice": s.voice,
            "src": s.text_src,
            "zh": s.text_zh,
        }
        for s in segments
    ]
    (out_dir / "segments.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube/Audio -> Multi-speaker Chinese dubbed audio (Qwen only)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube-url", help="YouTube URL")
    src.add_argument("--input-audio", help="Local audio/video file path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--qwen-api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    p.add_argument("--dashscope-api-url", default=os.getenv("DASHSCOPE_HTTP_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"))
    p.add_argument("--qwen-openai-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--asr-model", default="paraformer-realtime-v2")
    p.add_argument("--translate-model", default="qwen3-max")
    p.add_argument("--tts-model", default="qwen3-tts-instruct-flash")
    p.add_argument("--voices", default=",".join(DEFAULT_TTS_VOICES), help="Comma separated qwen3-tts voice list")
    p.add_argument("--output-language", default="zh-CN", choices=sorted(OUTPUT_LANGUAGE_MAP.keys()))
    p.add_argument("--translation-batch-size", type=int, default=30)
    p.add_argument("--speaker-count", type=int, default=2, help="Max speaker count for fallback diarization (1 = force single speaker)")
    p.add_argument("--max-seconds", type=float, default=None, help="Process only first N seconds (for long videos)")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument(
        "--tls-mode",
        default=TLS_MODE_AUTO,
        choices=[TLS_MODE_AUTO, TLS_MODE_SYSTEM, TLS_MODE_CERTIFI, TLS_MODE_CUSTOM_CA, TLS_MODE_DEFAULT],
        help="TLS certificate mode",
    )
    p.add_argument("--ca-bundle-file", default="", help="PEM CA bundle path when --tls-mode=custom_ca")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.qwen_api_key.strip():
        print("[ERROR] Missing Qwen API key. Set DASHSCOPE_API_KEY or --qwen-api-key", file=sys.stderr)
        return 1
    tls_mode = configure_tls_trust(mode=args.tls_mode, ca_bundle_file=args.ca_bundle_file)
    print(f"[INFO] TLS mode: {tls_mode}")
    require_binary("ffmpeg")
    require_binary("ffprobe")
    if args.youtube_url:
        require_binary("yt-dlp")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.keep_temp:
        temp_dir = out_dir / "tmp"
        temp_dir.mkdir(exist_ok=True)
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="qwen_dub_", dir=str(out_dir))
        temp_dir = Path(tmp_obj.name)

    try:
        source = download_youtube_audio(args.youtube_url, temp_dir) if args.youtube_url else str(Path(args.input_audio).resolve())
        normalized = str(temp_dir / "source_16k.wav")
        normalize_audio(source, normalized)
        asr_audio = normalized
        if args.max_seconds is not None and args.max_seconds > 0:
            trimmed = str(temp_dir / "source_16k_trim.wav")
            trim_audio(normalized, trimmed, args.max_seconds)
            asr_audio = trimmed
        duration = ffprobe_duration(asr_audio)

        asr_result = asr_with_qwen(args.qwen_api_key, args.asr_model, asr_audio, args.dashscope_api_url)
        segments = normalize_segments(asr_result.get("segments", []), duration)
        speaker_profiles = normalize_speaker_profiles(asr_result)
        collapsed = maybe_collapse_to_single_speaker(segments)
        if collapsed:
            speaker_profiles = {collapsed: speaker_profiles.get(collapsed, SpeakerProfile())}
        translate_target, tts_language_type = OUTPUT_LANGUAGE_MAP.get(args.output_language, ("Simplified Chinese", "Chinese"))
        # Fallback: if ASR does not provide useful speaker ids, cluster from raw audio.
        unique_speakers = {s.speaker for s in segments}
        need_cluster = len(unique_speakers) <= 1
        if need_cluster:
            seg_spk_map, inferred_profiles = infer_speakers_from_audio(
                segments=segments,
                wav_path=asr_audio,
                speaker_count=args.speaker_count,
            )
            if seg_spk_map:
                for idx, spk in seg_spk_map.items():
                    segments[idx].speaker = spk
                speaker_profiles.update(inferred_profiles)
                collapsed_after_infer = maybe_collapse_to_single_speaker(segments)
                if collapsed_after_infer:
                    speaker_profiles = {collapsed_after_infer: speaker_profiles.get(collapsed_after_infer, SpeakerProfile())}
        translate_segments_qwen(
            segments=segments,
            api_key=args.qwen_api_key,
            base_url=args.qwen_openai_base_url,
            model=args.translate_model,
            batch_size=args.translation_batch_size,
            target_language=translate_target,
        )

        voices = [v.strip() for v in args.voices.split(",") if v.strip()]
        if not voices:
            raise RuntimeError("No valid voices configured")
        voices = list_supported_voices(args.qwen_api_key, args.dashscope_api_url, args.tts_model, voices)
        if not voices:
            raise RuntimeError("No supported voices for current tts model")
        speaker_voice_map = assign_voices(segments, voices, speaker_profiles)
        synthesize_all(
            segments,
            temp_dir,
            args.qwen_api_key,
            args.tts_model,
            args.dashscope_api_url,
            language_type=tts_language_type,
        )

        dubbed = str(out_dir / "dubbed_zh.m4a")
        mix_segments(segments, duration, dubbed)
        save_artifacts(out_dir, source, dubbed, speaker_voice_map, segments)
        print(json.dumps({"ok": True, "output_audio": dubbed, "segments": len(segments)}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
