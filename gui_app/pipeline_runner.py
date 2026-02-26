from __future__ import annotations

import json
import re
import subprocess
import sys
import os
import shutil
import tempfile
import traceback
import threading
import time
import asyncio
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
try:
    import aiohttp
except Exception:  # pragma: no cover - optional fallback
    aiohttp = None
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import dub_pipeline as core

ProgressCallback = Callable[[int, str, str], None]

_SUPPORTED_COOKIE_BROWSERS = {"chrome", "safari", "firefox", "edge", "chromium", "brave", "opera", "vivaldi"}
_YT_JS_RUNTIMES = ("deno", "node", "quickjs", "bun")
_YT_REMOTE_COMPONENTS = ("ejs:github",)


def _inject_bundled_bin_to_path() -> None:
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / "bin",
                exe_dir.parent / "Resources" / "bin",
            ]
        )
    else:
        candidates.extend(
            [
                PROJECT_ROOT / "gui_app" / "bin",
                PROJECT_ROOT / "third_party" / "bin",
            ]
        )
    existing = [str(p) for p in candidates if p.exists() and p.is_dir()]
    if not existing:
        return
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(existing + [current]) if current else os.pathsep.join(existing)


def _fmt_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _start_asr_heartbeat(
    cb: ProgressCallback,
    audio_duration: float,
    start_percent: int = 48,
    end_percent: int = 58,
) -> tuple[threading.Event, threading.Thread, float]:
    stop_event = threading.Event()
    started_at = time.monotonic()
    span = max(1, end_percent - start_percent)
    # Heuristic ETA only for UX heartbeat; real completion still depends on remote ASR.
    expected = max(25.0, min(900.0, max(1.0, audio_duration) * 0.45))

    def _run() -> None:
        while not stop_event.wait(2.0):
            elapsed = time.monotonic() - started_at
            ratio = min(0.98, elapsed / expected)
            pct = start_percent + int(ratio * span)
            cb(
                pct,
                "语音识别",
                f"云端识别中... 已耗时 {_fmt_elapsed(elapsed)}（长视频会更久）",
            )

    th = threading.Thread(target=_run, name="asr-heartbeat", daemon=True)
    th.start()
    return stop_event, th, started_at


@dataclass
class PreviewInfo:
    url: str
    title: str
    description: str
    thumbnail_url: str
    duration_seconds: float
    uploader: str
    detected_language: str
    raw: Dict


@dataclass
class ConversionConfig:
    qwen_api_key: str
    source_url: str = ""
    source_file: str = ""
    output_root: str = ""
    output_language: str = "zh-CN"
    asr_model: str = "paraformer-realtime-v2"
    translate_model: str = "qwen3-max"
    tts_model: str = "qwen3-tts-flash"
    voices: str = "Ethan,Dylan,Sunny,Cherry,Serena,Chelsie,Bella"
    cookies_from_browser: str = "none"
    cookies_file: str = ""
    speaker_count: int = 2
    max_seconds: float = 0.0
    keep_temp: bool = False
    tls_mode: str = "auto"
    ca_bundle_file: str = ""
    dashscope_api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    qwen_openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class ConversionResult:
    output_dir: str
    source_path: str
    dubbed_audio_path: str
    dubbed_video_path: str
    manifest_path: str
    segments_path: str
    speaker_voice_map: Dict[str, str]
    segment_count: int


def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    return text[:60] or "task"


def detect_language_from_metadata(info: Dict) -> str:
    lang = (info.get("language") or info.get("release_language") or "").lower()
    if lang.startswith("zh"):
        return "zh-CN"
    if lang.startswith("ja"):
        return "ja-JP"
    if lang.startswith("es"):
        return "es-ES"
    if lang.startswith("fr"):
        return "fr-FR"
    if lang.startswith("en"):
        return "en-US"
    caps = info.get("automatic_captions") or {}
    if isinstance(caps, dict) and caps:
        first = str(next(iter(caps.keys()), "")).lower()
        if first.startswith("zh"):
            return "zh-CN"
        if first.startswith("ja"):
            return "ja-JP"
        if first.startswith("es"):
            return "es-ES"
        if first.startswith("fr"):
            return "fr-FR"
        if first.startswith("en"):
            return "en-US"
    return "en-US"


def _normalize_cookie_browser(browser: str) -> Optional[str]:
    b = str(browser or "").strip().lower()
    if not b or b == "none":
        return None
    if b in _SUPPORTED_COOKIE_BROWSERS:
        return b
    return None


def _normalize_cookie_file(cookie_file: str) -> Optional[str]:
    path = str(cookie_file or "").strip()
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    return str(p.resolve())


def _available_js_runtimes() -> List[str]:
    return [runtime for runtime in _YT_JS_RUNTIMES if shutil.which(runtime)]


def _yt_dlp_base_opts() -> Dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Enable multiple runtimes to improve compatibility on user machines.
        "js_runtimes": {runtime: {} for runtime in _YT_JS_RUNTIMES},
        # Allow yt-dlp to fetch official EJS components when local package is missing.
        "remote_components": list(_YT_REMOTE_COMPONENTS),
    }


def _build_cookie_candidates(cookie_browser: str, cookie_file: str) -> List[tuple[Dict, str]]:
    candidates: List[tuple[Dict, str]] = []
    cookie_path = _normalize_cookie_file(cookie_file)
    browser = _normalize_cookie_browser(cookie_browser)
    if cookie_path:
        candidates.append(({"cookiefile": cookie_path}, f"cookies.txt({cookie_path})"))
    if browser:
        candidates.append(({"cookiesfrombrowser": (browser,)}, f"{browser} 登录态"))
    return candidates


def _extractor_client_profiles() -> List[tuple[str, Dict]]:
    return [
        (
            "兼容优先客户端",
            {
                "extractor_args": {
                    "youtube": {
                        # Authenticated mode defaults to tv_downgraded+web+web_safari.
                        # Excluding TV clients avoids DRM-only responses on some accounts.
                        "player_client": ["default", "mweb", "-tv", "-tv_downgraded"],
                    }
                }
            },
        ),
        ("默认客户端", {}),
        (
            "WEB 客户端",
            {"extractor_args": {"youtube": {"player_client": ["web", "web_safari"]}}},
        ),
        (
            "MWEB 客户端",
            {"extractor_args": {"youtube": {"player_client": ["mweb", "web"]}}},
        ),
    ]


def _opts_has_cookie(opts: Dict) -> bool:
    return "cookiefile" in opts or "cookiesfrombrowser" in opts


def _cookie_source_from_opts(opts: Dict) -> str:
    if "cookiefile" in opts:
        return f"cookies.txt({opts.get('cookiefile')})"
    if "cookiesfrombrowser" in opts:
        raw = opts.get("cookiesfrombrowser")
        if isinstance(raw, tuple) and raw:
            return f"{raw[0]} 登录态"
        return "浏览器登录态"
    return ""


def _is_cookie_permission_error(err_msg: str) -> bool:
    msg = str(err_msg).lower()
    return (
        ("operation not permitted" in msg or "permission denied" in msg)
        and (
            "cookies.binarycookies" in msg
            or "com.apple.safari/data/library/cookies" in msg
            or "library/cookies/cookies.binarycookies" in msg
            or "keychain" in msg
        )
    )


def _build_cookie_permission_error(cookie_source: str, raw_err: Exception) -> RuntimeError:
    source = cookie_source or "浏览器登录态"
    return RuntimeError(
        f"无法读取 {source} 的 cookies（系统权限限制）。\n"
        "请在 macOS「系统设置 -> 隐私与安全性 -> 完全磁盘访问」中给 VideoDubStudio 授权后重启应用，"
        "或改用导入 cookies.txt。\n\n"
        f"原始错误: {raw_err}"
    )


def _youtube_needs_login(err_msg: str) -> bool:
    msg = err_msg.lower()
    return (
        "sign in to confirm you're not a bot" in msg
        or "use --cookies-from-browser" in msg
        or "this content isn't available" in msg
    )


def _youtube_unavailable(err_msg: str) -> bool:
    msg = err_msg.lower()
    return (
        "video unavailable" in msg
        or "private video" in msg
        or "this video is unavailable" in msg
        or "not available in your country" in msg
        or "members-only" in msg
    )


def _is_format_related_error(msg: str) -> bool:
    text = msg.lower()
    return (
        ("requested format" in text and "not available" in text)
        or "requested merging of multiple formats" in text
        or "no video formats found" in text
        or "only images are available for download" in text
        or "this video is drm protected" in text
        or "no suitable formats" in text
    )


def _build_youtube_login_error(raw_err: Exception, cookie_browser: str, cookie_file: str = "") -> RuntimeError:
    cookie_path = _normalize_cookie_file(cookie_file)
    chosen = _normalize_cookie_browser(cookie_browser)
    if cookie_path:
        tip = (
            "YouTube 访问被风控拦截，已尝试使用 cookies.txt 仍失败。"
            "请确认 cookies.txt 来自已登录且刚完成验证的 YouTube 会话，并重新导出后重试。"
        )
    elif chosen:
        tip = (
            f"YouTube 访问被风控拦截，已尝试使用 {chosen} 登录态仍失败。"
            "请确认该浏览器已登录 YouTube（非无痕模式），然后重试。"
        )
    else:
        tip = (
            "YouTube 访问被风控拦截，需要登录态（cookies）。"
            "请在“2) 配置输出”中选择浏览器登录态，或导入 cookies.txt 后重试。"
        )
    return RuntimeError(f"{tip}\n\n原始错误: {raw_err}")


def fetch_youtube_preview(url: str, cookie_browser: str = "none", cookie_file: str = "") -> PreviewInfo:
    _inject_bundled_bin_to_path()
    base_opts = _yt_dlp_base_opts()
    base_opts["skip_download"] = True
    cookie_candidates = _build_cookie_candidates(cookie_browser, cookie_file)
    auth_opts: List[Dict] = [dict(base_opts)]
    for cookie_opts, _ in cookie_candidates:
        auth_opts.append({**base_opts, **cookie_opts})

    attempt_opts: List[Dict] = []
    for _, client_patch in _extractor_client_profiles():
        for opts in auth_opts:
            merged = {**opts, **client_patch}
            attempt_opts.append(merged)
            attempt_opts.append(
                {
                    **merged,
                    # Preview should not be blocked by merged-format selection quirks.
                    "format": "b",
                }
            )

    last_error: Optional[Exception] = None
    permission_error: Optional[RuntimeError] = None
    info: Optional[Dict] = None
    for opts in attempt_opts:
        try:
            with YoutubeDL(opts) as ydl:
                # Preview only needs metadata; disable post-processing/format resolution
                # so metadata fetch is not blocked by transient format selection errors.
                maybe_info = ydl.extract_info(url, download=False, process=False)
            info = maybe_info if isinstance(maybe_info, dict) else {}
            last_error = None
            break
        except DownloadError as e:
            msg = str(e).lower()
            last_error = e
            if _youtube_unavailable(msg):
                raise RuntimeError(
                    "该 YouTube 视频当前不可访问（可能是私有视频、地区限制、会员专属或被下架）。"
                ) from e
            if _youtube_needs_login(msg):
                # Try all configured cookie sources before failing.
                if cookie_candidates:
                    continue
                raise _build_youtube_login_error(e, cookie_browser, cookie_file) from e
            if _is_format_related_error(msg):
                continue
            raise
        except Exception as e:
            last_error = e
            if _opts_has_cookie(opts) and _is_cookie_permission_error(str(e)):
                permission_error = _build_cookie_permission_error(_cookie_source_from_opts(opts), e)
                continue
            raise

    if info is None:
        if permission_error is not None:
            raise permission_error
        if last_error is not None and _youtube_needs_login(str(last_error)):
            raise _build_youtube_login_error(last_error, cookie_browser, cookie_file) from last_error
        if last_error is not None and _is_format_related_error(str(last_error)):
            raise RuntimeError("预览获取失败：YouTube 返回了不可用格式，请更换链接或稍后重试。") from last_error
        if last_error is not None:
            raise last_error
        raise RuntimeError("预览获取失败：未知错误")

    title = str(info.get("title") or "").strip()
    description = str(info.get("description") or "").strip()
    return PreviewInfo(
        url=url,
        title=title,
        description=description,
        thumbnail_url=str(info.get("thumbnail") or ""),
        duration_seconds=float(info.get("duration") or 0.0),
        uploader=str(info.get("uploader") or ""),
        detected_language=detect_language_from_metadata(info),
        raw=info,
    )


def _download_video_with_progress(
    url: str,
    out_dir: Path,
    cb: ProgressCallback,
    cookie_browser: str = "none",
    cookie_file: str = "",
) -> str:
    _inject_bundled_bin_to_path()
    out_tmpl = str(out_dir / "source.%(ext)s")

    def detect_stage(data: Dict) -> str:
        info = data.get("info_dict") if isinstance(data.get("info_dict"), dict) else {}
        vcodec = str(info.get("vcodec") or "").lower()
        acodec = str(info.get("acodec") or "").lower()
        ext = str(info.get("ext") or "").lower()
        if vcodec and vcodec != "none" and (not acodec or acodec == "none"):
            return "下载视频流"
        if acodec and acodec != "none" and (not vcodec or vcodec == "none"):
            return "下载音频流"
        if vcodec and vcodec != "none" and acodec and acodec != "none":
            return "下载单文件"
        if ext in {"m4a", "mp3", "aac", "wav", "opus", "ogg"}:
            return "下载音频流"
        return "下载媒体"

    def hook(data: Dict) -> None:
        status = data.get("status")
        stage = detect_stage(data)
        if status == "downloading":
            total = float(data.get("total_bytes") or data.get("total_bytes_estimate") or 0.0)
            done = float(data.get("downloaded_bytes") or 0.0)
            ratio = (done / total) if total > 0 else 0.0
            pct = min(35, int(8 + ratio * 27))
            speed = data.get("speed")
            detail = f"{done/1024/1024:.1f}MB"
            if speed:
                detail += f" | {float(speed)/1024/1024:.2f}MB/s"
            cb(pct, stage, detail)
        elif status == "finished":
            cb(36, stage, "下载完成，准备处理...")

    base_opts = _yt_dlp_base_opts()
    base_opts.update(
        {
            # Prefer progressive streams to avoid early ffmpeg merge dependency in yt-dlp.
            "outtmpl": out_tmpl,
            "progress_hooks": [hook],
        }
    )
    cookie_candidates = _build_cookie_candidates(cookie_browser, cookie_file)
    available_runtimes = _available_js_runtimes()
    if cookie_candidates and not available_runtimes:
        cb(
            5,
            "下载策略",
            "未检测到可用 JS 运行时（deno/node/quickjs/bun），登录态下载成功率可能下降。",
        )
    elif cookie_candidates:
        cb(5, "下载策略", f"JS 运行时: {', '.join(available_runtimes)}")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        base_opts["ffmpeg_location"] = ffmpeg_bin

    attempts = [
        (
            "progressive",
            {
                "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none][acodec!=none]",
            },
        ),
        (
            "adaptive_merge",
            {
                "format": "bv*+ba/b",
                "merge_output_format": "mp4",
            },
        ),
        (
            "any_best",
            {
                # Let yt-dlp auto-select downloadable best formats.
                # This is safer than forcing `best` (pre-merged only), which
                # often fails on YouTube where many videos are separate AV streams.
            },
        ),
        (
            "single_file_best",
            {
                "format": "b",
            },
        ),
    ]
    attempt_label = {
        "progressive": "单文件直下（优先）",
        "adaptive_merge": "音视频分流下载后合流",
        "any_best": "自动选择最佳可下载格式",
        "single_file_best": "仅单文件格式兜底",
    }

    def run_attempts(cookie_opts: Optional[Dict] = None, cookie_label: str = "") -> tuple[bool, Optional[Exception], bool]:
        last_error: Optional[Exception] = None
        local_base = dict(base_opts)
        if cookie_opts:
            local_base.update(cookie_opts)
        profile_count = 0
        for profile_name, client_patch in _extractor_client_profiles():
            profile_count += 1
            if profile_count > 1:
                cb(5, "下载策略", f"切换解析客户端: {profile_name}")
            profile_base = {**local_base, **client_patch}
            for name, patch in attempts:
                if name == "adaptive_merge" and not ffmpeg_bin:
                    continue
                cb(5, "下载策略", f"当前策略: {attempt_label.get(name, name)}")
                opts = dict(profile_base)
                opts.update(patch)
                try:
                    with YoutubeDL(opts) as ydl:
                        ydl.extract_info(url, download=True)
                    return True, None, False
                except DownloadError as e:
                    msg = str(e).lower()
                    last_error = e
                    if _youtube_needs_login(msg):
                        return False, last_error, True
                    if _youtube_unavailable(msg):
                        raise RuntimeError(
                            "该 YouTube 视频当前不可下载（可能是私有视频、地区限制、会员专属或被下架）。请更换视频或更换登录账号后重试。"
                        ) from e
                    if not _is_format_related_error(msg):
                        raise
                    cb(6, "下载策略", f"{attempt_label.get(name, name)}不可用，自动切换备用策略...")
                except Exception as e:
                    if cookie_opts and _is_cookie_permission_error(str(e)):
                        return False, _build_cookie_permission_error(cookie_label, e), False
                    raise
        return False, last_error, False

    success, last_error, need_cookie_retry = run_attempts(cookie_opts=None, cookie_label="")
    tried_cookie_sources: List[str] = []
    permission_errors: List[RuntimeError] = []
    if (not success) and cookie_candidates and (need_cookie_retry or (last_error and _is_format_related_error(str(last_error)))):
        for cookie_opts, cookie_source in cookie_candidates:
            tried_cookie_sources.append(cookie_source)
            cb(5, "下载策略", f"触发风控，切换到 {cookie_source} 重试...")
            s2, e2, _ = run_attempts(cookie_opts=cookie_opts, cookie_label=cookie_source)
            if s2:
                success = True
                last_error = None
                break
            if e2 is not None:
                last_error = e2
                if isinstance(e2, RuntimeError) and str(e2).startswith("无法读取 "):
                    permission_errors.append(e2)

    if (not success) and last_error is not None:
        if permission_errors and (_youtube_needs_login(str(last_error)) or _is_format_related_error(str(last_error))):
            raise permission_errors[-1]
        if _youtube_needs_login(str(last_error)):
            raise _build_youtube_login_error(last_error, cookie_browser, cookie_file) from last_error
        if not ffmpeg_bin:
            raise RuntimeError(
                "下载该视频需要 ffmpeg 合流，但当前环境不可用。请使用最新内置 ffmpeg 的安装包后重试。"
            ) from last_error
        msg = str(last_error).lower()
        if _is_format_related_error(msg):
            tried_msg = f"已尝试登录态: {', '.join(tried_cookie_sources)}。 " if tried_cookie_sources else ""
            runtime_tip = (
                " 当前未检测到可用 JS 运行时（deno/node/quickjs/bun）。"
                " 请使用包含内置 deno 的新版本安装包，或在系统安装 deno/node 后重试。"
                if (tried_cookie_sources and not available_runtimes)
                else ""
            )
            raise RuntimeError(
                f"未找到可下载的视频格式。{tried_msg}"
                "常见原因：视频受版权/地区限制、该链接无可用流、当前登录态无访问权限，或 cookies 已过期。"
                f"请先确认在浏览器可直接播放该视频，并重新导出 cookies.txt 后再试。{runtime_tip}"
            ) from last_error
        raise last_error

    candidates = [
        p
        for p in sorted(out_dir.glob("source.*"))
        if p.is_file() and p.suffix not in {".part", ".ytdl"}
    ]
    if not candidates:
        raise RuntimeError("视频下载成功但未找到本地文件")
    return str(candidates[-1])


def _has_video_stream(path: str) -> bool:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            path,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return "video" in (p.stdout or "").lower()


def _short_error(err: Exception, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(err or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _is_tls_cert_error_message(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "certificate_verify_failed" in msg
        or "certificate verify failed" in msg
        or "unable to get local issuer certificate" in msg
        or "sslcertverificationerror" in msg
    )


def _is_auth_error_message(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "invalid api-key" in msg
        or "invalid_api_key" in msg
        or "incorrect api key" in msg
        or "no api key provided" in msg
        or "401 unauthorized" in msg
        or "authenticationerror" in msg
        or "apikey-error" in msg
    )


def _is_rate_limit_message(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "quota" in msg
        or "throttl" in msg
    )


def _is_permission_message(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "403" in msg
        or "forbidden" in msg
        or "access denied" in msg
        or "permission" in msg
        or "not authorized" in msg
    )


def _is_model_message(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "model not found" in msg
        or "unsupported model" in msg
        or "invalid model" in msg
        or "model is not available" in msg
        or "no permission to access this model" in msg
    )


def _wrap_dashscope_stage_error(stage: str, err: Exception) -> RuntimeError:
    raw = str(err or "")
    raw_lower = raw.lower()
    if "file_download_failed" in raw_lower:
        return RuntimeError(
            f"{stage}失败：云端识别任务无法读取上传音频（FILE_DOWNLOAD_FAILED）。"
            "已自动启用 OSS 解析头重试；若仍失败，请将 failure_report.txt 发给开发者。"
        )
    if "fallback=" in raw_lower:
        fb_part = raw.split("fallback=", 1)[-1].strip()
        fb_lower = fb_part.lower()
        if "返回为空" in fb_part or "empty" in fb_lower:
            return RuntimeError(
                f"{stage}失败：HTTP 兜底已触发，但识别结果解析为空。"
                "请将 failure_report.txt 发给开发者进行适配。"
            )
        if _is_model_message(fb_part):
            return RuntimeError(f"{stage}失败：HTTP 兜底模型不可用或当前账号无权限访问该模型。")
        if _is_auth_error_message(fb_part):
            return RuntimeError(f"{stage}失败：HTTP 兜底调用鉴权失败，请检查 API Key。")
        if _is_rate_limit_message(fb_part):
            return RuntimeError(f"{stage}失败：HTTP 兜底请求被限流或额度不足（429/Quota）。")
        if _is_permission_message(fb_part):
            return RuntimeError(f"{stage}失败：HTTP 兜底无调用权限（403）。")
    if _is_tls_cert_error_message(raw):
        return RuntimeError(
            f"{stage}失败：TLS 证书校验异常。请优先使用“自动（推荐）”或“内置 certifi 证书链”，"
            "企业网络下可导入自定义 CA 证书后重试。"
        )
    if _is_auth_error_message(raw):
        return RuntimeError(f"{stage}失败：阿里云 API Key 无效或已过期。")
    if _is_rate_limit_message(raw):
        return RuntimeError(
            f"{stage}失败：阿里云请求被限流或额度不足（429/Quota）。"
            "若多人共用同一 Key，建议改用独立 Key。"
        )
    if _is_permission_message(raw):
        return RuntimeError(f"{stage}失败：当前账号或 API Key 无该模型调用权限（403）。")
    if _is_model_message(raw):
        return RuntimeError(f"{stage}失败：目标模型不可用或当前账号无权限访问该模型。")
    return RuntimeError(f"{stage}失败：{_short_error(err)}")


def _extract_error_text_from_response(resp: requests.Response) -> str:
    try:
        payload: Any = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "").strip()
            code = str(err.get("code") or "").strip()
            if msg and code:
                return f"{msg} (code={code})"
            if msg:
                return msg
        msg = str(payload.get("message") or "").strip()
        if msg:
            return msg
    text = str(getattr(resp, "text", "") or "").strip()
    if text:
        return text[:240]
    return ""


def _format_exception_chain(err: BaseException) -> str:
    rows: List[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = err
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 8:
        seen.add(id(cur))
        rows.append(f"[{depth}] {type(cur).__name__}: {cur}")
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt
        depth += 1
    return "\n".join(rows)


def _write_failure_report(out_dir: Path, config: ConversionConfig, err: BaseException) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "failure_report.txt"
    try:
        tls_mode = core.current_tls_mode()
    except Exception:
        tls_mode = "unknown"
    try:
        tls_verify = str(core.current_tls_verify_arg())
    except Exception:
        tls_verify = "unknown"

    lines = [
        "Video Dub Studio Failure Report",
        f"time={datetime.now().isoformat()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={sys.platform}",
        f"tls_mode_selected={config.tls_mode}",
        f"tls_mode_effective={tls_mode}",
        f"tls_verify={tls_verify}",
        f"dashscope_api_url={config.dashscope_api_url}",
        f"qwen_openai_base_url={config.qwen_openai_base_url}",
        f"asr_model={config.asr_model}",
        f"tts_model={config.tts_model}",
        "",
        "exception_chain:",
        _format_exception_chain(err),
        "",
        "traceback:",
        traceback.format_exc(),
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _probe_dashscope_status(api_url: str) -> int:
    resp = requests.get(
        api_url,
        timeout=8,
        allow_redirects=True,
        verify=core.current_tls_verify_arg(),
    )
    return int(getattr(resp, "status_code", 0))


def _build_aiohttp_ssl_context() -> ssl.SSLContext:
    verify_arg = core.current_tls_verify_arg()
    if isinstance(verify_arg, str) and verify_arg:
        return ssl.create_default_context(cafile=verify_arg)
    return ssl.create_default_context()


def _probe_dashscope_status_aiohttp(api_url: str) -> int:
    if aiohttp is None:
        raise RuntimeError("aiohttp 不可用，无法执行 ASR 通道 TLS 探测")

    async def _run() -> int:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                api_url,
                allow_redirects=True,
                ssl=_build_aiohttp_ssl_context(),
            ) as resp:
                return int(getattr(resp, "status", 0))

    try:
        return int(asyncio.run(_run()))
    except RuntimeError as e:
        # Defensive fallback in case there is already an event loop.
        if "asyncio.run()" in str(e):
            loop = asyncio.new_event_loop()
            try:
                return int(loop.run_until_complete(_run()))
            finally:
                loop.close()
        raise


def _probe_dashscope_websocket_tls(api_url: str) -> str:
    if aiohttp is None:
        raise RuntimeError("aiohttp 不可用，无法执行 websocket TLS 探测")
    ws_url = core.websocket_base_url_from_http_api(api_url)

    async def _run() -> str:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.ws_connect(
                    ws_url,
                    ssl=_build_aiohttp_ssl_context(),
                    heartbeat=20,
                ) as ws:
                    await ws.close()
                    return "ok"
            except aiohttp.WSServerHandshakeError as e:
                # Handshake failed means endpoint rejected unauthenticated probe,
                # but TLS handshake itself has already passed.
                return f"hs:{int(getattr(e, 'status', 0))}"

    try:
        return str(asyncio.run(_run()))
    except RuntimeError as e:
        if "asyncio.run()" in str(e):
            loop = asyncio.new_event_loop()
            try:
                return str(loop.run_until_complete(_run()))
            finally:
                loop.close()
        raise


def _ensure_probe_status_ok(status: int, channel: str) -> None:
    if status == 407:
        raise RuntimeError(f"DashScope 网络初始化失败：{channel} 通道遇到代理认证（HTTP 407）。")
    if status >= 500:
        raise RuntimeError(f"DashScope 网络初始化失败：{channel} 通道服务暂不可用（HTTP {status}）。")


def _normalize_tls_mode(mode: str) -> str:
    norm = str(mode or "").strip().lower()
    if norm in {"auto", "system", "certifi", "custom_ca", "default"}:
        return norm
    return "system"


def _prepare_dashscope_tls(config: ConversionConfig, cb: ProgressCallback) -> str:
    requested = _normalize_tls_mode(config.tls_mode)
    mode_candidates: List[str]
    if requested == "auto":
        mode_candidates = ["system", "certifi", "default"]
    elif requested == "system":
        # Backward compatibility: many old local configs are fixed to system.
        # If system TLS fails on a machine, auto-fallback to certifi to avoid hard failure.
        mode_candidates = ["system", "certifi"]
    else:
        mode_candidates = [requested]

    last_error: Optional[Exception] = None
    for idx, mode in enumerate(mode_candidates, start=1):
        try:
            ca_file = config.ca_bundle_file if mode == "custom_ca" else ""
            effective_mode = core.configure_tls_trust(mode=mode, ca_bundle_file=ca_file)

            req_status = _probe_dashscope_status(config.dashscope_api_url)
            aio_status = _probe_dashscope_status_aiohttp(config.dashscope_api_url)
            ws_probe = _probe_dashscope_websocket_tls(config.dashscope_api_url)
            _ensure_probe_status_ok(req_status, "HTTP")
            _ensure_probe_status_ok(aio_status, "ASR")

            if requested == "auto" and idx > 1:
                cb(1, "网络初始化", f"TLS 自动切换成功: {effective_mode}")
            cb(
                1,
                "网络初始化",
                f"TLS 证书模式: {effective_mode} | HTTP探测:{req_status} | ASR探测:{aio_status} | WS探测:{ws_probe}",
            )
            return effective_mode
        except Exception as e:
            last_error = e
            if requested != "auto":
                if requested == "system" and idx < len(mode_candidates):
                    cb(1, "网络初始化", "系统证书链不可用，自动切换到 certifi 证书链重试...")
                    continue
                if _is_tls_cert_error_message(str(e)):
                    raise RuntimeError(
                        f"DashScope 网络初始化失败（TLS={mode}）。"
                        "证书链校验失败，请切换“自动”或“内置 certifi 证书链”，"
                        "企业网络下可导入自定义 CA 证书。"
                    ) from e
                raise RuntimeError(
                    f"DashScope 网络初始化失败（TLS={mode}）：{_short_error(e)}"
                ) from e
            cb(1, "网络初始化", f"TLS 候选 {mode} 不可用，继续尝试下一策略...")

    if last_error is not None and _is_tls_cert_error_message(str(last_error)):
        raise RuntimeError(
            "DashScope 网络初始化失败：自动 TLS 策略均无法通过证书校验。"
            "请切换到“自定义 CA 文件”并导入企业根证书，或更换网络后重试。"
        ) from last_error
    if last_error is not None:
        raise RuntimeError(f"DashScope 网络初始化失败：{_short_error(last_error)}") from last_error
    raise RuntimeError("DashScope 网络初始化失败：未知错误")


def _preflight_dashscope_access(config: ConversionConfig, cb: ProgressCallback) -> None:
    cb(2, "网络初始化", "正在验证阿里云 API Key 与模型访问权限...")
    endpoint = config.qwen_openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.qwen_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.translate_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }

    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=20,
            verify=core.current_tls_verify_arg(),
        )
    except Exception as e:
        raise _wrap_dashscope_stage_error("阿里云预检", e) from e

    if resp.status_code == 200:
        cb(2, "网络初始化", f"阿里云预检通过 | 翻译模型: {config.translate_model}")
        return

    err_text = _extract_error_text_from_response(resp)
    if resp.status_code == 401:
        raise RuntimeError("阿里云预检失败：API Key 无效、过期或输入错误（401）。")
    if resp.status_code == 403:
        raise RuntimeError(
            f"阿里云预检失败：当前账号/Key 无调用权限（403）。{err_text}"
        )
    if resp.status_code == 429:
        raise RuntimeError(
            "阿里云预检失败：请求限流或额度不足（429）。多人共用同一 Key 时常见。"
        )
    if resp.status_code == 400:
        if _is_model_message(err_text):
            raise RuntimeError(
                f"阿里云预检失败：翻译模型不可用或当前账号无权限（{config.translate_model}）。{err_text}"
            )
        if _is_auth_error_message(err_text):
            raise RuntimeError("阿里云预检失败：API Key 无效、过期或输入错误（401/400）。")
        raise RuntimeError(f"阿里云预检失败：请求参数异常（400）。{err_text}")
    if resp.status_code == 404:
        raise RuntimeError(
            f"阿里云预检失败：OpenAI 兼容接口地址无效（404）。请检查地址：{config.qwen_openai_base_url}"
        )
    if resp.status_code == 407:
        raise RuntimeError("阿里云预检失败：当前网络代理需要认证（HTTP 407）。")
    if resp.status_code >= 500:
        raise RuntimeError(f"阿里云预检失败：服务异常（HTTP {resp.status_code}）。")
    raise RuntimeError(f"阿里云预检失败：HTTP {resp.status_code} {err_text}")


def run_conversion(config: ConversionConfig, cb: ProgressCallback) -> ConversionResult:
    if not config.qwen_api_key.strip():
        raise RuntimeError("缺少 Qwen API Key")

    _inject_bundled_bin_to_path()
    _prepare_dashscope_tls(config, cb)
    core.require_binary("ffmpeg")
    core.require_binary("ffprobe")
    _preflight_dashscope_access(config, cb)

    source_label = "local"
    if config.source_url:
        try:
            source_label = _slugify(
                fetch_youtube_preview(
                    config.source_url,
                    cookie_browser=config.cookies_from_browser,
                    cookie_file=config.cookies_file,
                ).title
            )
        except Exception:
            source_label = "youtube"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(config.output_root or (str(PROJECT_ROOT / "output" / "gui_tasks"))).expanduser().resolve()
    out_dir = root / f"{timestamp}_{source_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    if config.keep_temp:
        temp_dir = out_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="tmp_", dir=str(out_dir))
        temp_dir = Path(tmp_obj.name)

    try:
        cb(3, "准备任务", f"输出目录: {out_dir}")

        if config.source_url:
            source_path = _download_video_with_progress(
                config.source_url,
                temp_dir,
                cb,
                cookie_browser=config.cookies_from_browser,
                cookie_file=config.cookies_file,
            )
        elif config.source_file:
            source_path = str(Path(config.source_file).expanduser().resolve())
            cb(10, "读取本地文件", Path(source_path).name)
        else:
            raise RuntimeError("必须提供 YouTube URL 或本地文件")

        cb(40, "提取音频", "音频标准化中...")
        normalized = str(temp_dir / "source_16k.wav")
        core.normalize_audio(source_path, normalized)
        asr_audio = normalized
        if config.max_seconds and config.max_seconds > 0:
            trimmed = str(temp_dir / "source_16k_trim.wav")
            core.trim_audio(normalized, trimmed, config.max_seconds)
            asr_audio = trimmed
        duration = core.ffprobe_duration(asr_audio)

        cb(47, "语音识别", "提交音频到识别服务...")
        hb_stop, hb_thread, hb_started_at = _start_asr_heartbeat(cb, audio_duration=duration)
        try:
            asr_result = core.asr_with_qwen(config.qwen_api_key, config.asr_model, asr_audio, config.dashscope_api_url)
        except Exception as e:
            raise _wrap_dashscope_stage_error("语音识别", e) from e
        finally:
            hb_stop.set()
            hb_thread.join(timeout=0.5)
        asr_elapsed = time.monotonic() - hb_started_at
        cb(59, "语音识别", f"识别完成，耗时 {_fmt_elapsed(asr_elapsed)}，正在解析结果...")
        asr_backend = str(asr_result.get("asr_backend", "")).strip()
        if asr_backend == "http_transcription":
            cb(59, "语音识别", "检测到 websocket TLS 异常，已自动切换为 HTTP 文件识别兜底")
        segments = core.normalize_segments(asr_result.get("segments", []), duration)
        speaker_profiles = core.normalize_speaker_profiles(asr_result)
        collapsed = core.maybe_collapse_to_single_speaker(segments)
        if collapsed:
            speaker_profiles = {collapsed: speaker_profiles.get(collapsed, core.SpeakerProfile())}
            cb(59, "说话人修正", "检测到标签抖动，已自动收敛为单说话人")
        unique_speakers = {s.speaker for s in segments}
        if len(unique_speakers) <= 1:
            seg_spk_map, inferred_profiles = core.infer_speakers_from_audio(
                segments=segments,
                wav_path=asr_audio,
                speaker_count=max(1, config.speaker_count),
            )
            if seg_spk_map:
                for idx, spk in seg_spk_map.items():
                    segments[idx].speaker = spk
                speaker_profiles.update(inferred_profiles)
                collapsed_after_infer = core.maybe_collapse_to_single_speaker(segments)
                if collapsed_after_infer:
                    speaker_profiles = {
                        collapsed_after_infer: speaker_profiles.get(collapsed_after_infer, core.SpeakerProfile())
                    }
                    cb(59, "说话人修正", "推断后检测到抖动，已回退为单说话人")

        translate_target, tts_language_type = core.OUTPUT_LANGUAGE_MAP.get(
            config.output_language, ("Simplified Chinese", "Chinese")
        )
        cb(60, "文本翻译", f"目标语言: {config.output_language}")
        try:
            core.translate_segments_qwen(
                segments=segments,
                api_key=config.qwen_api_key,
                base_url=config.qwen_openai_base_url,
                model=config.translate_model,
                batch_size=30,
                target_language=translate_target,
                progress_callback=lambda done, total: cb(
                    60 + int((done / max(1, total)) * 18),
                    "文本翻译",
                    f"{done}/{total}",
                ),
            )
        except Exception as e:
            raise _wrap_dashscope_stage_error("文本翻译", e) from e

        cb(79, "语音合成", "检测可用音色...")
        voices = [v.strip() for v in config.voices.split(",") if v.strip()]
        try:
            voices = core.list_supported_voices(config.qwen_api_key, config.dashscope_api_url, config.tts_model, voices)
        except Exception as e:
            raise _wrap_dashscope_stage_error("语音合成", e) from e
        if not voices:
            raise RuntimeError("当前 TTS 模型下无可用音色")
        speaker_voice_map = core.assign_voices(segments, voices, speaker_profiles)

        try:
            core.synthesize_all(
                segments=segments,
                temp_dir=temp_dir,
                api_key=config.qwen_api_key,
                tts_model=config.tts_model,
                api_url=config.dashscope_api_url,
                language_type=tts_language_type,
                progress_callback=lambda done, total: cb(
                    79 + int((done / max(1, total)) * 13),
                    "语音合成",
                    f"{done}/{total}",
                ),
            )
        except Exception as e:
            raise _wrap_dashscope_stage_error("语音合成", e) from e

        cb(93, "音频混合", "对齐时间轴...")
        dubbed_audio_path = str(out_dir / "dubbed_audio.m4a")
        core.mix_segments(segments, duration, dubbed_audio_path)

        dubbed_video_path = ""
        if _has_video_stream(source_path):
            cb(97, "视频合成", "封装新音轨...")
            dubbed_video_path = str(out_dir / "dubbed_video.mp4")
            core.mux_video_with_audio(source_path, dubbed_audio_path, dubbed_video_path)

        core.save_artifacts(out_dir, source_path, dubbed_audio_path, speaker_voice_map, segments)
        manifest_path = str(out_dir / "manifest.json")
        segments_path = str(out_dir / "segments.json")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest.update(
            {
                "output_language": config.output_language,
                "dubbed_video": dubbed_video_path,
                "source_url": config.source_url,
            }
        )
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cb(100, "任务完成", "全部完成")

        return ConversionResult(
            output_dir=str(out_dir),
            source_path=source_path,
            dubbed_audio_path=dubbed_audio_path,
            dubbed_video_path=dubbed_video_path,
            manifest_path=manifest_path,
            segments_path=segments_path,
            speaker_voice_map=speaker_voice_map,
            segment_count=len(segments),
        )
    except Exception as e:
        try:
            report = _write_failure_report(out_dir, config, e)
            cb(0, "诊断信息", f"失败报告已写入: {report}")
            raise RuntimeError(f"{e}\n诊断报告: {report}") from e
        except Exception:
            raise
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()
