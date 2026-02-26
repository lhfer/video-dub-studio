#!/usr/bin/env bash
set -u

OUT_PATH="${1:-}"
HOST="$(hostname -s 2>/dev/null || hostname || echo unknown)"
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_PATH" ]]; then
  OUT_PATH="$HOME/Desktop/vds_diag_${HOST}_${TS}.txt"
fi

mkdir -p "$(dirname "$OUT_PATH")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

section() {
  printf '\n===== %s =====\n' "$1"
}

run_cmd() {
  local label="$1"
  shift
  printf '\n--- %s ---\n' "$label"
  "$@" 2>&1 || true
}

PY_BIN=""
if [[ -x "$REPO_ROOT/.venv312/bin/python" ]]; then
  PY_BIN="$REPO_ROOT/.venv312/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
fi

{
  echo "Video Dub Studio Environment Diagnostics"
  echo "GeneratedAt=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "GeneratedBy=$HOST"
  echo "OutputFile=$OUT_PATH"

  echo "KV|host|$HOST"
  echo "KV|date_local|$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "KV|date_utc|$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

  section "System"
  run_cmd "uname" uname -a
  run_cmd "sw_vers" sw_vers
  run_cmd "whoami" whoami
  run_cmd "arch" arch

  section "Network Proxy"
  run_cmd "scutil --proxy" scutil --proxy
  printf '\n--- proxy env ---\n'
  env | grep -Ei '^(http|https|all|no)_proxy=' || true

  section "DNS"
  run_cmd "dscacheutil dashscope" dscacheutil -q host -a name dashscope.aliyuncs.com
  run_cmd "dig dashscope" dig +short dashscope.aliyuncs.com
  run_cmd "dig oss beijing" dig +short dashscope-instant.oss-cn-beijing.aliyuncs.com
  run_cmd "dig oss hangzhou" dig +short dashscope-instant.oss-cn-hangzhou.aliyuncs.com

  section "curl TLS (DashScope)"
  CURL_OUT="$(curl -Iv --connect-timeout 12 https://dashscope.aliyuncs.com/api/v1 2>&1 || true)"
  echo "$CURL_OUT"
  CURL_ISSUER="$(printf '%s\n' "$CURL_OUT" | grep -Eio 'issuer:.*' | head -n1 | sed 's/^issuer:[[:space:]]*//')"
  CURL_VERIFY="no"
  if printf '%s\n' "$CURL_OUT" | grep -Eqi 'SSL certificate verify ok'; then
    CURL_VERIFY="yes"
  fi
  echo "KV|curl_dashscope_issuer|${CURL_ISSUER:-unknown}"
  echo "KV|curl_dashscope_verify_ok|$CURL_VERIFY"

  section "OpenSSL Handshake (DashScope)"
  run_cmd "openssl s_client" bash -lc "echo | openssl s_client -connect dashscope.aliyuncs.com:443 -servername dashscope.aliyuncs.com 2>/dev/null | sed -n '1,40p'"

  section "Python TLS Probes"
  if [[ -n "$PY_BIN" ]]; then
    "$PY_BIN" - <<'PY' || true
import asyncio
import ssl
import socket
import traceback

try:
    import requests
except Exception as e:
    requests = None
    requests_err = repr(e)

try:
    import aiohttp
except Exception as e:
    aiohttp = None
    aiohttp_err = repr(e)

try:
    import certifi
except Exception:
    certifi = None

HTTP_URL = "https://dashscope.aliyuncs.com/api/v1"
WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


def kv(k, v):
    print(f"KV|{k}|{str(v).replace(chr(10), ' ')[:500]}")


def show_cert(label, ctx):
    try:
        with socket.create_connection(("dashscope.aliyuncs.com", 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname="dashscope.aliyuncs.com") as ssock:
                cert = ssock.getpeercert()
                subject = cert.get("subject", "")
                issuer = cert.get("issuer", "")
                kv(f"{label}_subject", subject)
                kv(f"{label}_issuer", issuer)
                kv(f"{label}_version", ssock.version())
    except Exception as e:
        kv(f"{label}_cert_error", repr(e))


def req_probe(name, verify):
    if requests is None:
        kv(f"requests_{name}", f"not_available:{requests_err}")
        return
    try:
        r = requests.get(HTTP_URL, timeout=12, verify=verify, allow_redirects=True)
        kv(f"requests_{name}", f"ok_status:{r.status_code}")
    except Exception as e:
        kv(f"requests_{name}", f"error:{type(e).__name__}:{e}")


async def aio_get_probe(name, ssl_ctx):
    if aiohttp is None:
        kv(f"aiohttp_get_{name}", f"not_available:{aiohttp_err}")
        return
    try:
        timeout = aiohttp.ClientTimeout(total=12)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(HTTP_URL, ssl=ssl_ctx, allow_redirects=True) as resp:
                kv(f"aiohttp_get_{name}", f"ok_status:{resp.status}")
    except Exception as e:
        kv(f"aiohttp_get_{name}", f"error:{type(e).__name__}:{e}")


async def aio_ws_probe(name, ssl_ctx):
    if aiohttp is None:
        kv(f"aiohttp_ws_{name}", f"not_available:{aiohttp_err}")
        return
    try:
        timeout = aiohttp.ClientTimeout(total=12)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.ws_connect(WS_URL, ssl=ssl_ctx, heartbeat=10) as ws:
                await ws.close()
                kv(f"aiohttp_ws_{name}", "ok")
    except aiohttp.WSServerHandshakeError as e:
        kv(f"aiohttp_ws_{name}", f"handshake_status:{e.status}")
    except Exception as e:
        kv(f"aiohttp_ws_{name}", f"error:{type(e).__name__}:{e}")


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


kv("python_executable", __import__("sys").executable)
kv("python_version", __import__("sys").version)
kv("requests_version", getattr(requests, "__version__", "not_installed"))
kv("aiohttp_version", getattr(aiohttp, "__version__", "not_installed"))
kv("certifi_where", certifi.where() if certifi else "not_installed")

ctx_default = ssl.create_default_context()
show_cert("ssl_default", ctx_default)

if certifi:
    ctx_certifi = ssl.create_default_context(cafile=certifi.where())
else:
    ctx_certifi = None

req_probe("default", True)
if certifi:
    req_probe("certifi", certifi.where())

run_async(aio_get_probe("default", ctx_default))
run_async(aio_ws_probe("default", ctx_default))

if ctx_certifi is not None:
    show_cert("ssl_certifi", ctx_certifi)
    run_async(aio_get_probe("certifi", ctx_certifi))
    run_async(aio_ws_probe("certifi", ctx_certifi))
PY
  else
    echo "python not found"
  fi

  section "Summary"
  echo "请把本文件发给开发者；若要对比两台机器，再运行 compare_vds_diag.py。"
} | tee "$OUT_PATH"

printf '\nDONE: %s\n' "$OUT_PATH"
