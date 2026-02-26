# Video Dub Studio GUI

桌面应用目标：把 YouTube/本地视频转换为多角色配音后的新音频和新视频，并提供进度、历史和文件管理。

## 运行（开发模式）

```bash
cd "/Users/bytedance/Documents/codex project/youtube-cn-dub"
source .venv312/bin/activate
pip install -r gui_app/requirements-gui.txt
./gui_app/run_gui.sh
```

## 打包 `.app` + `.dmg`

```bash
cd "/Users/bytedance/Documents/codex project/youtube-cn-dub"
./gui_app/build_dmg.sh
```

若仅重新打 DMG（跳过重编译）：

```bash
cd "/Users/bytedance/Documents/codex project/youtube-cn-dub"
SKIP_BUILD=1 ./gui_app/build_dmg.sh
```

输出：
- `.app`: `dist/VideoDubStudio.app`
- `.dmg`: `dist/VideoDubStudio.dmg`

说明：
- 打包产物已内置 `ffmpeg/ffprobe/deno`，分享给其他人时无需额外安装。
- 已集成 `yt-dlp-ejs`，用于 YouTube JS challenge 解析，降低“有预览但无可下载格式”的概率。
- 已加入 TLS 证书兼容层，并支持 GUI 切换：
  - 自动（推荐）：同时探测 HTTP 与 ASR 通道，自动选择可用策略
  - 系统证书链（企业网络优先）
  - 内置 certifi 证书链
  - 自定义 CA 文件（PEM）
- 任务启动前会先做阿里云预检（API Key / 模型权限 / TLS 连通性），失败会在下载前直接提示。
- ASR 在 websocket 证书失败时会自动切换到 HTTP 文件识别兜底（上传后识别），减少“仅语音识别阶段 TLS 失败”。
- 失败时会在任务输出目录生成 `failure_report.txt`（包含异常链与 TLS 诊断），便于远程排障。
- 翻译阶段已增加 JSON 容错与逐句回退，避免单批次格式异常导致整任务失败。
- 默认不保留中间分段文件；仅在勾选“保留中间文件（调试）”时才会保留 `tmp` 目录。
- `Qwen API Key` 首次输入后会保存到 `~/.video-dub-studio/config.json`，后续启动自动回填。
- `cookies.txt` 路径同样会保存到 `~/.video-dub-studio/config.json`，便于下次直接复用。

## 已实现功能

1. 输入 YouTube URL 后预览封面/标题/简介/时长/语言。
2. 可选输出语言（含自动识别源语言展示）。
3. 可见下载进度和转换阶段进度。
4. 输出配音音频与新视频（保留原视频画面，替换音轨，时间轴对齐）。
5. 任务历史记录（可直接打开输出目录/音频/视频）。
6. 自动说话人分离兜底（ASR 无 speaker_id 时），并按性别/语气匹配音色。
7. 说话人标签稳定化（短时抖动自动平滑，单说话人场景优先收敛，减少男女声来回切换）。

## YouTube 风控兜底：cookies.txt

当 YouTube 返回 `sign in to confirm you're not a bot`、`Requested format is not available` 等风控相关问题时，可导入 `cookies.txt` 提高成功率。

应用内操作：
1. 在 `2) 配置输出` 中找到 `cookies.txt（可选）`。
2. 点击 `导入文件` 选择本地 `cookies.txt`。
3. 再次点击 `获取预览` 或 `开始转换`。

如何获取 `cookies.txt`：
1. 在浏览器登录 YouTube，并确认目标视频可正常播放。
2. 安装可导出 cookies 的浏览器扩展（例如 `Get cookies.txt LOCALLY`）。
3. 在 `youtube.com` 页面导出 cookies，格式选择 `Netscape`，保存为 `cookies.txt`。
4. 回到应用导入该文件。

安全建议：
- `cookies.txt` 是登录凭证，请勿外传。
- 账号状态变化、退出登录或 cookies 过期后需要重新导出。

## TLS 证书排障（ASR/TTS 证书错误）

当报错包含 `CERTIFICATE_VERIFY_FAILED` 或 `unable to get local issuer certificate` 时：
1. 在 `2) 配置输出` 把 `TLS 证书模式` 切到 `系统证书链（企业网络优先）` 后重试。
2. 若仍失败，切到 `自定义 CA 文件（高级）`，导入公司代理/安全网关提供的 `PEM` 证书再试。
3. 若依旧失败，关闭代理或切换网络测试，确认不是本地网络设备重签导致。

## 一键环境诊断（推荐给“同事机器失败”的场景）

### 1) 采集报告
```bash
cd "/Users/bytedance/Documents/codex project/youtube-cn-dub"
./scripts/collect_vds_diag.sh
```

默认在桌面生成：`vds_diag_<hostname>_<timestamp>.txt`

### 2) 对比两台机器报告
```bash
python3 ./scripts/compare_vds_diag.py /path/your_report.txt /path/friend_report.txt
```

建议同时提供：
- `vds_diag_*.txt`（环境报告）
- 任务目录下的 `failure_report.txt`（任务级异常链）
