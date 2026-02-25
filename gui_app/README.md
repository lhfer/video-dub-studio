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
  - 自动（推荐）：系统证书链 -> certifi 回退
  - 系统证书链（企业网络优先）
  - 内置 certifi 证书链
  - 自定义 CA 文件（PEM）
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
