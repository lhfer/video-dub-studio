# Video Dub Studio

将 YouTube / 本地视频一键转成多语种配音音轨，专为“访谈后台听”设计。  
模型链路全走阿里云 Qwen（ASR + 翻译 + TTS）。

## 为什么这个项目可能会被很多人用到
很多人看英文访谈时，真正痛点不是“看不懂”，而是：
- 必须盯着字幕，无法像播客一样后台听
- 长访谈信息密度高，通勤/运动场景没法消化
- 自动配音工具经常音色乱跳，听感割裂

Video Dub Studio 解决的是这个完整链路：
- 识别说话内容
- 翻译成目标语言
- 按说话人音色做配音
- 对齐时间轴并封装回视频

## 核心能力
- YouTube 链接预览：封面、标题、简介、时长、语言检测
- 本地视频/音频转换：无需上传原视频到第三方站点
- 多语输出：`zh-CN` / `en-US` / `ja-JP` / `es-ES` / `fr-FR`
- 说话人音色映射：基于音频特征做人数与风格推断
- 说话人稳定化：减少单人场景“男声/女声来回切”
- 音画同步：合成后自动封装新音轨回视频
- 历史任务中心：快速打开输出目录、音频、视频

## 模型与技术栈（Qwen-only）
- ASR: `paraformer-realtime-v2`
- Translation: `qwen3-max`
- TTS: `qwen3-tts-flash` / `qwen3-tts-instruct-flash`
- GUI: PySide6
- 下载与解析: yt-dlp + yt-dlp-ejs
- 媒体处理: ffmpeg / ffprobe

## 快速开始

### 1) GUI（推荐）
```bash
cd youtube-cn-dub
source .venv312/bin/activate
pip install -r gui_app/requirements-gui.txt
./gui_app/run_gui.sh
```

### 2) CLI
```bash
cd youtube-cn-dub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

YouTube 示例：
```bash
python src/dub_pipeline.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output-dir "./output/demo" \
  --qwen-api-key "YOUR_DASHSCOPE_KEY" \
  --asr-model paraformer-realtime-v2 \
  --translate-model qwen3-max \
  --tts-model qwen3-tts-flash \
  --output-language zh-CN
```

本地文件示例：
```bash
python src/dub_pipeline.py \
  --input-audio "/absolute/path/to/video.mp4" \
  --output-dir "./output/local-demo" \
  --qwen-api-key "YOUR_DASHSCOPE_KEY"
```

## 打包桌面应用（macOS DMG）
```bash
cd youtube-cn-dub
./gui_app/build_dmg.sh
```
产物：
- `dist/VideoDubStudio.app`
- `dist/VideoDubStudio.dmg`

## 关键参数建议
- `speaker_count`：最大说话人数上限（`1` 可强制单人稳定模式）
- `max_seconds`：长视频先抽样前 N 秒，先验证质量再全量跑
- `tts_model`：
  - `qwen3-tts-flash`：速度优先
  - `qwen3-tts-instruct-flash`：表达控制更强

## 常见问题

### 1) YouTube 提示风控 / 格式不可用
- 先尝试在 GUI 导入 `cookies.txt`（Netscape 格式）
- 确认目标视频在浏览器内可播放
- 尽量使用最新版打包应用（内置 deno + ejs 兼容链路）

### 2) 阿里云接口证书错误（CERTIFICATE_VERIFY_FAILED）
GUI 支持 TLS 模式切换：
- 自动（推荐）
- 系统证书链
- certifi 证书链
- 自定义 CA（PEM）

适用于企业代理 / 安全网关重签证书场景。

### 3) 同公司机器表现不一致（你能跑、同事失败）
请先用仓库内诊断脚本采集两台机器报告，再对比：

```bash
cd "/Users/bytedance/Documents/codex project/youtube-cn-dub"
./scripts/collect_vds_diag.sh
```

默认会在桌面生成 `vds_diag_<hostname>_<timestamp>.txt`。  
然后对比两份报告：

```bash
python3 ./scripts/compare_vds_diag.py /path/A.txt /path/B.txt
```

该报告会包含：
- 代理配置（`scutil --proxy`）
- DashScope 证书签发者（issuer）
- Python `requests` / `aiohttp` 的 HTTP 与 WebSocket TLS 探测
- certifi 与系统证书链差异

另外，GUI 每次失败会在任务目录生成 `failure_report.txt`，也请一并提供。

### 4) 文本翻译阶段 JSON 解析报错
已在新版本加入翻译结果容错解析与逐句回退策略，避免因单批次返回格式异常导致整任务失败。

## 输出文件
- `dubbed_audio.m4a`: 最终配音音频
- `dubbed_video.mp4`: 新视频（原画面 + 新音轨）
- `manifest.json`: 任务摘要
- `segments.json`: 分段结果（时间、说话人、文本、音色）

## 路线图
- 更强说话人建模（embedding 级别特征）
- 可选保留原声底噪（ducking）
- 批量任务队列与并发策略
- Windows 打包支持

## 免责声明
请遵守平台服务条款、版权和当地法律法规，仅在你有权处理的内容上使用本工具。

---
如果这个项目对你有帮助，欢迎 Star。
