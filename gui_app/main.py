from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QComboBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from history_store import HistoryStore, build_history_row
from pipeline_runner import ConversionConfig, ConversionResult, fetch_youtube_preview, run_conversion

APP_HOME = Path.home() / ".video-dub-studio"
CONFIG_PATH = APP_HOME / "config.json"
APP_VERSION = "0.3.30"
LANG_OPTIONS = [
    ("简体中文", "zh-CN"),
    ("English", "en-US"),
    ("日本語", "ja-JP"),
    ("Español", "es-ES"),
    ("Français", "fr-FR"),
]
LANG_LABEL = {code: label for label, code in LANG_OPTIONS}
COOKIE_BROWSER_OPTIONS = [
    ("不使用（默认）", "none"),
    ("Chrome（已登录 YouTube）", "chrome"),
    ("Safari（已登录 YouTube）", "safari"),
    ("Firefox（已登录 YouTube）", "firefox"),
    ("Edge（已登录 YouTube）", "edge"),
]
TLS_MODE_OPTIONS = [
    ("自动（推荐）", "auto"),
    ("系统证书链（企业网络优先）", "system"),
    ("内置 certifi 证书链", "certifi"),
    ("自定义 CA 文件（高级）", "custom_ca"),
]
VOICE_PRESETS = {
    "auto": "Ethan,Dylan,Sunny,Cherry,Serena,Chelsie,Bella",
    "male_first": "Ethan,Dylan,Sunny,Serena,Cherry",
    "female_first": "Serena,Cherry,Chelsie,Bella,Ethan",
    "male_only": "Ethan,Dylan,Sunny",
    "female_only": "Serena,Cherry,Chelsie,Bella",
}

TOOLTIPS = {
    "youtube_url": "输入 YouTube 链接后可获取封面、标题、简介和语言检测。",
    "local_file": "本地视频/音频文件路径。与 YouTube 链接二选一。",
    "api_key": "阿里云百炼 API Key。首次输入后会保存在本机 ~/.video-dub-studio/config.json。",
    "output_language": "要合成输出的目标语言。源语言会自动识别。",
    "output_dir": "转换结果输出目录。每次任务会生成独立时间戳子目录。",
    "youtube_cookies": "默认不读取浏览器登录态；仅在触发 YouTube 风控时按需使用。应用内置 JS challenge 运行时和增强解析策略。Safari 可能需要“完全磁盘访问”权限。",
    "cookies_file": "可选兜底方案。导入 Netscape 格式 cookies.txt 后，YouTube 风控环境下成功率更高。",
    "tls_mode": "网络证书校验策略。默认“自动”会同时验证 HTTP 与 ASR 通道，并自动选择可用策略。",
    "ca_bundle_file": "仅在“自定义 CA 文件”模式下生效。选择 PEM 格式证书文件，用于企业代理/网关证书场景。",
    "max_seconds": "限制处理前 N 秒，0 表示处理完整音频。",
    "speaker_count": "高级参数。系统会自动识别说话人数；该值用于限制最多识别人数，填 1 可强制单人音色。",
    "tts_model": "qwen3-tts-flash 速度优先；qwen3-tts-instruct-flash 风格控制更强。",
    "voice_preset": "选择音色方案。推荐使用“自动推荐”。",
    "voices": "仅在“自定义音色”时填写，英文逗号分隔，例如：Ethan,Serena,Cherry",
    "keep_temp": "是否保留中间分段文件（seg_*.wav）。关闭后只保留最终音频/视频。",
}


class ConversionWorker(QObject):
    progress = Signal(int, str, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ConversionConfig):
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            result = run_conversion(self.config, self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Dub Studio {APP_VERSION}")
        self.setMinimumSize(900, 660)
        self.resize(1320, 860)

        self.preview_title = ""
        self.preview_url = ""
        self.preview_detected_lang = "en-US"
        self.thumb_pixmap: Optional[QPixmap] = None

        self.history_store = HistoryStore(APP_HOME)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ConversionWorker] = None

        self._build_layout()
        self._load_config()
        self._refresh_history()

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        root_layout.addWidget(scroll)

        content = QWidget()
        content.setObjectName("AppSurface")
        scroll.setWidget(content)

        main = QVBoxLayout(content)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(14)

        title = QLabel("AI 视频多语配音工作台")
        title.setObjectName("HeroTitle")
        subtitle = QLabel(f"输入视频链接，自动识别语言和说话人，输出对应语言的新音轨与新视频  ·  v{APP_VERSION}")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)

        main.addWidget(title)
        main.addWidget(subtitle)

        source_card = self._build_source_card()
        control_card = self._build_control_card()
        progress_card = self._build_progress_card()
        history_card = self._build_history_card()

        for card in (source_card, control_card, progress_card, history_card):
            self._apply_glass_effect(card)
            main.addWidget(card)

        main.addStretch(1)

    def _apply_glass_effect(self, widget: QWidget) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(42, 63, 89, 45))
        widget.setGraphicsEffect(shadow)

    def _tip_label(self, text: str, tip_key: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(TOOLTIPS[tip_key])
        label.setStatusTip(TOOLTIPS[tip_key])
        return label

    def _build_source_card(self) -> QWidget:
        box = QGroupBox("1) 选择视频源并预览")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        layout.setColumnStretch(1, 1)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("粘贴 YouTube 地址，例如: https://www.youtube.com/watch?v=...")
        self.url_input.setToolTip(TOOLTIPS["youtube_url"])
        self.fetch_btn = QPushButton("获取预览")
        self.fetch_btn.clicked.connect(self.fetch_preview_clicked)

        self.local_input = QLineEdit()
        self.local_input.setPlaceholderText("或选择本地视频/音频文件")
        self.local_input.setToolTip(TOOLTIPS["local_file"])
        self.local_btn = QPushButton("选择文件")
        self.local_btn.clicked.connect(self.select_local_file)

        self.thumb = QLabel("封面预览")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setMinimumSize(280, 158)
        self.thumb.setObjectName("Thumb")

        self.title_label = QLabel("标题：")
        self.title_label.setWordWrap(True)
        self.meta_label = QLabel("作者：-    时长：-    检测语言：-")
        self.meta_label.setWordWrap(True)
        self.desc = QTextEdit()
        self.desc.setReadOnly(True)
        self.desc.setPlaceholderText("简介预览")
        self.desc.setMinimumHeight(120)

        layout.addWidget(self._tip_label("YouTube URL", "youtube_url"), 0, 0)
        layout.addWidget(self.url_input, 0, 1)
        layout.addWidget(self.fetch_btn, 0, 2)

        layout.addWidget(self._tip_label("本地文件", "local_file"), 1, 0)
        layout.addWidget(self.local_input, 1, 1)
        layout.addWidget(self.local_btn, 1, 2)

        layout.addWidget(self.thumb, 2, 0, 3, 1)
        layout.addWidget(self.title_label, 2, 1, 1, 2)
        layout.addWidget(self.meta_label, 3, 1, 1, 2)
        layout.addWidget(self.desc, 4, 1, 1, 2)

        hint = QLabel("提示：链接模式和本地文件模式只能二选一。")
        hint.setObjectName("HintText")
        layout.addWidget(hint, 5, 0, 1, 3)
        return box

    def _build_control_card(self) -> QWidget:
        box = QGroupBox("2) 配置输出")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("输入阿里云百炼 API Key")
        self.key_input.setToolTip(TOOLTIPS["api_key"])
        self.show_key_btn = QToolButton()
        self.show_key_btn.setText("显示")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visible)

        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)
        key_layout.addWidget(self.key_input, 1)
        key_layout.addWidget(self.show_key_btn)

        self.output_lang_combo = QComboBox()
        for label, code in LANG_OPTIONS:
            self.output_lang_combo.addItem(label, code)
        self.output_lang_combo.setToolTip(TOOLTIPS["output_language"])

        self.cookies_combo = QComboBox()
        for label, code in COOKIE_BROWSER_OPTIONS:
            self.cookies_combo.addItem(label, code)
        self.cookies_combo.setToolTip(TOOLTIPS["youtube_cookies"])

        self.cookies_file_input = QLineEdit()
        self.cookies_file_input.setPlaceholderText("可选：导入 cookies.txt（Netscape 格式）")
        self.cookies_file_input.setToolTip(TOOLTIPS["cookies_file"])
        cookie_file_btn = QPushButton("导入文件")
        cookie_file_btn.clicked.connect(self.select_cookie_file)
        cookie_file_clear_btn = QPushButton("清除")
        cookie_file_clear_btn.clicked.connect(self.clear_cookie_file)
        cookie_file_help_btn = QPushButton("如何获取？")
        cookie_file_help_btn.clicked.connect(self.show_cookie_help)
        cookie_perm_btn = QPushButton("打开权限")
        cookie_perm_btn.clicked.connect(self.open_full_disk_access_settings)
        cookie_file_row = QWidget()
        cookie_file_layout = QHBoxLayout(cookie_file_row)
        cookie_file_layout.setContentsMargins(0, 0, 0, 0)
        cookie_file_layout.setSpacing(8)
        cookie_file_layout.addWidget(self.cookies_file_input, stretch=1)
        cookie_file_layout.addWidget(cookie_file_btn)
        cookie_file_layout.addWidget(cookie_file_clear_btn)
        cookie_file_layout.addWidget(cookie_file_help_btn)
        cookie_file_layout.addWidget(cookie_perm_btn)
        self.cookie_file_hint = QLabel("优先使用 cookies.txt；未填写时才按浏览器登录态读取。")
        self.cookie_file_hint.setObjectName("HintText")

        self.tls_mode_combo = QComboBox()
        for label, code in TLS_MODE_OPTIONS:
            self.tls_mode_combo.addItem(label, code)
        self.tls_mode_combo.setToolTip(TOOLTIPS["tls_mode"])
        self.tls_mode_combo.currentIndexChanged.connect(self._on_tls_mode_changed)

        self.ca_bundle_input = QLineEdit()
        self.ca_bundle_input.setPlaceholderText("可选：选择 PEM 证书文件（仅自定义 CA 模式）")
        self.ca_bundle_input.setToolTip(TOOLTIPS["ca_bundle_file"])
        ca_file_btn = QPushButton("选择文件")
        ca_file_btn.clicked.connect(self.select_ca_bundle_file)
        ca_file_clear_btn = QPushButton("清除")
        ca_file_clear_btn.clicked.connect(self.clear_ca_bundle_file)
        ca_file_row = QWidget()
        ca_file_layout = QHBoxLayout(ca_file_row)
        ca_file_layout.setContentsMargins(0, 0, 0, 0)
        ca_file_layout.setSpacing(8)
        ca_file_layout.addWidget(self.ca_bundle_input, stretch=1)
        ca_file_layout.addWidget(ca_file_btn)
        ca_file_layout.addWidget(ca_file_clear_btn)
        self.ca_hint = QLabel("自定义 CA 仅在企业代理/安全网关替换证书时需要。")
        self.ca_hint.setObjectName("HintText")

        self.output_root_input = QLineEdit(str((Path.home() / "Movies" / "VideoDubStudio").resolve()))
        self.output_root_input.setToolTip(TOOLTIPS["output_dir"])
        out_btn = QPushButton("选择目录")
        out_btn.clicked.connect(self.select_output_root)
        out_row = QWidget()
        out_row_layout = QHBoxLayout(out_row)
        out_row_layout.setContentsMargins(0, 0, 0, 0)
        out_row_layout.setSpacing(8)
        out_row_layout.addWidget(self.output_root_input, stretch=1)
        out_row_layout.addWidget(out_btn)

        self.max_seconds = QSpinBox()
        self.max_seconds.setRange(0, 7200)
        self.max_seconds.setValue(0)
        self.max_seconds.setSuffix(" 秒（0=全量）")
        self.max_seconds.setToolTip(TOOLTIPS["max_seconds"])

        self.speaker_count = QSpinBox()
        self.speaker_count.setRange(1, 8)
        self.speaker_count.setValue(2)
        self.speaker_count.setToolTip(TOOLTIPS["speaker_count"])
        self.speaker_hint = QLabel("默认自动识别说话人数；该值用于限制最多人数（1=单人稳定模式）。")
        self.speaker_hint.setObjectName("HintText")

        self.tts_combo = QComboBox()
        self.tts_combo.addItem("qwen3-tts-flash（速度优先）", "qwen3-tts-flash")
        self.tts_combo.addItem("qwen3-tts-instruct-flash（风格控制）", "qwen3-tts-instruct-flash")
        self.tts_combo.setToolTip(TOOLTIPS["tts_model"])

        self.voice_preset_combo = QComboBox()
        self.voice_preset_combo.addItem("自动推荐（默认）", "auto")
        self.voice_preset_combo.addItem("访谈男声优先", "male_first")
        self.voice_preset_combo.addItem("访谈女声优先", "female_first")
        self.voice_preset_combo.addItem("仅男声", "male_only")
        self.voice_preset_combo.addItem("仅女声", "female_only")
        self.voice_preset_combo.addItem("自定义音色", "custom")
        self.voice_preset_combo.setToolTip(TOOLTIPS["voice_preset"])
        self.voice_preset_combo.currentIndexChanged.connect(self._on_voice_preset_changed)

        self.voices_input = QLineEdit(VOICE_PRESETS["auto"])
        self.voices_input.setToolTip(TOOLTIPS["voices"])
        self.voices_input.setEnabled(False)
        self.voices_hint = QLabel("若选“自定义音色”，按英文逗号填写，例如：Ethan,Serena,Cherry")
        self.voices_hint.setObjectName("HintText")

        self.keep_temp_checkbox = QCheckBox("保留中间文件（调试）")
        self.keep_temp_checkbox.setChecked(False)
        self.keep_temp_checkbox.setToolTip(TOOLTIPS["keep_temp"])
        self.keep_temp_hint = QLabel("默认关闭：仅保留最终输出，避免看到大量中间片段文件。")
        self.keep_temp_hint.setObjectName("HintText")

        sized_inputs = [
            self.key_input,
            self.output_lang_combo,
            self.output_root_input,
            self.cookies_combo,
            self.cookies_file_input,
            self.tls_mode_combo,
            self.ca_bundle_input,
            self.max_seconds,
            self.speaker_count,
            self.tts_combo,
            self.voice_preset_combo,
            self.voices_input,
        ]
        for w in sized_inputs:
            w.setMinimumHeight(34)

        self.persist_hint = QLabel("API Key 仅需输入一次；任务启动后会自动保存到本机配置。")
        self.persist_hint.setObjectName("HintText")

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(12)
        self.start_btn = QPushButton("开始转换")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.start_clicked)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked)
        action_layout.addStretch(1)
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addStretch(1)

        form.addRow(self._tip_label("Qwen API Key", "api_key"), key_row)
        form.addRow(self._tip_label("输出语言", "output_language"), self.output_lang_combo)
        form.addRow(self._tip_label("输出目录", "output_dir"), out_row)
        form.addRow(self._tip_label("YouTube 登录态（可选）", "youtube_cookies"), self.cookies_combo)
        form.addRow(self._tip_label("cookies.txt（可选）", "cookies_file"), cookie_file_row)
        form.addRow("", self.cookie_file_hint)
        form.addRow(self._tip_label("TLS 证书模式", "tls_mode"), self.tls_mode_combo)
        form.addRow(self._tip_label("CA 证书文件（可选）", "ca_bundle_file"), ca_file_row)
        form.addRow("", self.ca_hint)
        form.addRow(self._tip_label("处理时长", "max_seconds"), self.max_seconds)
        form.addRow(self._tip_label("说话人数（高级）", "speaker_count"), self.speaker_count)
        form.addRow("", self.speaker_hint)
        form.addRow(self._tip_label("TTS 模型", "tts_model"), self.tts_combo)
        form.addRow(self._tip_label("音色方案", "voice_preset"), self.voice_preset_combo)
        form.addRow(self._tip_label("自定义音色（高级）", "voices"), self.voices_input)
        form.addRow("", self.voices_hint)
        form.addRow(self._tip_label("中间文件", "keep_temp"), self.keep_temp_checkbox)
        form.addRow("", self.keep_temp_hint)
        form.addRow("", self.persist_hint)
        form.addRow("", action_row)
        self._on_tls_mode_changed()
        return box

    def _build_progress_card(self) -> QWidget:
        box = QGroupBox("3) 任务进度")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.stage_label = QLabel("等待任务开始")
        self.stage_label.setWordWrap(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setMaximumHeight(220)

        layout.addWidget(self.progress)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.log)
        return box

    def _build_history_card(self) -> QWidget:
        box = QGroupBox("4) 历史任务")
        layout = QVBoxLayout(box)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["时间", "标题", "状态", "输出语言", "音频", "视频"])
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.history_table.setMinimumHeight(170)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)

        open_output_btn = QPushButton("打开输出目录")
        open_output_btn.clicked.connect(lambda: self.open_history_path("output_dir"))
        open_audio_btn = QPushButton("打开音频")
        open_audio_btn.clicked.connect(lambda: self.open_history_path("audio_path"))
        open_video_btn = QPushButton("打开视频")
        open_video_btn.clicked.connect(lambda: self.open_history_path("video_path"))
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_history)

        btn_layout.addWidget(open_output_btn)
        btn_layout.addWidget(open_audio_btn)
        btn_layout.addWidget(open_video_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(refresh_btn)

        layout.addWidget(self.history_table)
        layout.addWidget(btn_row)
        return box

    def _toggle_key_visible(self, checked: bool) -> None:
        self.key_input.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.show_key_btn.setText("隐藏" if checked else "显示")

    def _on_voice_preset_changed(self) -> None:
        preset = str(self.voice_preset_combo.currentData())
        custom = preset == "custom"
        self.voices_input.setEnabled(custom)
        if not custom:
            self.voices_input.setText(VOICE_PRESETS.get(preset, VOICE_PRESETS["auto"]))

    def _on_tls_mode_changed(self) -> None:
        custom = str(self.tls_mode_combo.currentData() or "auto") == "custom_ca"
        self.ca_bundle_input.setEnabled(custom)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.fetch_btn.setEnabled(not busy)

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.key_input.setText(str(cfg.get("qwen_api_key", "")))
        self.output_root_input.setText(str(cfg.get("output_root", self.output_root_input.text())))
        self.max_seconds.setValue(int(cfg.get("max_seconds", 0)))
        self.speaker_count.setValue(int(cfg.get("speaker_count", 2)))
        tts_model = str(cfg.get("tts_model", "qwen3-tts-flash"))
        idx = self.tts_combo.findData(tts_model)
        if idx >= 0:
            self.tts_combo.setCurrentIndex(idx)
        lang = str(cfg.get("output_language", "zh-CN"))
        lang_idx = self.output_lang_combo.findData(lang)
        if lang_idx >= 0:
            self.output_lang_combo.setCurrentIndex(lang_idx)
        self.voices_input.setText(str(cfg.get("voices", self.voices_input.text())))
        cookie_browser = str(cfg.get("cookies_from_browser", "none"))
        cookie_idx = self.cookies_combo.findData(cookie_browser)
        if cookie_idx >= 0:
            self.cookies_combo.setCurrentIndex(cookie_idx)
        self.cookies_file_input.setText(str(cfg.get("cookies_file", "")))
        tls_mode = str(cfg.get("tls_mode", "auto"))
        tls_idx = self.tls_mode_combo.findData(tls_mode)
        if tls_idx >= 0:
            self.tls_mode_combo.setCurrentIndex(tls_idx)
        self.ca_bundle_input.setText(str(cfg.get("ca_bundle_file", "")))
        preset = str(cfg.get("voice_preset", "auto"))
        pidx = self.voice_preset_combo.findData(preset)
        if pidx >= 0:
            self.voice_preset_combo.setCurrentIndex(pidx)
        else:
            self.voice_preset_combo.setCurrentIndex(0)
        self._on_voice_preset_changed()
        self._on_tls_mode_changed()
        self.keep_temp_checkbox.setChecked(bool(cfg.get("keep_temp", False)))

    def _save_config(self) -> None:
        APP_HOME.mkdir(parents=True, exist_ok=True)
        cfg = {
            "qwen_api_key": self.key_input.text().strip(),
            "output_root": self.output_root_input.text().strip(),
            "max_seconds": self.max_seconds.value(),
            "speaker_count": self.speaker_count.value(),
            "tts_model": self.tts_combo.currentData(),
            "output_language": self.output_lang_combo.currentData(),
            "cookies_from_browser": self.cookies_combo.currentData(),
            "cookies_file": self.cookies_file_input.text().strip(),
            "tls_mode": self.tls_mode_combo.currentData(),
            "ca_bundle_file": self.ca_bundle_input.text().strip(),
            "voices": self.voices_input.text().strip(),
            "voice_preset": self.voice_preset_combo.currentData(),
            "keep_temp": self.keep_temp_checkbox.isChecked(),
        }
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_thumbnail()

    def _refresh_thumbnail(self) -> None:
        if not self.thumb_pixmap or self.thumb_pixmap.isNull():
            return
        target_size = self.thumb.size() - QSize(8, 8)
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.thumb.setPixmap(self.thumb_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def select_local_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择本地视频/音频",
            str(Path.home()),
            "Media Files (*.mp4 *.mov *.mkv *.webm *.m4a *.mp3 *.wav);;All Files (*)",
        )
        if file_path:
            self.local_input.setText(file_path)

    def select_output_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_root_input.text().strip() or str(Path.home()))
        if folder:
            self.output_root_input.setText(folder)

    def _validate_cookie_file_input(self) -> bool:
        cookie_file = self.cookies_file_input.text().strip()
        if not cookie_file:
            return True
        p = Path(cookie_file).expanduser()
        if p.exists() and p.is_file():
            return True
        QMessageBox.warning(self, "提示", f"cookies.txt 路径无效，请重新选择：{cookie_file}")
        return False

    def _validate_ca_bundle_input(self) -> bool:
        tls_mode = str(self.tls_mode_combo.currentData() or "auto")
        if tls_mode != "custom_ca":
            return True
        path = self.ca_bundle_input.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "当前 TLS 模式是“自定义 CA 文件”，请先选择 PEM 证书文件")
            return False
        p = Path(path).expanduser()
        if not p.exists() or not p.is_file():
            QMessageBox.warning(self, "提示", f"CA 证书文件路径无效，请重新选择：{path}")
            return False
        return True

    def select_cookie_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 cookies.txt",
            str(Path.home()),
            "Cookie Files (*.txt);;All Files (*)",
        )
        if file_path:
            self.cookies_file_input.setText(file_path)

    def clear_cookie_file(self) -> None:
        self.cookies_file_input.clear()

    def select_ca_bundle_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CA 证书（PEM）",
            str(Path.home()),
            "Certificate Files (*.pem *.crt *.cer *.txt);;All Files (*)",
        )
        if file_path:
            self.ca_bundle_input.setText(file_path)

    def clear_ca_bundle_file(self) -> None:
        self.ca_bundle_input.clear()

    def show_cookie_help(self) -> None:
        QMessageBox.information(
            self,
            "如何获取 cookies.txt",
            (
                "当 YouTube 提示风控（机器人验证）时，可使用 cookies.txt 兜底。\n\n"
                "步骤：\n"
                "1. 在浏览器登录 YouTube，并手动打开目标视频，确认可正常播放。\n"
                "2. 安装可导出 cookies 的浏览器扩展（例如 Get cookies.txt LOCALLY）。\n"
                "3. 在 youtube.com 页面导出 cookies，格式选择 Netscape，保存为 cookies.txt。\n"
                "4. 回到本应用，点击“导入文件”选择该 cookies.txt，再重试预览/转换。\n\n"
                "注意：\n"
                "- cookies.txt 等同登录凭证，请勿分享给他人。\n"
                "- 若过期或账号变更，需要重新导出。\n"
                "- 若使用 Safari 登录态，可能需要先在系统设置里给本应用“完全磁盘访问”。"
            ),
        )

    def open_full_disk_access_settings(self) -> None:
        # macOS does not allow apps to grant this permission automatically;
        # we can only deep-link users to the corresponding settings page.
        urls = [
            "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
            "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles",
        ]
        for u in urls:
            subprocess.run(["open", u], check=False)

    def fetch_preview_clicked(self) -> None:
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请先输入 YouTube 地址")
            return
        if not self._validate_cookie_file_input():
            return
        try:
            self._append_log("正在获取视频预览...")
            info = fetch_youtube_preview(
                url,
                cookie_browser=str(self.cookies_combo.currentData() or "none"),
                cookie_file=self.cookies_file_input.text().strip(),
            )
            self.preview_title = info.title
            self.preview_url = info.url
            self.preview_detected_lang = info.detected_language
            self.title_label.setText(f"标题：{info.title}")
            self.meta_label.setText(
                f"作者：{info.uploader or '-'}    时长：{int(info.duration_seconds//60)}m {int(info.duration_seconds%60)}s    检测语言：{LANG_LABEL.get(info.detected_language, info.detected_language)}"
            )
            self.desc.setPlainText(info.description[:1800])
            self._load_thumbnail(info.thumbnail_url)
            self._append_log("预览获取完成")
        except Exception as e:
            QMessageBox.critical(self, "预览失败", str(e))

    def _load_thumbnail(self, url: str) -> None:
        if not url:
            self.thumb.setText("无封面")
            self.thumb_pixmap = None
            return
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            pix = QPixmap()
            pix.loadFromData(r.content)
            self.thumb_pixmap = pix
            self._refresh_thumbnail()
        except Exception:
            self.thumb_pixmap = None
            self.thumb.setText("封面加载失败")

    def start_clicked(self) -> None:
        url = self.url_input.text().strip()
        local_file = self.local_input.text().strip()
        if not url and not local_file:
            QMessageBox.warning(self, "提示", "请提供 YouTube 地址或本地文件")
            return
        if url and local_file:
            QMessageBox.warning(self, "提示", "请二选一：YouTube 地址 或 本地文件")
            return
        key = self.key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请输入 Qwen API Key")
            return
        if not self._validate_cookie_file_input():
            return
        if not self._validate_ca_bundle_input():
            return

        self._save_config()
        self._set_busy(True)
        self.progress.setValue(0)
        self.stage_label.setText("任务启动中...")
        self._append_log("开始执行转换任务")
        self._append_log(
            "网络策略: "
            f"TLS={self.tls_mode_combo.currentData()} | "
            f"cookies={self.cookies_combo.currentData()} | "
            f"cookies.txt={'已设置' if self.cookies_file_input.text().strip() else '未设置'}"
        )

        voices = self.voices_input.text().strip()
        preset = str(self.voice_preset_combo.currentData())
        if preset != "custom":
            voices = VOICE_PRESETS.get(preset, VOICE_PRESETS["auto"])

        cfg = ConversionConfig(
            qwen_api_key=key,
            source_url=url,
            source_file=local_file,
            output_root=self.output_root_input.text().strip(),
            output_language=self.output_lang_combo.currentData(),
            cookies_from_browser=str(self.cookies_combo.currentData() or "none"),
            cookies_file=self.cookies_file_input.text().strip(),
            tls_mode=str(self.tls_mode_combo.currentData() or "auto"),
            ca_bundle_file=self.ca_bundle_input.text().strip(),
            tts_model=self.tts_combo.currentData(),
            voices=voices,
            speaker_count=self.speaker_count.value(),
            max_seconds=float(self.max_seconds.value()),
            keep_temp=self.keep_temp_checkbox.isChecked(),
        )

        self.worker_thread = QThread(self)
        self.worker = ConversionWorker(cfg)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def stop_clicked(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.requestInterruption()
            self._append_log("停止请求已发出（当前版本会在本阶段结束后停止）")

    def on_progress(self, percent: int, stage: str, detail: str) -> None:
        self.progress.setValue(max(0, min(100, percent)))
        self.stage_label.setText(f"{stage} | {detail}")
        self._append_log(f"[{percent:>3}%] {stage}: {detail}")

    def on_finished(self, result_obj: object) -> None:
        result = result_obj  # type: ignore[assignment]
        assert isinstance(result, ConversionResult)
        self._set_busy(False)
        self.progress.setValue(100)
        self.stage_label.setText("完成")
        self._append_log(f"任务完成: {result.output_dir}")

        row = build_history_row(
            title=self.preview_title or Path(result.source_path).name,
            source_url=self.preview_url or "",
            output_language=self.output_lang_combo.currentData(),
            status="SUCCESS",
            output_dir=result.output_dir,
            audio_path=result.dubbed_audio_path,
            video_path=result.dubbed_video_path,
        )
        self.history_store.add(row)
        self._refresh_history()
        QMessageBox.information(
            self,
            "完成",
            (
                f"转换完成。\n\n输出目录: {result.output_dir}\n"
                f"音频: {result.dubbed_audio_path}\n"
                f"视频: {result.dubbed_video_path or '（源文件无视频流）'}"
            ),
        )

    def on_failed(self, err: str) -> None:
        self._set_busy(False)
        display_err = err
        err_lower = err.lower()
        if "Missing required binary" in err:
            display_err = (
                f"{err}\n\n当前应用未找到 ffmpeg/ffprobe。\n"
                "请使用最新打包版本（已内置 ffmpeg），或在系统安装 ffmpeg 后重试。"
            )
        is_cookie_permission_error = ("系统权限限制" in err) or (
            "operation not permitted" in err.lower() and "cookies" in err.lower()
        )
        if is_cookie_permission_error:
            display_err = (
                f"{err}\n\n"
                "检测到是 macOS 权限拦截（读取浏览器 cookies 被拒绝）。\n"
                "请在“系统设置 -> 隐私与安全性 -> 完全磁盘访问”中勾选 VideoDubStudio，"
                "然后彻底退出并重新打开应用。"
            )
        is_tls_cert_error = (
            "certificate_verify_failed" in err_lower
            or "certificate verify failed" in err_lower
            or "unable to get local issuer certificate" in err_lower
            or "https 证书校验失败" in err
        )
        if is_tls_cert_error:
            display_err = (
                f"{err}\n\n"
                "检测到本机 TLS 证书链与应用运行时不一致。\n"
                "建议按以下顺序重试：\n"
                "1) 将「TLS 证书模式」改为“自动（推荐）”；\n"
                "2) 若仍失败，改为“内置 certifi 证书链”；\n"
                "3) 企业网络下再尝试“自定义 CA 文件”，导入公司/代理提供的 PEM 证书；\n"
                "4) 若 curl 可用但应用仍失败，删除本机配置后重启应用再试：\n"
                f"   {CONFIG_PATH}\n"
                "5) 关闭代理或切换网络再试。"
            )
        if ("阿里云预检失败：api key" in err) or ("api key 无效" in err) or ("authenticationerror" in err_lower):
            display_err = (
                f"{err}\n\n"
                "请检查：\n"
                "1) API Key 是否填写完整（无多余空格）；\n"
                "2) 该 Key 是否已过期/被禁用；\n"
                "3) 尽量使用个人独立 Key，不要多人共享同一个 Key。"
            )
        if ("无调用权限（403）" in err) or ("无该模型调用权限" in err):
            display_err = (
                f"{err}\n\n"
                "当前账号没有目标模型权限。\n"
                "请在阿里云百炼控制台确认：\n"
                "- 翻译模型（如 qwen3-max）已开通\n"
                "- TTS 模型（如 qwen3-tts-flash）已开通\n"
                "- 所用 API Key 绑定的账号/项目具备调用权限"
            )
        if ("限流或额度不足" in err) or ("429" in err_lower) or ("quota" in err_lower):
            display_err = (
                f"{err}\n\n"
                "这是限流/额度问题。\n"
                "建议：\n"
                "1) 稍后重试；\n"
                "2) 更换独立 API Key；\n"
                "3) 降低并发任务或处理更短片段。"
            )
        if ("模型不可用" in err) or ("model is not available" in err_lower) or ("unsupported model" in err_lower):
            display_err = (
                f"{err}\n\n"
                "请更换可用模型，或在百炼控制台确认该模型对当前账号已开放。"
            )
        if ("代理需要认证（http 407）" in err_lower) or ("http 407" in err_lower):
            display_err = (
                f"{err}\n\n"
                "当前网络有代理认证要求。请先完成代理登录，或切换到无需代理认证的网络后重试。"
            )
        self._append_log(f"任务失败: {err}")

        row = build_history_row(
            title=self.preview_title or "unknown",
            source_url=self.preview_url or self.url_input.text().strip(),
            output_language=self.output_lang_combo.currentData(),
            status="FAILED",
            output_dir=self.output_root_input.text().strip(),
            error=err,
        )
        self.history_store.add(row)
        self._refresh_history()
        if is_cookie_permission_error:
            ret = QMessageBox.question(
                self,
                "任务失败",
                f"{display_err}\n\n是否现在打开系统设置？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Yes:
                self.open_full_disk_access_settings()
            return
        QMessageBox.critical(self, "任务失败", display_err)

    def _refresh_history(self) -> None:
        rows = self.history_store.load()
        self.history_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(row.get("created_at", ""))))
            self.history_table.setItem(i, 1, QTableWidgetItem(str(row.get("title", ""))))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(row.get("status", ""))))
            self.history_table.setItem(
                i,
                3,
                QTableWidgetItem(LANG_LABEL.get(str(row.get("output_language", "")), str(row.get("output_language", "")))),
            )
            self.history_table.setItem(i, 4, QTableWidgetItem("打开" if row.get("audio_path") else "-"))
            self.history_table.setItem(i, 5, QTableWidgetItem("打开" if row.get("video_path") else "-"))
            for c in range(6):
                item = self.history_table.item(i, c)
                if item:
                    item.setData(Qt.UserRole, row)

    def _selected_row_data(self) -> Optional[dict]:
        items = self.history_table.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.UserRole)

    def open_history_path(self, key: str) -> None:
        row = self._selected_row_data()
        if not row:
            QMessageBox.information(self, "提示", "请先在历史列表中选择一行")
            return
        path = str(row.get(key, "")).strip()
        if not path:
            QMessageBox.information(self, "提示", "该项没有可打开的路径")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "提示", f"路径不存在: {path}")
            return
        subprocess.run(["open", path], check=False)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QMainWindow {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 rgba(235,244,255,255),
                stop:0.45 rgba(232,247,244,255),
                stop:1 rgba(255,242,235,255));
        }
        QWidget#AppSurface {
            background: transparent;
        }
        QLabel#HeroTitle {
            font-size: 26px;
            font-weight: 700;
            color: #193252;
            letter-spacing: 0.4px;
        }
        QLabel#HeroSubtitle {
            font-size: 13px;
            color: #3d546f;
            margin-bottom: 4px;
        }
        QGroupBox {
            font-weight: 700;
            border: 1px solid rgba(255,255,255,175);
            border-radius: 18px;
            margin-top: 12px;
            padding: 14px 14px 12px 14px;
            background: rgba(255,255,255,150);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            color: #173a5c;
            font-size: 14px;
            background: rgba(255,255,255,140);
            border-radius: 8px;
        }
        QLabel {
            color: #1f3148;
            font-size: 13px;
        }
        QLabel#HintText {
            color: #4b6381;
            font-size: 12px;
        }
        QLabel#Thumb {
            border: 1px solid rgba(127, 164, 205, 110);
            border-radius: 12px;
            background: rgba(15,33,57,160);
            color: #d4e4f8;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QTableWidget {
            border: 1px solid rgba(132, 162, 201, 120);
            border-radius: 10px;
            padding: 6px 8px;
            background: rgba(255,255,255,185);
            color: #1b2f4d;
            selection-background-color: rgba(43,113,208,140);
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {
            border: 1px solid rgba(45, 109, 201, 180);
            background: rgba(255,255,255,220);
        }
        QToolButton, QPushButton {
            min-height: 34px;
            padding: 0 14px;
            border-radius: 10px;
            border: 1px solid rgba(93, 132, 173, 120);
            background: rgba(255,255,255,160);
            color: #17395a;
        }
        QPushButton:hover, QToolButton:hover {
            background: rgba(255,255,255,210);
            border: 1px solid rgba(70, 120, 177, 160);
        }
        QPushButton#PrimaryButton {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(52,125,236,220), stop:1 rgba(45,181,210,220));
            color: white;
            border: none;
            font-weight: 700;
        }
        QPushButton#PrimaryButton:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(41,113,223,235), stop:1 rgba(37,167,194,235));
        }
        QProgressBar {
            border: 1px solid rgba(132, 162, 201, 130);
            border-radius: 10px;
            text-align: center;
            min-height: 24px;
            background: rgba(255,255,255,170);
            color: #1f3148;
        }
        QProgressBar::chunk {
            border-radius: 10px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(52,125,236,230), stop:1 rgba(45,181,210,230));
        }
        QHeaderView::section {
            background: rgba(236, 245, 255, 230);
            border: none;
            border-bottom: 1px solid rgba(132, 162, 201, 100);
            padding: 7px;
            color: #2a4565;
            font-weight: 600;
        }
        """
    )

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
