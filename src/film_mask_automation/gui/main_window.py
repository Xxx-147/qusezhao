from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from film_mask_automation.cli import SUPPORTED_EXTENSIONS
from film_mask_automation.processor import ConversionParams, convert_file, convert_image
from film_mask_automation.profile import apply_color_profile, fit_color_profile, load_color_profile, save_color_profile
from film_mask_automation.smart import convert_image_smart


@dataclass(frozen=True)
class QueuedImage:
    input_path: Path
    output_path: Path | None = None
    status: str = "等待"


class BatchWorker(QObject):
    progress = Signal(int, int, str)
    file_done = Signal(int, str, str)
    log = Signal(str)
    finished = Signal()

    def __init__(
        self,
        images: list[Path],
        output_dir: Path,
        params: ConversionParams,
        profile_path: Path | None,
        processing_mode: str,
        ai_model_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._images = images
        self._output_dir = output_dir
        self._params = params
        self._profile_path = profile_path
        self._processing_mode = processing_mode
        self._ai_model_path = ai_model_path
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        profile = load_color_profile(self._profile_path) if self._profile_path else None
        model_converter = None
        if self._ai_model_path:
            from film_mask_automation.ml.inference import convert_with_model

            model_converter = convert_with_model
        if self._processing_mode == "ai" and not self._ai_model_path:
            self.log.emit("AI 模式需要先选择模型 checkpoint。")
            self.finished.emit()
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        total = len(self._images)
        for index, input_path in enumerate(self._images, start=1):
            if self._cancelled:
                self.log.emit("批处理已取消。")
                break
            output_path = self._output_dir / f"{input_path.stem}_positive{input_path.suffix}"
            try:
                self.progress.emit(index, total, input_path.name)
                if self._processing_mode == "ai" and model_converter and self._ai_model_path:
                    with Image.open(input_path) as image:
                        model_converter(image, self._ai_model_path).save(output_path)
                    diagnostics = {"ai_model": str(self._ai_model_path), "ai_enhance": True, "ai_hybrid_anchor": True}
                elif self._processing_mode == "smart":
                    with Image.open(input_path) as image:
                        result = convert_image_smart(image)
                        result.image.save(output_path)
                        diagnostics = result.diagnostics
                else:
                    diagnostics = convert_file(input_path, output_path, self._params)
                if profile and self._processing_mode != "ai":
                    with Image.open(output_path) as image:
                        apply_color_profile(image, profile).save(output_path)
                self.file_done.emit(index - 1, str(output_path), "完成")
                if "mask_rgb" in diagnostics:
                    self.log.emit(f"完成：{input_path.name}  色罩={diagnostics['mask_rgb']}")
                elif "ai_model" in diagnostics:
                    self.log.emit(
                        f"完成：{input_path.name}  AI模型={diagnostics['ai_model']}  "
                        f"增强={diagnostics.get('ai_enhance', True)}  "
                        f"混合锚点={diagnostics.get('ai_hybrid_anchor', True)}"
                    )
                else:
                    self.log.emit(f"完成：{input_path.name}")
            except Exception as exc:
                self.file_done.emit(index - 1, "", "失败")
                self.log.emit(f"失败：{input_path.name}  {exc}")
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Film Mask Automation - 胶片自动去色罩")
        self.resize(1480, 920)

        self._items: list[QueuedImage] = []
        self._current_original: Image.Image | None = None
        self._current_output: Image.Image | None = None
        self._worker_thread: QThread | None = None
        self._worker: BatchWorker | None = None

        self._build_actions()
        self._build_ui()
        self._apply_style()
        self._log("项目已启动。添加负片或文件夹后，可以预览、校准 profile 或批量输出。")

    def _build_actions(self) -> None:
        self.open_files_action = QAction("添加图片", self)
        self.open_files_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.open_files_action.triggered.connect(self._add_files)
        self.open_folder_action = QAction("添加文件夹", self)
        self.open_folder_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_folder_action.triggered.connect(self._add_folder)
        self.process_all_action = QAction("批量处理", self)
        self.process_all_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.process_all_action.triggered.connect(self._process_all)
        self.preview_action = QAction("预览当前", self)
        self.preview_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.preview_action.triggered.connect(self._preview_current)
        self.calibrate_action = QAction("校准 Profile", self)
        self.calibrate_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.calibrate_action.triggered.connect(self._calibrate_profile)
        self.clear_action = QAction("清空队列", self)
        self.clear_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.clear_action.triggered.connect(self._clear_queue)

    def _build_ui(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        for action in [
            self.open_files_action,
            self.open_folder_action,
            self.preview_action,
            self.calibrate_action,
            self.process_all_action,
            self.clear_action,
        ]:
            toolbar.addAction(action)
        self.addToolBar(toolbar)

        root = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_center_panel())
        root.addWidget(self._build_right_panel())
        root.setSizes([330, 790, 360])
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("处理队列")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.queue_table = QTableWidget(0, 3)
        self.queue_table.setHorizontalHeaderLabels(["文件", "状态", "输出"])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.queue_table.itemSelectionChanged.connect(self._preview_current)
        layout.addWidget(self.queue_table, 1)

        buttons = QGridLayout()
        add_files = QPushButton("添加图片")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("添加文件夹")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("移除选中")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear_queue)
        buttons.addWidget(add_files, 0, 0)
        buttons.addWidget(add_folder, 0, 1)
        buttons.addWidget(remove, 1, 0)
        buttons.addWidget(clear, 1, 1)
        layout.addLayout(buttons)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        header = QHBoxLayout()
        title = QLabel("预览")
        title.setObjectName("PanelTitle")
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(["左右对比", "只看输出", "只看原图"])
        self.preview_mode.currentTextChanged.connect(self._refresh_preview_visibility)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.preview_mode)
        layout.addLayout(header)

        preview_split = QSplitter(Qt.Orientation.Horizontal)
        self.original_label = self._image_label("原图")
        self.output_label = self._image_label("输出预览")
        preview_split.addWidget(self.original_label)
        preview_split.addWidget(self.output_label)
        preview_split.setSizes([1, 1])
        layout.addWidget(preview_split, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(150)
        layout.addWidget(self.log_view)
        return panel

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_adjust_panel(), "调整")
        tabs.addTab(self._build_output_panel(), "自动化")
        tabs.addTab(self._build_profile_panel(), "Profile")
        return tabs

    def _build_adjust_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        mask_group = QGroupBox("色罩估计")
        mask_form = QFormLayout(mask_group)
        self.mask_source = QComboBox()
        self.mask_source.addItems(["auto", "border", "percentile", "manual"])
        self.mask_rgb = QLineEdit("230,145,75")
        mask_form.addRow("模式", self.mask_source)
        mask_form.addRow("手动 RGB", self.mask_rgb)
        layout.addWidget(mask_group)

        tone_group = QGroupBox("影调")
        tone_form = QFormLayout(tone_group)
        self.border_fraction = self._slider(1, 25, 6)
        self.black_percentile = self._slider(0, 100, 5)
        self.white_percentile = self._slider(900, 1000, 995)
        self.reference_exponent = self._slider(50, 220, 100)
        self.red_ratio = self._slider(50, 160, 100)
        self.blue_ratio = self._slider(50, 160, 100)
        self.exposure = self._slider(-200, 200, 0)
        self.brightness = self._slider(-100, 100, 0)
        self.gamma = self._slider(50, 180, 104)
        self.contrast = self._slider(50, 180, 112)
        self.saturation = self._slider(0, 200, 100)
        self.sharpen = self._slider(0, 200, 0)
        tone_form.addRow("边框比例", self.border_fraction)
        tone_form.addRow("黑点百分位", self.black_percentile)
        tone_form.addRow("白点百分位", self.white_percentile)
        tone_form.addRow("参考指数", self.reference_exponent)
        tone_form.addRow("红通道比例", self.red_ratio)
        tone_form.addRow("蓝通道比例", self.blue_ratio)
        tone_form.addRow("曝光 EV", self.exposure)
        tone_form.addRow("亮度", self.brightness)
        tone_form.addRow("伽马", self.gamma)
        tone_form.addRow("对比度", self.contrast)
        tone_form.addRow("饱和度", self.saturation)
        tone_form.addRow("锐化", self.sharpen)
        layout.addWidget(tone_group)

        color_group = QGroupBox("色彩")
        color_form = QFormLayout(color_group)
        self.white_balance = QComboBox()
        self.white_balance.addItems(["grayworld", "none"])
        self.temperature = self._slider(-100, 100, 0)
        self.tint = self._slider(-100, 100, 0)
        self.red_gain = self._slider(50, 180, 100)
        self.green_gain = self._slider(50, 180, 100)
        self.blue_gain = self._slider(50, 180, 100)
        self.auto_preview = QCheckBox("参数变化后自动预览")
        self.auto_preview.setChecked(False)
        color_form.addRow("白平衡", self.white_balance)
        color_form.addRow("色温", self.temperature)
        color_form.addRow("色调", self.tint)
        color_form.addRow("红通道", self.red_gain)
        color_form.addRow("绿通道", self.green_gain)
        color_form.addRow("蓝通道", self.blue_gain)
        color_form.addRow("", self.auto_preview)
        layout.addWidget(color_group)

        for widget in [
            self.mask_source,
            self.white_balance,
            self.border_fraction,
            self.black_percentile,
            self.white_percentile,
            self.reference_exponent,
            self.red_ratio,
            self.blue_ratio,
            self.exposure,
            self.brightness,
            self.gamma,
            self.contrast,
            self.saturation,
            self.temperature,
            self.tint,
            self.red_gain,
            self.green_gain,
            self.blue_gain,
            self.sharpen,
        ]:
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._preview_if_auto)
            else:
                widget.valueChanged.connect(self._preview_if_auto)

        preview = QPushButton("生成当前预览")
        preview.clicked.connect(self._preview_current)
        layout.addWidget(preview)
        layout.addStretch(1)
        return panel

    def _build_output_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        group = QGroupBox("批量输出")
        form = QFormLayout(group)
        self.output_dir = QLineEdit(str(Path.home() / "Desktop" / "film-mask-output"))
        self.conversion_mode = QComboBox()
        self.conversion_mode.addItems(["规则手动", "智能自动", "AI模型"])
        self.conversion_mode.setCurrentText("规则手动")
        self.conversion_mode.currentTextChanged.connect(self._preview_current)
        choose_output = QPushButton("选择输出目录")
        choose_output.clicked.connect(self._choose_output_dir)
        form.addRow("输出目录", self.output_dir)
        form.addRow("处理模式", self.conversion_mode)
        form.addRow("", choose_output)
        layout.addWidget(group)

        batch = QPushButton("开始批量自动去色罩")
        batch.setObjectName("PrimaryButton")
        batch.clicked.connect(self._process_all)
        cancel = QPushButton("取消批处理")
        cancel.clicked.connect(self._cancel_batch)
        layout.addWidget(batch)
        layout.addWidget(cancel)
        layout.addStretch(1)
        return panel

    def _build_profile_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        group = QGroupBox("复用色彩 Profile")
        form = QFormLayout(group)
        self.profile_path = QLineEdit("")
        choose_profile = QPushButton("选择 Profile")
        choose_profile.clicked.connect(self._choose_profile)
        clear_profile = QPushButton("不使用 Profile")
        clear_profile.clicked.connect(lambda: self.profile_path.setText(""))
        form.addRow("Profile JSON", self.profile_path)
        form.addRow("", choose_profile)
        form.addRow("", clear_profile)
        layout.addWidget(group)

        ai_group = QGroupBox("AI 模型")
        ai_form = QFormLayout(ai_group)
        self.ai_model_path = QLineEdit(str(self._default_ai_model_path() or ""))
        choose_ai_model = QPushButton("选择模型 Checkpoint")
        choose_ai_model.clicked.connect(self._choose_ai_model)
        clear_ai_model = QPushButton("不使用 AI 模型")
        clear_ai_model.clicked.connect(lambda: self.ai_model_path.setText(""))
        ai_form.addRow("模型文件", self.ai_model_path)
        ai_form.addRow("", choose_ai_model)
        ai_form.addRow("", clear_ai_model)
        layout.addWidget(ai_group)

        calibrate = QGroupBox("用参考样图校准")
        calibrate_layout = QVBoxLayout(calibrate)
        tip = QLabel("选择队列中的负片，再选择对应的目标去色罩样图，程序会生成可复用 profile。")
        tip.setWordWrap(True)
        calibrate_layout.addWidget(tip)
        make_profile = QPushButton("从当前负片生成 Profile")
        make_profile.clicked.connect(self._calibrate_profile)
        calibrate_layout.addWidget(make_profile)
        layout.addWidget(calibrate)
        layout.addStretch(1)
        return panel

    def _image_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(300, 420)
        label.setFrameShape(QFrame.Shape.StyledPanel)
        label.setObjectName("ImagePane")
        return label

    def _slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        return slider

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择负片图片",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)",
        )
        self._add_paths(Path(path) for path in paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择负片文件夹", str(Path.home()))
        if not folder:
            return
        paths = [path for path in Path(folder).iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        self._add_paths(paths)

    def _add_paths(self, paths: Iterable[Path]) -> None:
        existing = {item.input_path for item in self._items}
        added = 0
        for path in paths:
            if path in existing or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            self._items.append(QueuedImage(input_path=path))
            row = self.queue_table.rowCount()
            self.queue_table.insertRow(row)
            self.queue_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.queue_table.setItem(row, 1, QTableWidgetItem("等待"))
            self.queue_table.setItem(row, 2, QTableWidgetItem(""))
            added += 1
        self._log(f"已添加 {added} 张图片。")
        if self.queue_table.rowCount() and not self.queue_table.selectedItems():
            self.queue_table.selectRow(0)

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.queue_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.queue_table.removeRow(row)
            del self._items[row]
        self._log(f"已移除 {len(rows)} 张图片。")

    def _clear_queue(self) -> None:
        self._items.clear()
        self.queue_table.setRowCount(0)
        self.original_label.setText("原图")
        self.output_label.setText("输出预览")
        self._log("队列已清空。")

    def _selected_row(self) -> int | None:
        rows = sorted({index.row() for index in self.queue_table.selectedIndexes()})
        return rows[0] if rows else None

    def _preview_current(self) -> None:
        row = self._selected_row()
        if row is None or row >= len(self._items):
            return
        input_path = self._items[row].input_path
        try:
            with Image.open(input_path) as image:
                original = image.convert("RGB")
                mode = self._conversion_mode()
                ai_model_path = self._ai_model_path()
                if mode == "ai":
                    if not ai_model_path:
                        raise ValueError("AI 模式需要先选择模型 checkpoint")
                    from film_mask_automation.ml.inference import convert_with_model

                    output = convert_with_model(original, ai_model_path)
                    result = None
                elif mode == "smart":
                    result = convert_image_smart(original)
                    output = result.image
                else:
                    result = convert_image(original, self._params())
                    output = result.image
                    profile_path = self._profile_path()
                    if profile_path:
                        output = apply_color_profile(output, load_color_profile(profile_path))
                self._current_original = original.copy()
                self._current_output = output.copy()
                self._set_pixmap(self.original_label, original)
                self._set_pixmap(self.output_label, output)
                if result:
                    self._log(f"预览：{input_path.name}  色罩={result.diagnostics['mask_rgb']}")
                else:
                    self._log(f"AI 预览：{input_path.name}")
        except Exception as exc:
            self._log(f"预览失败：{exc}")

    def _preview_if_auto(self) -> None:
        if self.auto_preview.isChecked():
            self._preview_current()

    def _refresh_preview_visibility(self) -> None:
        mode = self.preview_mode.currentText()
        self.original_label.setVisible(mode != "只看输出")
        self.output_label.setVisible(mode != "只看原图")

    def _process_all(self) -> None:
        if not self._items:
            QMessageBox.information(self, "没有图片", "请先添加负片图片或文件夹。")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            QMessageBox.information(self, "正在处理", "批处理正在运行。")
            return

        output_dir = Path(self.output_dir.text()).expanduser()
        images = [item.input_path for item in self._items]
        self._worker_thread = QThread()
        self._worker = BatchWorker(
            images,
            output_dir,
            self._params(),
            self._profile_path(),
            self._conversion_mode(),
            self._ai_model_path(),
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()
        self._log(f"开始批量处理 {len(images)} 张图片。")

    def _cancel_batch(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, index: int, total: int, name: str) -> None:
        self.progress.setValue(int(index / max(total, 1) * 100))
        self.statusBar().showMessage(f"处理中 {index}/{total}: {name}")

    def _on_file_done(self, row: int, output: str, status: str) -> None:
        if row < self.queue_table.rowCount():
            self.queue_table.setItem(row, 1, QTableWidgetItem(status))
            self.queue_table.setItem(row, 2, QTableWidgetItem(output))

    def _on_batch_finished(self) -> None:
        self.progress.setValue(100)
        self.statusBar().showMessage("批处理完成")
        self._log("批处理完成。")
        self._worker = None
        self._worker_thread = None

    def _choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text())
        if folder:
            self.output_dir.setText(folder)

    def _choose_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Profile", str(Path.cwd()), "JSON (*.json)")
        if path:
            self.profile_path.setText(path)
            self._log(f"已选择 Profile：{path}")

    def _choose_ai_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 AI 模型", str(Path.cwd()), "Model (*.pt *.pth *.ckpt)")
        if path:
            self.ai_model_path.setText(path)
            self.conversion_mode.setCurrentText("AI模型")
            self._log(f"已选择 AI 模型：{path}")
            self._preview_current()

    def _calibrate_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "请选择负片", "请先在队列中选择一张负片。")
            return
        reference, _ = QFileDialog.getOpenFileName(
            self,
            "选择对应的去色罩目标样图",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)",
        )
        if not reference:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 Profile",
            str(Path.cwd() / "profiles" / "custom_profile.json"),
            "JSON (*.json)",
        )
        if not save_path:
            return
        negative_path = self._items[row].input_path
        try:
            with Image.open(negative_path) as negative, Image.open(reference) as reference_image:
                converted = convert_image(negative, self._params()).image
                profile = fit_color_profile(converted, reference_image)
                save_color_profile(profile, Path(save_path))
                preview = apply_color_profile(converted, profile)
                preview_path = Path(save_path).with_suffix(".preview.jpg")
                preview.save(preview_path)
                self.profile_path.setText(save_path)
                self._set_pixmap(self.output_label, preview)
                self._log(f"Profile 已保存：{save_path}")
                self._log(f"校准预览已保存：{preview_path}")
        except Exception as exc:
            self._log(f"校准失败：{exc}")

    def _params(self) -> ConversionParams:
        manual_mask = None
        if self.mask_source.currentText() == "manual":
            parts = [float(part.strip()) for part in self.mask_rgb.text().split(",")]
            if len(parts) != 3:
                raise ValueError("手动 RGB 必须是 R,G,B 三个数字")
            manual_mask = (parts[0], parts[1], parts[2])
        return ConversionParams(
            mask_source=self.mask_source.currentText(),
            manual_mask_rgb=manual_mask,
            border_fraction=self.border_fraction.value() / 100.0,
            black_percentile=self.black_percentile.value() / 10.0,
            white_percentile=self.white_percentile.value() / 10.0,
            reference_exponent=self.reference_exponent.value() / 100.0,
            red_ratio=self.red_ratio.value() / 100.0,
            blue_ratio=self.blue_ratio.value() / 100.0,
            exposure=self.exposure.value() / 100.0,
            brightness=self.brightness.value() / 255.0,
            gamma=self.gamma.value() / 100.0,
            contrast=self.contrast.value() / 100.0,
            saturation=self.saturation.value() / 100.0,
            temperature=self.temperature.value() / 100.0,
            tint=self.tint.value() / 100.0,
            red_gain=self.red_gain.value() / 100.0,
            green_gain=self.green_gain.value() / 100.0,
            blue_gain=self.blue_gain.value() / 100.0,
            white_balance=self.white_balance.currentText(),
            sharpen=self.sharpen.value() / 100.0,
        )

    def _profile_path(self) -> Path | None:
        text = self.profile_path.text().strip()
        return Path(text) if text else None

    def _conversion_mode(self) -> str:
        text = self.conversion_mode.currentText()
        if text == "AI模型":
            return "ai"
        if text == "智能自动":
            return "smart"
        return "manual"

    def _default_ai_model_path(self) -> Path | None:
        candidates = [
            Path.cwd() / "release_assets" / "models" / "film_mask_tiny_mixed_true_negative.pt",
            Path.cwd() / "models" / "film_mask_tiny_mixed_true_negative.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _ai_model_path(self) -> Path | None:
        text = self.ai_model_path.text().strip()
        return Path(text) if text else None

    def _set_pixmap(self, label: QLabel, image: Image.Image) -> None:
        pixmap = _pil_to_pixmap(image)
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _log(self, message: str) -> None:
        self.log_view.append(message)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._current_original is not None:
            self._set_pixmap(self.original_label, self._current_original)
        if self._current_output is not None:
            self._set_pixmap(self.output_label, self._current_output)

    def _apply_style(self) -> None:
        QApplication.instance().setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #202124;
                color: #e8eaed;
                font-size: 13px;
            }
            QToolBar {
                background: #2b2d31;
                border: 0;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton, QPushButton {
                background: #34373d;
                border: 1px solid #4b4f58;
                border-radius: 6px;
                padding: 7px 10px;
                color: #f1f3f4;
            }
            QToolButton:hover, QPushButton:hover {
                background: #41454d;
            }
            QPushButton#PrimaryButton {
                background: #1f6feb;
                border-color: #388bfd;
                font-weight: 600;
            }
            QLabel#PanelTitle {
                font-size: 16px;
                font-weight: 700;
                padding: 4px 0;
            }
            QLabel#ImagePane {
                background: #111315;
                border: 1px solid #34373d;
                border-radius: 8px;
                color: #9aa0a6;
            }
            QTableWidget, QTextEdit, QLineEdit, QComboBox {
                background: #17191c;
                border: 1px solid #34373d;
                border-radius: 6px;
                selection-background-color: #294f8f;
                color: #e8eaed;
            }
            QHeaderView::section {
                background: #2b2d31;
                color: #e8eaed;
                border: 0;
                padding: 6px;
            }
            QGroupBox {
                border: 1px solid #3b3f46;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #34373d;
            }
            QTabBar::tab {
                background: #2b2d31;
                padding: 9px 14px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #3b3f46;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #4b4f58;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #d2d7de;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            """
        )


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap
