import os


def test_gui_defaults_to_manual_mode_with_model_path() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from film_mask_automation.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.conversion_mode.currentText() == "规则手动"
        assert window._conversion_mode() == "manual"
        assert window.ai_model_path.text()

        window.conversion_mode.setCurrentText("智能自动")
        assert window._conversion_mode() == "smart"

        window.conversion_mode.setCurrentText("AI模型")
        assert window._conversion_mode() == "ai"
    finally:
        window.close()
