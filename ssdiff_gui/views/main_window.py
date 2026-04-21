"""Main application window for SSD."""

from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QStatusBar,
    QFrame,
    QApplication,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt, Signal, QTimer

from ssdiff_gui import __version__, __ssdiff_version__
from ..models.project import Project
from ..utils.file_io import ProjectIO
from ..utils.settings import app_settings


class MainWindow(QMainWindow):
    """Main application window with stage navigation."""

    project_changed = Signal(Project)

    def __init__(self):
        super().__init__()
        self.project = None
        self.current_stage = 0

        self._settings = app_settings()

        self._init_ui()
        self._create_menus()
        self._create_status_bar()
        QTimer.singleShot(1000, self._start_update_check)

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("SSD - Supervised Semantic Differential")
        self.setMinimumSize(1350, 900)

        # Restore saved window geometry, or use sensible defaults
        restored = False
        saved_geometry = self._settings.value("window/geometry")
        if saved_geometry is not None:
            self.restoreGeometry(saved_geometry)
            restored = self._validate_window_geometry()

        if not restored:
            self.setGeometry(100, 100, 1350, 900)

        if self._settings.value("window/maximized", False, type=bool):
            self.showMaximized()

        # Central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Stage navigation bar
        self._create_stage_nav_bar()
        main_layout.addWidget(self.stage_nav_bar)
        self.stage_nav_bar.hide()  # Hidden until a project is loaded

        # Stage stack
        self.stage_stack = QStackedWidget()
        main_layout.addWidget(self.stage_stack)

        # Create pages
        self._create_welcome_page()
        self._create_stage_pages()

    def _create_stage_nav_bar(self):
        """Create the visual stage navigation bar with numbered steps."""
        self.stage_nav_bar = QFrame()
        self.stage_nav_bar.setObjectName("stage_nav_bar")
        self.stage_nav_bar.setFixedHeight(50)

        nav_layout = QHBoxLayout(self.stage_nav_bar)
        nav_layout.setContentsMargins(16, 0, 16, 0)
        nav_layout.setSpacing(0)

        # Home button
        home_btn = QPushButton("Home")
        home_btn.setObjectName("btn_ghost")
        home_btn.setFixedHeight(36)
        home_btn.setCursor(Qt.PointingHandCursor)
        home_btn.setToolTip("Close project and return to main menu")
        home_btn.clicked.connect(lambda: self._go_home())
        nav_layout.addWidget(home_btn)

        nav_layout.addSpacing(20)

        # Stage steps
        self._stage_btns = []
        self._stage_numbers = []
        stage_labels = ["Setup", "Run", "Results"]

        for i, label in enumerate(stage_labels):
            stage_num = i + 1

            # Number circle
            num_label = QLabel(str(stage_num))
            num_label.setObjectName("stage_step_number")
            num_label.setAlignment(Qt.AlignCenter)
            num_label.setFixedSize(22, 22)
            self._stage_numbers.append(num_label)

            # Step button
            step_btn = QPushButton(f"  {label}")
            step_btn.setObjectName("stage_step")
            step_btn.setCursor(Qt.PointingHandCursor)
            step_btn.setEnabled(False)
            step_btn.clicked.connect(lambda checked, s=stage_num: self.go_to_stage(s))
            self._stage_btns.append(step_btn)

            # Combine number + button in a mini layout
            step_layout = QHBoxLayout()
            step_layout.setSpacing(4)
            step_layout.setContentsMargins(0, 0, 0, 0)
            step_layout.addWidget(num_label)
            step_layout.addWidget(step_btn)

            nav_layout.addLayout(step_layout)

            # Separator between steps (but not after last)
            if i < len(stage_labels) - 1:
                sep = QLabel("\u203A")  # single right-pointing angle quotation
                sep.setObjectName("stage_step_separator")
                sep.setAlignment(Qt.AlignCenter)
                sep.setFixedWidth(30)
                nav_layout.addWidget(sep)

        nav_layout.addStretch()

    def _update_stage_nav_bar(self):
        """Update the visual state of the stage navigation bar."""
        for i, btn in enumerate(self._stage_btns):
            stage_num = i + 1
            is_current = (stage_num == self.current_stage)
            btn.isEnabled()

            if is_current:
                btn.setObjectName("stage_step_active")
                self._stage_numbers[i].setObjectName("stage_step_number_active")
            else:
                btn.setObjectName("stage_step")
                self._stage_numbers[i].setObjectName("stage_step_number")

            # Force style refresh
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            self._stage_numbers[i].style().unpolish(self._stage_numbers[i])
            self._stage_numbers[i].style().polish(self._stage_numbers[i])

    def _go_home(self):
        """Navigate back to the welcome page, closing the current project."""
        if self.project:
            # Warn about unsaved analysis run
            if self.stage3_widget.has_unsaved_result():
                reply = QMessageBox.warning(
                    self,
                    "Unsaved Result",
                    "You have an unsaved result.\n"
                    "If you return to the main menu, it will be discarded.\n\n"
                    "Go back to save it?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.go_to_stage(3)
                    return

            if self.project._dirty:
                reply = QMessageBox.question(
                    self,
                    "Return to Main Menu?",
                    "Do you want to save the project before returning to the main menu?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                )
                if reply == QMessageBox.Save:
                    self.save_project()
                    self._close_project()
                elif reply == QMessageBox.Discard:
                    self._close_project()
                else:
                    return
            else:
                self._close_project()

        self.stage_stack.setCurrentWidget(self.welcome_page)
        self.current_stage = 0
        self.stage_nav_bar.hide()

    def _close_project(self):
        """Close the current project and reset UI state."""
        self.project = None

        # Disable and hide menu actions
        self.save_action.setEnabled(False)
        self.stage1_action.setEnabled(False)
        self.stage2_action.setEnabled(False)
        self.stage3_action.setEnabled(False)
        self.view_menu.menuAction().setVisible(False)

        # Reset stage widgets
        self.stage1_widget.reset()
        self.stage2_widget.reset()
        self.stage3_widget.reset()

        # Reset nav bar buttons
        for btn in self._stage_btns:
            btn.setEnabled(False)

        self._update_title()
        self.status_bar.showMessage("Welcome to SSD")

    def _create_menus(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        self.import_result_action = QAction("&Import Results...", self)
        self.import_result_action.triggered.connect(self.import_result)
        self.import_result_action.setEnabled(False)
        file_menu.addAction(self.import_result_action)

        file_menu.addSeparator()

        self.save_action = QAction("&Save Project", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.triggered.connect(self.save_project)
        self.save_action.setEnabled(False)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.setMenuRole(QAction.PreferencesRole)
        settings_action.triggered.connect(self._open_settings_dialog)
        file_menu.addAction(settings_action)

        appearance_action = QAction("&Appearance...", self)
        appearance_action.triggered.connect(self._open_appearance_dialog)
        file_menu.addAction(appearance_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setMenuRole(QAction.QuitRole)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu (hidden until a project is loaded)
        self.view_menu = menubar.addMenu("&View")

        self.stage1_action = QAction("Stage 1: &Setup", self)
        self.stage1_action.triggered.connect(lambda: self.go_to_stage(1))
        self.stage1_action.setEnabled(False)
        self.view_menu.addAction(self.stage1_action)

        self.stage2_action = QAction("Stage 2: &Run", self)
        self.stage2_action.triggered.connect(lambda: self.go_to_stage(2))
        self.stage2_action.setEnabled(False)
        self.view_menu.addAction(self.stage2_action)

        self.stage3_action = QAction("Stage 3: &Results", self)
        self.stage3_action.triggered.connect(lambda: self.go_to_stage(3))
        self.stage3_action.setEnabled(False)
        self.view_menu.addAction(self.stage3_action)

        self.view_menu.menuAction().setVisible(False)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        tutorial_action = QAction("&Tutorial", self)
        tutorial_action.triggered.connect(self._open_tutorial)
        help_menu.addAction(tutorial_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.setMenuRole(QAction.AboutRole)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to SSD")

    def _create_welcome_page(self):
        """Create the welcome/landing page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(0)

        layout.addStretch(2)

        # Logo
        self._welcome_logo = QLabel()
        self._welcome_logo.setAlignment(Qt.AlignCenter)
        self._welcome_logo.setFixedSize(355, 355)
        self._welcome_logo.setStyleSheet("background: transparent;")
        layout.addWidget(self._welcome_logo, alignment=Qt.AlignCenter)
        self._refresh_welcome_logo()

        layout.addSpacing(16)

        # Title
        title = QLabel("SSD")
        title.setObjectName("label_welcome_title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(8)

        # Subtitle
        subtitle = QLabel("Supervised Semantic Differential")
        subtitle.setObjectName("label_welcome_subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Description
        description = QLabel("Uncover the dimensions of meaning.")
        description.setObjectName("label_welcome_desc")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)

        layout.addSpacing(48)

        # Buttons in a centered container
        button_container = QHBoxLayout()
        button_container.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)

        new_btn = QPushButton("New Project")
        new_btn.setObjectName("btn_welcome_primary")
        new_btn.setMinimumSize(200, 56)
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.clicked.connect(self.new_project)
        button_layout.addWidget(new_btn)

        open_btn = QPushButton("Open Project")
        open_btn.setObjectName("btn_welcome_secondary")
        open_btn.setMinimumSize(200, 56)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self.open_project)
        button_layout.addWidget(open_btn)

        button_container.addLayout(button_layout)
        button_container.addStretch()
        layout.addLayout(button_container)

        # Version label
        layout.addSpacing(40)
        version_text = f"v{__version__}"
        if __ssdiff_version__:
            version_text += f"  •  based on SSDiff v{__ssdiff_version__}"
        version_label = QLabel(version_text)
        version_label.setObjectName("label_muted")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        layout.addStretch(3)

        self.welcome_page = page
        self.stage_stack.addWidget(page)

    def _refresh_welcome_logo(self):
        """Re-render the welcome-page logo for the current theme."""
        from ..theme import build_current_palette, get_saved_theme_name
        from ..logo import create_logo_pixmap

        palette = build_current_palette()
        pixmap = create_logo_pixmap(palette, get_saved_theme_name(), size=355)
        if pixmap:
            self._welcome_logo.setPixmap(pixmap)

    def _create_stage_pages(self):
        """Create placeholder pages for stages (will be replaced with actual widgets)."""
        # Import here to avoid circular imports
        from .stage1_setup import Stage1Widget
        from .stage2_concept import Stage2Widget
        from .stage3_results import Stage3Widget

        # Stage 1
        self.stage1_widget = Stage1Widget()
        self.stage1_widget.stage_complete.connect(self._on_stage1_complete)
        self.stage_stack.addWidget(self.stage1_widget)

        # Stage 2
        self.stage2_widget = Stage2Widget()
        self.stage2_widget.run_requested.connect(self._on_run_requested)
        self.stage_stack.addWidget(self.stage2_widget)

        # Stage 3
        self.stage3_widget = Stage3Widget()
        self.stage3_widget.new_run_requested.connect(self._on_new_run_requested)
        self.stage3_widget.result_saved.connect(self._on_result_saved)
        self.stage_stack.addWidget(self.stage3_widget)

    def new_project(self):
        """Create a new project."""
        # Get project name
        project_name, ok = QInputDialog.getText(
            self, "New Project", "Project name:"
        )
        if not ok or not project_name.strip():
            return

        project_name = project_name.strip()

        # Use default projects directory, or ask if not set
        project_dir = self._get_projects_directory()
        if not project_dir:
            project_dir = QFileDialog.getExistingDirectory(
                self, "Select Project Location"
            )
            if not project_dir:
                return

        # Create project path
        project_path = Path(project_dir) / project_name

        # Check if exists
        if project_path.exists():
            reply = QMessageBox.question(
                self,
                "Project Exists",
                f"A folder named '{project_name}' already exists.\n"
                "Do you want to use it anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Create project structure
        try:
            ProjectIO.create_project_structure(project_path)

            # Create project object
            self.project = Project(
                project_path=project_path,
                name=project_name,
                created_date=datetime.now(),
                modified_date=datetime.now(),
            )

            # Save initial state
            ProjectIO.save_project(self.project)

            # Load into UI
            self._load_project_into_ui()

            self.status_bar.showMessage(f"Created project: {project_name}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create project: {e}"
            )

    def open_project(self):
        """Open an existing project by selecting its folder."""
        project_path_str = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", self._get_projects_directory()
        )
        if not project_path_str:
            return

        # Walk upward to find the directory containing project.json
        project_path = Path(project_path_str)
        candidate = project_path
        while candidate != candidate.parent:
            if (candidate / "project.json").exists():
                project_path = candidate
                break
            candidate = candidate.parent
        else:
            QMessageBox.critical(
                self,
                "Invalid Project Folder",
                f"No project.json found in or above:\n{project_path_str}",
            )
            return

        try:
            self.project = ProjectIO.load_project(project_path)
            self._load_project_into_ui()

            if self.project:
                self.status_bar.showMessage(f"Opened project: {self.project.name}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to open project: {e}"
            )

    def import_result(self):
        """Import a result folder from anywhere on disk into the current project.

        Mirrors open_project's upward-search: if the user picks a subdirectory
        of a result folder, walk up parents until we find one that contains
        both config.json and results.pkl. The loaded result lands in the
        unsaved slot and only persists when the user clicks Save Result.
        """
        if not self.project:
            return

        picked = QFileDialog.getExistingDirectory(
            self, "Select Result Folder", str(self.project.project_path)
        )
        if not picked:
            return

        result_path = None
        candidate = Path(picked)
        while True:
            if (candidate / "config.json").exists() and (candidate / "results.pkl").exists():
                result_path = candidate
                break
            if candidate == candidate.parent:
                break
            candidate = candidate.parent

        if result_path is None:
            QMessageBox.critical(
                self,
                "Invalid Result Folder",
                f"No config.json + results.pkl found in or above:\n{picked}",
            )
            return

        # Discard any current unsaved result (with confirmation)
        if self.stage3_widget.has_unsaved_result():
            reply = QMessageBox.warning(
                self,
                "Unsaved Result",
                "You have an unsaved result.\n"
                "Importing will discard it.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            result = ProjectIO._load_result_folder(result_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Import Failed", f"Could not load result:\n{e}"
            )
            return

        # Imported results go into the unsaved slot — no local folder yet.
        result.result_path = None
        result.folder_name = None
        result.status = "complete"

        self.stage3_action.setEnabled(True)
        self.stage3_widget.load_project(self.project)
        self.stage3_widget.show_unsaved_result(result)
        self.go_to_stage(3)
        self.status_bar.showMessage(
            f"Imported result from {result_path} (unsaved)"
        )

    def save_project(self):
        """Save the current project."""
        if not self.project:
            return

        try:
            # Update configs from UI
            self.stage1_widget.save_to_project(self.project)

            ProjectIO.save_project(self.project)
            self.project.mark_clean()
            self._update_title()
            self.status_bar.showMessage("Project saved")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save project: {e}"
            )

    def _load_project_into_ui(self):
        """Load the current project into the UI widgets."""
        if not self.project:
            return

        # Enable menus
        self.save_action.setEnabled(True)
        self.import_result_action.setEnabled(True)
        self.stage1_action.setEnabled(True)
        self.view_menu.menuAction().setVisible(True)

        # Load into stage 1 (may show a modal embeddings dialog)
        if not self.stage1_widget.load_project(self.project):
            # User cancelled a required loading step — abort project open
            self.project = None
            self.save_action.setEnabled(False)
            self.import_result_action.setEnabled(False)
            self.stage1_action.setEnabled(False)
            self._go_home()
            return

        # Determine which stage to show
        if self.project.results:
            # Has runs - show results
            self.stage2_action.setEnabled(True)
            self.stage3_action.setEnabled(True)
            self.stage2_widget.load_project(self.project)
            self.stage3_widget.load_project(self.project)
            self.go_to_stage(3)
        elif self.project.stage1_ready:
            # Ready for runs - show stage 2
            self.stage2_action.setEnabled(True)
            self.go_to_stage(2)
        else:
            # Still in setup
            self.go_to_stage(1)

        # Close the stage-1 loading dialog now that all UI is ready
        QApplication.processEvents()
        if (
            hasattr(self.stage1_widget, '_progress_dialog')
            and self.stage1_widget._progress_dialog
            and self.stage1_widget._progress_dialog.isVisible()
        ):
            self.stage1_widget._progress_dialog.set_complete("Project loaded.")
            self.stage1_widget._progress_dialog.accept()

        self._update_title()
        self.project_changed.emit(self.project)

    def go_to_stage(self, stage: int):
        """Navigate to a specific stage."""
        # Save state of the stage we're leaving
        if self.project and self.current_stage == 2:
            self.stage2_widget._save_config_to_project()
            self.project.mark_dirty()
            self._update_title()

        if stage == 1:
            self.stage_stack.setCurrentWidget(self.stage1_widget)
            self.current_stage = 1
        elif stage == 2:
            # Save latest Setup tab settings before rebuilding the review panel
            if self.project:
                self.stage1_widget.save_to_project(self.project)
                self.stage2_widget.load_project(self.project)
            self.stage_stack.setCurrentWidget(self.stage2_widget)
            self.current_stage = 2
        elif stage == 3:
            self.stage_stack.setCurrentWidget(self.stage3_widget)
            self.current_stage = 3

        # Show nav bar when in a stage
        self.stage_nav_bar.show()

        self._update_stage_actions()

    def _update_stage_actions(self):
        """Update the enabled state of stage menu actions and nav bar."""
        if not self.project:
            return

        self.stage1_action.setEnabled(True)
        self.stage2_action.setEnabled(self.project.stage1_ready)
        has_results = bool(self.project.results) or self.stage3_widget.has_unsaved_result()
        self.stage3_action.setEnabled(has_results)

        # Sync nav bar buttons
        self._stage_btns[0].setEnabled(True)
        self._stage_btns[1].setEnabled(self.project.stage1_ready)
        self._stage_btns[2].setEnabled(has_results)

        self._update_stage_nav_bar()

    def _update_title(self):
        if self.project:
            dirty_marker = "*" if self.project._dirty else ""
            self.setWindowTitle(f"{self.project.name}{dirty_marker} — SSD")
        else:
            self.setWindowTitle("SSD - Supervised Semantic Differential")

    def _on_stage1_complete(self):
        """Handle Stage 1 completion."""
        if not self.project:
            return

        if self.project.stage1_ready:
            self.stage2_action.setEnabled(True)
            self.go_to_stage(2)

            self.status_bar.showMessage("Stage 1 complete - Ready to run analysis")

    def _on_run_requested(self):
        """Handle run request from Stage 2."""
        if not self.project:
            return

        # Pre-run validation — catch obvious issues before starting the runner
        if self.project._docs is None:
            QMessageBox.warning(
                self, "Missing Data",
                "No documents loaded. Please go back to Setup and load a dataset.",
            )
            return

        if self.project._kv is None:
            QMessageBox.warning(
                self, "Missing Embeddings",
                "No word embeddings loaded.\n\n"
                "Please go back to Setup and load an embeddings file.",
            )
            return

        if self.project.analysis_type in ("pls", "pca_ols"):
            if self.project._y is None:
                QMessageBox.warning(
                    self, "Missing Outcome",
                    "No outcome variable loaded. Please select an outcome column in Setup.",
                )
                return
        elif self.project.analysis_type == "groups":
            if self.project._groups is None:
                QMessageBox.warning(
                    self, "Missing Groups",
                    "No group variable loaded. Please select a group column in Setup.",
                )
                return

        # Import SSD runner
        from ..controllers.ssd_runner import SSDRunner

        # Create and start the runner
        self.runner = SSDRunner(self.project)
        self.runner.progress.connect(self._on_run_progress)
        self.runner.finished.connect(self._on_run_finished)
        self.runner.error.connect(self._on_run_error)

        # Show progress dialog
        from .widgets.progress_dialog import ProgressDialog
        self.progress_dialog = ProgressDialog("Running SSD Analysis", self)
        self.progress_dialog.cancel_button.clicked.connect(self.runner.cancel)

        self.progress_dialog.show()
        QApplication.processEvents()
        self.runner.start()
        self.progress_dialog.exec()

    def _on_run_progress(self, percent: int, message: str):
        """Handle progress updates from runner."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(percent, message)

    def _on_run_finished(self, result):
        """Handle run completion — show as unsaved, don't persist yet."""
        if hasattr(self, 'progress_dialog') and self.progress_dialog.is_cancelled():
            return

        try:
            if self.project:
                self.project.mark_dirty()

            # Set up Stage 3 BEFORE closing the dialog so the UI is fully
            # rendered behind it and there's no flash of stale content.
            self.stage3_action.setEnabled(True)
            self.stage3_widget.load_project(self.project)
            self.stage3_widget.show_unsaved_result(result)
            self.go_to_stage(3)
            QApplication.processEvents()  # let UI paint behind dialog
        except Exception:
            import traceback
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.set_error(
                    f"Failed to display results:\n\n{traceback.format_exc()}"
                )
            return

        # NOW close the dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.set_complete("Analysis complete!")
            self.progress_dialog.accept()

        self.status_bar.showMessage(
            f"Analysis complete — Result ID: {result.result_id}  (unsaved)"
        )

    def _on_result_saved(self):
        """Handle the result_saved signal from Stage 3."""
        self.status_bar.showMessage("Result saved")
        if self.project:
            self.project.mark_dirty()

    def _on_run_error(self, error_message: str):
        """Handle run error."""
        if hasattr(self, 'progress_dialog'):
            if self.progress_dialog.is_cancelled():
                return
            self.progress_dialog.set_error(error_message)

    def _on_new_run_requested(self):
        """Handle request for a new run from Stage 3."""
        self.go_to_stage(2)

    def _get_projects_directory(self) -> str:
        """Return the stored default projects directory, or empty string."""
        return self._settings.value("projects_directory", "")

    def _open_settings_dialog(self):
        """Open the application settings dialog."""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            new_dir = self._get_projects_directory()
            if new_dir:
                self.status_bar.showMessage(f"Projects directory: {new_dir}")

    def _open_appearance_dialog(self):
        """Open the appearance/theme settings dialog."""
        from .appearance_dialog import AppearanceDialog
        dialog = AppearanceDialog(self)
        dialog.exec()

    def check_first_run_settings(self):
        """On first launch, prompt the user to set a default projects directory."""
        if self._get_projects_directory():
            return

        reply = QMessageBox.question(
            self,
            "Welcome to SSD",
            "It looks like this is your first time running SSD.\n\n"
            "Would you like to configure your settings now?\n"
            "You can set a default projects directory and other preferences.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._open_settings_dialog()

        # Always offer appearance choice on first run
        self._open_appearance_dialog()

    def _open_tutorial(self):
        """Open the tutorial dialog (non-modal so the main UI stays usable)."""
        from .tutorial_dialog import TutorialDialog

        # Reuse existing window if still open
        if hasattr(self, "_tutorial_dialog") and self._tutorial_dialog is not None:
            self._tutorial_dialog.raise_()
            self._tutorial_dialog.activateWindow()
            return

        self._tutorial_dialog = TutorialDialog(self)
        self._tutorial_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self._tutorial_dialog.destroyed.connect(
            lambda: setattr(self, "_tutorial_dialog", None)
        )
        self._tutorial_dialog.show()

    def show_about(self):
        """Show the about dialog."""
        ssdiff_line = f"<p>Based on SSDiff v{__ssdiff_version__}</p>" if __ssdiff_version__ else ""
        QMessageBox.about(
            self,
            "About SSD",
            "<h2>SSD</h2>"
            "<p>Supervised Semantic Differential</p>"
            f"<p>Version {__version__}</p>"
            f"{ssdiff_line}"
            "<p>A desktop application for running Supervised Semantic "
            "Differential analysis on text data.</p>"
            "<p>Designed for psychologists and researchers working with "
            "text-based outcome data.</p>"
            "<hr>"
            "<p><b>Author:</b> Hubert Plisiecki</p>"
            "<p><b>Contact:</b> <a href='mailto:hplisiecki@gmail.com'>hplisiecki@gmail.com</a></p>"
            "<p><b>GitHub:</b> <a href='https://github.com/hplisiecki/SSD_APP'>github.com/hplisiecki/SSD_APP</a></p>"
        )

    def _start_update_check(self):
        """Silently check GitHub for a newer release in the background."""
        from ..utils.worker_threads import UpdateCheckWorker
        self._update_worker = UpdateCheckWorker(__version__)
        self._update_worker.update_available.connect(self._on_update_available)
        self._update_worker.start()

    def _on_update_available(self, version: str, url: str):
        """Show the update banner at the bottom of the window."""
        from .widgets.update_banner import UpdateBanner
        banner = UpdateBanner(version, url, parent=self.centralWidget())
        self.centralWidget().layout().addWidget(banner)
        self._update_banner = banner

    def _validate_window_geometry(self) -> bool:
        """Check that the restored window is visible and fits the display.

        Returns True if the geometry is acceptable, False if a fallback
        to default size/position is needed.
        """
        rect = self.frameGeometry()
        app = QApplication.instance()
        if app is None:
            return False

        # Check if the window intersects any available screen
        for screen in app.screens():
            if screen.availableGeometry().intersects(rect):
                # Clamp size to fit within this screen's bounds
                avail = screen.availableGeometry()
                w = min(rect.width(), avail.width())
                h = min(rect.height(), avail.height())
                if w != rect.width() or h != rect.height():
                    self.resize(w, h)
                return True

        return False

    def _save_window_geometry(self):
        """Persist current window geometry and maximized state to settings."""
        self._settings.setValue("window/maximized", self.isMaximized())
        if not self.isMaximized():
            self._settings.setValue("window/geometry", self.saveGeometry())

    def closeEvent(self, event):
        """Handle window close event."""
        if self.project:
            # Warn about unsaved analysis run
            if self.stage3_widget.has_unsaved_result():
                reply = QMessageBox.warning(
                    self,
                    "Unsaved Result",
                    "You have an unsaved result.\n"
                    "If you close now, it will be discarded.\n\n"
                    "Go back to save it?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.go_to_stage(3)
                    event.ignore()
                    return

            if self.project._dirty:
                reply = QMessageBox.question(
                    self,
                    "Save Project?",
                    "Do you want to save the project before closing?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                )
                if reply == QMessageBox.Save:
                    self.save_project()
                    self._save_window_geometry()
                    event.accept()
                elif reply == QMessageBox.Discard:
                    self._save_window_geometry()
                    event.accept()
                else:
                    event.ignore()
            else:
                self._save_window_geometry()
                event.accept()
        else:
            self._save_window_geometry()
            event.accept()
