import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt


class CameraPanel(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setStyleSheet("""
            QFrame{
                background:#1c1f26;
                border:1px solid #2c313c;
                border-radius:8px;
            }
        """)
        layout = QVBoxLayout()

        title = QLabel(name)
        title.setStyleSheet("color:#00C853;font-weight:bold")
        layout.addWidget(title)

        feed = QLabel("No Signal")
        feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feed.setStyleSheet("color:#777;font-size:16px")
        layout.addWidget(feed)

        self.setLayout(layout)


class RelayPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:#1c1f26;border-radius:8px;")
        layout = QHBoxLayout()

        for i in range(1,10):
            r = QLabel(f"R{i}\nOFF")
            r.setAlignment(Qt.AlignmentFlag.AlignCenter)
            r.setStyleSheet("""
                QLabel{
                    background:#2c313c;
                    color:#aaa;
                    padding:10px;
                    border-radius:5px;
                    min-width:40px;
                }
            """)
            layout.addWidget(r)

        self.setLayout(layout)


class StatusBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:#1c1f26;border-radius:6px;")

        layout = QHBoxLayout()
        stats = [
            "CPU: 5%",
            "RAM: 10GB",
            "GPU: 32%",
            "VRAM: 1.2GB",
            "Temp: 60°C",
            "Uptime: 00:02:34"
        ]

        for s in stats:
            lbl = QLabel(s)
            lbl.setStyleSheet("color:#E0E0E0;padding:6px")
            layout.addWidget(lbl)

        layout.addStretch()
        self.setLayout(layout)


class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Industrial Vision System")
        self.setStyleSheet("background:#0f1115")

        main_layout = QVBoxLayout()

        # Top status bar
        main_layout.addWidget(StatusBar())

        # Camera grid
        cam_grid = QGridLayout()
        cam_grid.addWidget(CameraPanel("Camera 1"),0,0)
        cam_grid.addWidget(CameraPanel("Camera 2"),0,1)
        cam_grid.addWidget(CameraPanel("Camera 3"),0,2)

        main_layout.addLayout(cam_grid)

        # Relay panel
        relay_title = QLabel("Relay Status")
        relay_title.setStyleSheet("color:#00C853;font-weight:bold")
        main_layout.addWidget(relay_title)

        main_layout.addWidget(RelayPanel())

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.resize(1400,800)
    window.show()
    sys.exit(app.exec())