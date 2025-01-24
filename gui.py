from imports import *
from utils import get_dir


class EnvScene(QGraphicsScene):
    def __init__(self, width=400, height=250):
        super().__init__(0, 0, width, height)
        self.setBackgroundBrush(QBrush(Qt.black))


class EnvView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.viewport().installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ensureVisible(self.scene().sceneRect())


class Gui(QWidget):
    def __init__(self, scene):
        super().__init__()

        # window
        self.setWindowTitle('RL')
        
        self.scene = scene
        self.view = EnvView(self.scene)

        self.env_widget = QWidget()
        self.env = QVBoxLayout(self.env_widget)
        self.env.addWidget(self.view)

        self.setLayout(self.env)
