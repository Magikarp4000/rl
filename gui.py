from imports import *

from baseagent import Agent


class EnvScene(QGraphicsScene):
    def update_state(*args, **kwargs):
        pass


class EnvView(QGraphicsView):
    def __init__(self, scene: EnvScene):
        super().__init__(scene)
        self.scene().setParent(self)
        self.setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ensureVisible(self.scene().sceneRect())


class EnvControl:
    def __init__(self, agent: Agent, scene: EnvScene):
        self.agent = agent
        self.scene = scene
        self.env = self.agent.env

        self.active_id = self.agent.active_id
        self.step_id = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def start(self):
        self.step_id = 0
        self.timer.start(1000 / FPS)

    def update(self):
        try:
            step = self.agent.replay[self.active_id][self.step_id]
            self.update_state(*step)
            print(self.env.actions[step[1]], end=" ", flush=True)
            self.step_id += 1
        except IndexError:
            self.active_id = self.agent.active_id
            self.step_id = 0
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.env.decode_state(s))


class Gui(QWidget):
    def __init__(self, envcontrol: EnvControl):
        super().__init__()

        # window
        self.setWindowTitle('RL')
        
        self.control = envcontrol
        self.view = EnvView(self.control.scene)

        self.env_widget = QWidget()
        self.env = QVBoxLayout(self.env_widget)
        self.env.addWidget(self.view)

        self.setLayout(self.env)

    def animate(self):
        self.control.start()
