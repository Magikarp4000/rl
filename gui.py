from imports import *

from baseagent import Agent

from observer import Observer, Observable
from rlsignal import RLSignal


class EnvScene(QGraphicsScene):
    def update_state(self, *args, **kwargs):
        pass


class EnvView(QGraphicsView):
    def __init__(self, scene: EnvScene):
        super().__init__(scene)
        self.scene().setParent(self)
        self.setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("border: 2px solid white;")
        self.ensureVisible(self.scene().sceneRect())


class EnvControl(Observable):
    def __init__(self, scene: EnvScene):
        super().__init__()
        self.agent = None
        self.scene = scene
        self.step = None
    
    def observe(self, agent: Agent):
        self.agent = agent

    def update(self):
        self.step = self.agent.replay.read()
        self.update_state(*self.step.sar)
        self.notify(RLSignal.EP_UPDATE)
        self.agent.replay.next()
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.agent.env.decode_state(s))


class Label(Observer):
    def __init__(self, text=None):
        super().__init__()
        self.label = QLabel(text)
        self.label.setStyleSheet("color: white; font-size: 14pt")
    
    def respond(self, obj: EnvControl, signal):
        if signal == RLSignal.EP_UPDATE:
            ep_num = f"Episode: {obj.step.ep_num}"
            step_num = f"Step: {obj.step.step_num}"
            action = f"Action: {obj.step.a}"
            self.label.setText(f"{ep_num}\n{step_num}\n{action}")


class Gui(QWidget):
    def __init__(self, agent: Agent, control: EnvControl):
        super().__init__()
        self.agent = agent
        self.control = control
        self.info = Label("Info")

        self.control.observe(agent)
        self.control.attach(self.info)

        # window
        self.setWindowTitle('RL')
        # self.setStyleSheet("background-color: #000d6e;")
        self.setStyleSheet("background-color: #051232;")
        self.root = QGridLayout()

        # env
        self.env_view = EnvView(self.control.scene)
        self.env = QVBoxLayout(self.env_view)

        self.root.addWidget(self.env_view)
        self.root.addWidget(self.info.label)
        self.setLayout(self.root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def start(self):
        self.timer.start(1000 / FPS)

    def update(self):
        self.control.update()

    def animate(self):
        self.control.start()
