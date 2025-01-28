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


class EnvControl(Observable, Observer):
    def __init__(self, scene: EnvScene):
        super().__init__()
        self.agent = None
        self.scene = scene
        self.step = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def observe(self, agent: Agent):
        self.agent = agent

    def update(self):
        self.step = self.agent.replay.read()
        self.update_state(*self.step.sar)
        self.notify(RLSignal.EP_UPDATE)
        self.agent.replay.next()
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.agent.env.decode_state(s))
    
    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_START:
            self.timer.start(1000 / FPS)
        elif signal == RLSignal.TRAIN_STOP:
            self.timer.stop()


class GuiLabel(Observer):
    def __init__(self, text=None):
        super().__init__()
        self.label = QLabel(text)
        self.label.setStyleSheet("color: white; font-size: 14pt")
    
    def respond(self, obj: EnvControl, signal):
        if signal == RLSignal.EP_UPDATE:
            ep_num = f"Episode: {obj.step.ep_num}"
            step_num = f"Step: {obj.step.step_num}"
            action = f"Action: {obj.step.a}"
            self.label.setText(f"Info:\n{ep_num}\n{step_num}\n{action}")


class GuiButton(QPushButton):
    def __init__(self, text=None, bgcolor="white", color="black", fontsize=12):
        super().__init__(text=text)
        self.setStyleSheet(f"background-color: {bgcolor}; color: {color}; font-size: {fontsize}pt")
        self.setFixedSize(100, 50)


class TrainButton(Observable, Observer):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Train", "lightblue")
        self.btn.clicked.connect(self.click)
    
    def click(self):
        self.btn.setEnabled(False)
        self.notify(RLSignal.TRAIN_START)
    
    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_STOP:
            self.btn.setEnabled(True)


class StopButton(Observable, Observer):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Stop", "red")
        self.btn.setVisible(False)
        self.btn.clicked.connect(self.click)
    
    def click(self):
        self.btn.setVisible(False)
        self.notify(RLSignal.TRAIN_STOP)

    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_START:
            self.btn.setVisible(True)


class Gui(QWidget):
    def __init__(self, agent: Agent, control: EnvControl):
        super().__init__()
        self.agent = agent
        self.control = control

        self.env_view = EnvView(self.control.scene)
        self.info = GuiLabel("Info:")
        self.train_btn = TrainButton()
        self.stop_btn = StopButton()

        self.control.observe(agent)
        self.control.attach(self.info)
        self.train_btn.attach(self.stop_btn)
        self.train_btn.attach(self.agent)
        self.train_btn.attach(self.control)
        self.stop_btn.attach(self.train_btn)
        self.stop_btn.attach(self.agent)
        self.stop_btn.attach(self.control)

        self.setWindowTitle('RL')
        self.setStyleSheet(f"background-color: {BG_COLOUR};")
        self.root = QGridLayout()
        self.root.addWidget(self.env_view, 0, 0)
        self.root.addWidget(self.info.label, 0, 1)
        self.root.addWidget(self.train_btn.btn, 1, 0)
        self.root.addWidget(self.stop_btn.btn, 2, 0)
        self.setLayout(self.root)
