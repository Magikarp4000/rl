from imports import *

from baseagent import Agent

from observer import Observer, Observable
from agentobserver import AgentObserver
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


class EnvControl(Observable, Observer, AgentObserver):
    def __init__(self, scene: EnvScene):
        super().__init__()
        self.scene = scene
        self.step = None
        
        self.fps = FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def update(self):
        self.step = self.agent.replay.read()
        self.update_state(*self.step.sar)
        self.notify(RLSignal.EP_UPDATE)
        self.agent.replay.next()
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.agent.env.decode_state(s))
    
    def respond(self, obj, signal):
        match signal:
            case RLSignal.TRAIN_START:
                self.timer.start(1000 / self.fps)
            case RLSignal.TRAIN_STOP:
                self.timer.stop()
            case RLSignal.FPS_CHANGE:
                self.fps = obj.slider.value()
                self.timer.setInterval(1000 / self.fps)
            case RLSignal.SYNC_WITH_AGENT:
                self.agent.replay.sync()
            case RLSignal.TOGGLE_AUTO_SYNC:
                self.agent.replay.toggle_auto_sync()


class GuiLabel(Observer, AgentObserver):
    def __init__(self, text=None):
        super().__init__()
        self.label = QLabel(text)
        self.label.setStyleSheet("color: white; font-size: 14pt")
        self.upd_text = ""
        self.env_text = ""
    
    def respond(self, obj, signal):
        if signal == RLSignal.CLOCK_UPDATE:
            self.upd_text = f"Global step: {self.agent.glo_steps}"
        elif signal == RLSignal.EP_UPDATE:
            ep_num = f"Episode: {obj.step.ep_num}"
            step_num = f"Step: {obj.step.step_num}"
            action = f"Action: {obj.step.a}"
            self.env_text = f"{ep_num}\n{step_num}\n{action}"
        self._update_text()
    
    def _update_text(self):
        self.label.setText(f"Info:\n\n{self.upd_text}\n{self.env_text}")


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
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self.click)
    
    def click(self):
        self.btn.setEnabled(False)
        self.notify(RLSignal.TRAIN_STOP)

    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_START:
            self.btn.setEnabled(True)


class SyncButton(Observable):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Sync", "lightblue")
        self.btn.clicked.connect(lambda: self.notify(RLSignal.SYNC_WITH_AGENT))


class AutoSyncButton(Observable):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Auto-Sync", "lightblue")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(lambda: self.notify(RLSignal.TOGGLE_AUTO_SYNC))


class FPSSlider(Observable):
    def __init__(self):
        super().__init__()
        self.wgt = QWidget()
        self.wgt.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout = QVBoxLayout(self.wgt)

        self.slider = QSlider()
        self.slider.setMinimum(1)
        self.slider.setMaximum(60)
        self.slider.setValue(FPS)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimumWidth(180)
        self.slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slider.valueChanged.connect(self.update)

        self.text = QLabel(f"FPS: {self.slider.value()}")
        self.text.setStyleSheet("color: white; font-size: 12pt")
        self.text.setAlignment(Qt.AlignHCenter)

        self.layout.addWidget(self.text)
        self.layout.addWidget(self.slider)
    
    def update(self):
        self.text.setText(f"FPS: {self.slider.value()}")
        self.notify(RLSignal.FPS_CHANGE)


class Clock(Observable):
    def __init__(self):
        super().__init__()
        self._clock = QTimer()
        self._clock.timeout.connect(lambda: self.notify(RLSignal.CLOCK_UPDATE))
    
    def start(self):
        self._clock.start()


class Gui(QWidget):
    def __init__(self, agent: Agent, control: EnvControl):
        super().__init__()
        self.agent = agent
        self.control = control

        self.env_view = EnvView(self.control.scene)
        self.info = GuiLabel("Info:")
        self.train_btn = TrainButton()
        self.stop_btn = StopButton()
        self.sync_btn = SyncButton()
        self.auto_sync_btn = AutoSyncButton()
        self.fpsslider = FPSSlider()

        self.ctrl_panel_wgt = QWidget()
        self.ctrl_panel = QHBoxLayout(self.ctrl_panel_wgt)
        self.ctrl_panel.addWidget(self.train_btn.btn)
        self.ctrl_panel.addWidget(self.stop_btn.btn)
        self.ctrl_panel.addWidget(self.sync_btn.btn)
        self.ctrl_panel.addWidget(self.auto_sync_btn.btn)
        self.ctrl_panel.addWidget(self.fpsslider.wgt)
        self.ctrl_panel.setAlignment(Qt.AlignLeft)

        self.control.observe(agent)
        self.info.observe(agent)
        self.control.attach(self.info)
        self.train_btn.attach(self.stop_btn)
        self.train_btn.attach(self.agent)
        self.train_btn.attach(self.control)
        self.stop_btn.attach(self.train_btn)
        self.stop_btn.attach(self.agent)
        self.stop_btn.attach(self.control)
        self.sync_btn.attach(self.control)
        self.auto_sync_btn.attach(self.control)
        self.fpsslider.attach(self.control)

        self.setWindowTitle('RL')
        self.setStyleSheet(f"background-color: {BG_COLOUR};")
        self.root = QGridLayout()
        self.root.addWidget(self.env_view, 0, 0)
        self.root.addWidget(self.info.label, 0, 1)
        self.root.addWidget(self.ctrl_panel_wgt, 1, 0)
        self.setLayout(self.root)

        self.clock = Clock()
        self.clock.attach(self.info)
        self.clock.start()
