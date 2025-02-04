from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from imports import *

from baseagent import Agent
from envcontrol import EnvControl

from observer import Observer, Observable
from agentobserver import AgentObserver
from rlsignal import RLSignal


class Clock(Observable):
    def __init__(self):
        super().__init__()
        self._clock = QTimer()
        self._clock.timeout.connect(lambda: self.notify(RLSignal.CLOCK_UPDATE))
    
    def start(self):
        self._clock.start()


class EnvScene(QGraphicsScene):
    def update_state(self, *args, **kwargs):
        pass


class EnvView(QGraphicsView):
    def __init__(self, scene: EnvScene, width=0, height=0):
        super().__init__(scene)
        self.setFixedSize(width, height)
        self.scene().setParent(self)
        self.setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("border: 2px solid white;")
        self.ensureVisible(self.scene().sceneRect())
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = event.size().toTuple()
        self.scene().setSceneRect(0, 0, w, h)


class GuiObject(Observer, Observable):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._wgt = None
    
    def set_widget(self, wgt: QWidget):
        self._wgt = wgt

    def widget(self):
        return self._wgt


class GuiLabel(GuiObject, AgentObserver):
    def __init__(self, text=None, width=0):
        super().__init__()
        self.label = QLabel(text)
        self.label.setStyleSheet("color: white; font-size: 14pt")
        self.label.setFixedWidth(width)
        self.upd_text = ""
        self.env_text = ""
        self.set_widget(self.label)
    
    def respond(self, obj, signal):
        if signal == RLSignal.CLOCK_UPDATE:
            self.upd_text = f"Global step: {self.agent.glo_steps}"
        elif signal == RLSignal.VIEW_UPDATE:
            ep_num = f"Episode: {obj.step.ep_num}"
            step_num = f"Step: {obj.step.step_num}"
            action = f"Action: {obj.action}"
            reward = f"Reward: {round(obj.step.r, 3)}"
            cumr = f"Cumr: {round(obj.step.cumr, 3)}"
            aval = f"Action-Value: {round(obj.aval, 6)}"
            avals = "\n".join([str(round(x, 3)) for x in obj.step.avals])
            tgt = f"Target-Value: {round(obj.step.tgt, 3)}"
            self.env_text = f"{ep_num}\n{step_num}\n{action}\n{cumr}\n{reward}\n{tgt}\nAction-Values:\n{avals}"
        self._update_text()
    
    def _update_text(self):
        self.label.setText(f"Info:\n\n{self.upd_text}\n{self.env_text}")


class GuiButton(QPushButton):
    def __init__(self, text=None, bgcolor="white", color="black", fontsize=12):
        super().__init__(text=text)
        self.setStyleSheet(f"background-color: {bgcolor}; color: {color}; font-size: {fontsize}pt")
        self.setFixedSize(100, 50)


class TrainButton(GuiObject):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Train", "lightblue")
        self.btn.clicked.connect(self.click)
        self.set_widget(self.btn)
    
    def click(self):
        self.btn.setEnabled(False)
        self.notify(RLSignal.TRAIN_START)
    
    def respond(self, obj, signal):
        if signal == RLSignal.STOP_SIMULATION:
            self.btn.setEnabled(True)
        elif signal == RLSignal.TEST_START:
            self.btn.setEnabled(False)


class StopButton(GuiObject):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Stop", "red")
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self.click)
        self.set_widget(self.btn)
    
    def click(self):
        self.btn.setEnabled(False)
        self.notify(RLSignal.STOP_SIMULATION)

    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_START or signal == RLSignal.TEST_START:
            self.btn.setEnabled(True)


class TestButton(GuiObject):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Test", "lightblue")
        self.btn.clicked.connect(self.click)
        self.set_widget(self.btn)
    
    def click(self):
        self.btn.setEnabled(False)
        self.notify(RLSignal.TEST_START)
    
    def respond(self, obj, signal):
        if signal == RLSignal.STOP_SIMULATION:
            self.btn.setEnabled(True)
        elif signal == RLSignal.TRAIN_START:
            self.btn.setEnabled(False)


class SyncButton(GuiObject):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Sync", "lightblue")
        self.btn.clicked.connect(lambda: self.notify(RLSignal.SYNC_WITH_AGENT))
        self.set_widget(self.btn)


class AutoSyncButton(GuiObject):
    def __init__(self):
        super().__init__()
        self.btn = GuiButton("Auto-Sync", "lightblue")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(lambda: self.notify(RLSignal.TOGGLE_AUTO_SYNC))
        self.set_widget(self.btn)


class FPSSlider(GuiObject):
    def __init__(self):
        super().__init__()
        self.wgt = QWidget()
        self.wgt.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout = QVBoxLayout(self.wgt)
        self.set_widget(self.wgt)

        self.slider = QSlider()
        self.slider.setMinimum(1)
        self.slider.setMaximum(200)
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


class GuiGraph(GuiObject):
    def __init__(self, width=0, height=0):
        super().__init__()

        dpi = plt.rcParams['figure.dpi']
        self.fig = Figure(figsize=(width / dpi, height / dpi))
        self.fig.set_facecolor(BG_COLOUR)

        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel("Steps", color='white')
        self.ax.set_ylabel("Action-Value", color='white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['right'].set_alpha(0)
        self.ax.spines['top'].set_alpha(0)
        self.ax.tick_params(axis='x', colors='white', labelsize=8)
        self.ax.tick_params(axis='y', colors='white', labelsize=8)
        self.ax.set_facecolor(BG_COLOUR)

        self.graph = FigureCanvasQTAgg(self.fig)
        self.set_widget(self.graph)

        self.prev = None
    
    def respond(self, obj: EnvControl, signal):
        if signal == RLSignal.VIEW_UPDATE:
            cur = obj.aval
            t = obj.step.step_num
            if t > 0:
                lines = self.ax.get_lines()
                if len(lines) >= 100:
                    self.ax.get_lines()[0].remove()
                    self.ax.set_xlim(t - 99, t)
                self.ax.plot([t-1, t], [self.prev, cur], color='white', lw=1)
                self.fig.canvas.draw()
            self.prev = cur
        elif signal == RLSignal.VIEW_NEW_EP:
            self.ax.clear()


class Gui(QWidget, Observer):
    def __init__(self, agent: Agent, scene: EnvScene):
        super().__init__()
        self.agent = agent
        self.scene = scene

        width, height = self.screen().size().toTuple()
        self.view = EnvView(self.scene, width / 2, height * 3 / 5)
        self.control = EnvControl(self.scene)

        self.train_btn = TrainButton()
        self.test_btn = TestButton()
        self.stop_btn = StopButton()
        self.sync_btn = SyncButton()
        self.auto_sync_btn = AutoSyncButton()
        self.fpsslider = FPSSlider()

        self.ctrl_panel_wgt = QWidget()
        self.ctrl_panel = QGridLayout(self.ctrl_panel_wgt)
        self.ctrl_panel.addWidget(self.train_btn.widget(), 0, 0)
        self.ctrl_panel.addWidget(self.test_btn.widget(), 0, 1)
        self.ctrl_panel.addWidget(self.stop_btn.widget(), 1, 0)
        self.ctrl_panel.addWidget(self.sync_btn.widget(), 0, 2)
        self.ctrl_panel.addWidget(self.auto_sync_btn.widget(), 0, 3)
        self.ctrl_panel.addWidget(self.fpsslider.widget(), 0, 4)
        self.ctrl_panel.setAlignment(Qt.AlignLeft)

        self.info = GuiLabel("Info:", width / 6)

        self.side_panel_wgt = QWidget()
        self.side_panel_wgt.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.side_panel = QGridLayout(self.side_panel_wgt)
        self.side_panel.addWidget(self.info.widget())
        self.side_panel.setAlignment(Qt.AlignTop)

        self.action_graph = GuiGraph(400, 400)
        self.graph_panel_wgt = QWidget()
        self.graph_panel_wgt.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.graph_panel = QGridLayout(self.graph_panel_wgt)
        self.graph_panel.addWidget(self.action_graph.widget())
        self.graph_panel.setAlignment(Qt.AlignTop)

        self.control.observe(agent)
        self.info.observe(agent)

        self.control.attach(self.info)
        self.control.attach(self.action_graph)

        self.train_btn.attach(self.stop_btn)
        self.train_btn.attach(self.test_btn)
        self.train_btn.attach(self.agent)
        self.train_btn.attach(self.control)

        self.test_btn.attach(self.stop_btn)
        self.test_btn.attach(self.train_btn)
        self.test_btn.attach(self.control)
        self.test_btn.attach(self.agent)

        self.stop_btn.attach(self.train_btn)
        self.stop_btn.attach(self.test_btn)
        self.stop_btn.attach(self.agent)
        self.stop_btn.attach(self.control)

        self.sync_btn.attach(self.control)

        self.auto_sync_btn.attach(self.control)

        self.fpsslider.attach(self.control)

        self.setWindowTitle('RL')
        self.setStyleSheet(f"background-color: {BG_COLOUR};")
        self.root = QGridLayout()
        self.root.addWidget(self.view, 0, 0)
        self.root.addWidget(self.ctrl_panel_wgt, 1, 0)
        self.root.addWidget(self.side_panel_wgt, 0, 1)
        self.root.addWidget(self.graph_panel_wgt, 0, 2)
        self.root.setAlignment(Qt.AlignLeft)
        self.setLayout(self.root)

        self.showMaximized()
        
        self.clock = Clock()
        self.clock.attach(self.info)
        self.clock.attach(self)
        self.clock.start()

    
    # def respond(self, obj, signal):
    #     if signal == RLSignal.CLOCK_UPDATE:
    #         print(self.view.size())