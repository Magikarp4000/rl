from imports import *
from exceptions import *

from observer import Observable, Observer
from agentobserver import AgentObserver
from rlsignal import RLSignal


class EnvControl(Observable, Observer, AgentObserver):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.step = None
        self.ep = None

        self.fps = FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def update(self):
        try:
            self.step = self.agent.replay.read()
            self.ep = self.agent.replay.read_ep()
            self.update_state(self.step.s, self.step.a, self.step.r)
            if self.step.t == 0:
                self.notify(RLSignal.VIEW_NEW_EP)
            self.notify(RLSignal.VIEW_UPDATE)
        except BufferIndexError:
            pass
        self.agent.replay.next()
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.agent.env.decode_state(s))
    
    def respond(self, obj, signal):
        match signal:
            case RLSignal.TRAIN_CLICKED | RLSignal.TEST_CLICKED:
                self.timer.start(1000 / self.fps)
            case RLSignal.STOP_SIMULATION:
                self.timer.stop()
            case RLSignal.FPS_CHANGE:
                self.fps = obj.slider.value()
                self.timer.setInterval(1000 / self.fps)
            case RLSignal.SYNC_WITH_AGENT:
                self.agent.replay.sync()
            case RLSignal.TOGGLE_AUTO_SYNC:
                self.agent.replay.toggle_auto_sync()
