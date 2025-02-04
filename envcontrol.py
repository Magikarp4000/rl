from imports import *

from observer import Observable, Observer
from agentobserver import AgentObserver
from rlsignal import RLSignal


class EnvControl(Observable, Observer, AgentObserver):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.step = None
        self.action = None
        self.aval = None
        self.avals = None

        self.fps = FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def update(self):
        self.step = self.agent.replay.read()
        if self.step != None:
            self.action = self.agent.env.actions[self.step.a]
            self.aval = self.agent.q(self.step.s, self.step.a)
            # self.avals = self.agent.action_vals_xnn(self.step.s, self.agent.bnn).flat
            self.update_state(*self.step.sar)
            self.notify(RLSignal.EP_UPDATE)
        self.agent.replay.next()
        
    def update_state(self, s, a, r):
        self.scene.update_state(*self.agent.env.decode_state(s))
    
    def respond(self, obj, signal):
        match signal:
            case RLSignal.TRAIN_START | RLSignal.TEST_START:
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
