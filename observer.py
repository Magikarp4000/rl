from rlsignal import RLSignal


class Observable:
    def __init__(self):
        self.observers = []
    
    def attach(self, obj):
        self.observers.append(obj)
    
    def detach(self, obj):
        self.observers.remove(obj)
    
    def notify(self, signal=RLSignal.DEFAULT):
        for obj in self.observers:
            obj.respond(self, signal)


class Observer:
    def respond(self, obj: Observable, signal: RLSignal):
        pass
