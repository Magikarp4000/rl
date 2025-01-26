import threading

from agents import NN
from network import NetParams
from algos import QLearn
from lbf import LBF
from params import Param

from imports import *
from gui import Gui, EnvControl
from gridenv import LBFScene


if __name__ == '__main__':
    agent = NN(
        env=LBF(NUM_CELLS_X, NUM_CELLS_Y, NUM_ITEMS),
        algo=QLearn(gamma=0.9, nstep=2),
        netparams=NetParams([6], cost_type='mse'),
        batch=20,
        upd_interval=100,
    )
    app = QApplication([])
    gui = Gui(EnvControl(agent, LBFScene(WIDTH, HEIGHT)))

    t1 = threading.Thread(
        target=agent.train,
        kwargs={'n': 1, 'eps': Param(0.1), 'alpha': Param(0.1),
                'batch_size': 1, 'display_graph': False},
        daemon=True,
    )
    t1.start()

    gui.animate()
    gui.show()
    app.exec()
