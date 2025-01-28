import threading

from agents import NN
from network import NetParams
from algos import QLearn
from lbf import LBF
from params import Param, UniformDecay

from imports import *
from gui import Gui, EnvControl
from gridenv import LBFScene


if __name__ == '__main__':
    agent = NN(
        env=LBF(NUM_CELLS_X, NUM_CELLS_Y, NUM_ITEMS),
        algo=QLearn(gamma=0.9, nstep=2),
        netparams=NetParams([10], cost_type='mse'),
        batch=100,
        upd_interval=1000,
    )
    app = QApplication([])
    gui = Gui(agent, EnvControl(LBFScene(WIDTH, HEIGHT)))

    # t1 = threading.Thread(
    #     target=agent.train,
    #     kwargs={'n': 1000, 'eps': Param(0.1), 'alpha': Param(0.01),
    #             'batch_size': 1, 'display_graph': False},
    #     daemon=True,
    # )
    # t1.start()

    # gui.start()
    # gui.animate()
    gui.show()
    app.exec()
