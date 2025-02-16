import threading

from agents import NN
from network import NetParams
from algos import QLearn, ExploreBonus
from params import Param, UniformDecay
from lbf import LBF
from snake import Snake

from imports import *
from gui import Gui, EnvControl
from gridenv import LBFScene, SnakeScene


if __name__ == '__main__':
    agent = NN(
        env=LBF(NUM_CELLS_X, NUM_CELLS_Y),
        algo=QLearn(gamma=0.99, nstep=1),
        netparams=NetParams([32, 16], cost_type='mse'),
        batch=512,
        upd_interval=100,
        buf_size=10000
    )
    # agent.running = True
    # agent.train(n=1000, eps=UniformDecay(0.2, 0.05, decay=5*10**-5), alpha=Param(3*10**-4),
    #             maxstep=1000, batch_size=1, display_graph=False)
    app = QApplication([])
    gui = Gui(agent, LBFScene())
    gui.show()
    app.exec()
