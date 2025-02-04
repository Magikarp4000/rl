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
        env=Snake(NUM_CELLS_X, NUM_CELLS_Y),
        algo=QLearn(gamma=0.99, nstep=1),
        netparams=NetParams([32, 16], cost_type='mse'),
        batch=128,
        upd_interval=100,
        buf_size=2500
    )
    app = QApplication([])
    gui = Gui(agent, SnakeScene())
    gui.show()
    app.exec()
