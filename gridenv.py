import random

import numpy as np

from imports import *
from gui import Gui, EnvScene, EnvControl

from utils import flatten
from baseagent import Agent


class GridSquare(QGraphicsRectItem):
    def __init__(self, x, y, size):
        super().__init__()
        self.setPen(QPen(Qt.white))
        self.setBrush(Qt.black)
        self.setRect(x * size, y * size, size, size)


class GridScene(EnvScene):
    def __init__(self, width=400, height=250):
        super().__init__(0, 0, width, height)
        self.setBackgroundBrush(QBrush(Qt.black))

        self.grid_width = CELL_SIZE * NUM_CELLS_X
        self.grid_height = CELL_SIZE * NUM_CELLS_Y
        self.grid = [[GridSquare(x, y, CELL_SIZE) for x in range(NUM_CELLS_X)]
                     for y in range(NUM_CELLS_Y)]
        self.cells = self.createItemGroup(flatten(self.grid))
    
    def mouseMoveEvent(self, event):
        d_pos = event.scenePos() - event.lastScenePos()
        if Qt.MouseButton.LeftButton in event.buttons():
            self.cells.moveBy(*d_pos.toTuple())
            if self.cells.x() < 0:
                self.cells.setX(0)
            if self.cells.x() > self.width() - self.grid_width:
                self.cells.setX(self.width() - self.grid_width)
            if self.cells.y() < 0:
                self.cells.setY(0)
            if self.cells.y() > self.height() - self.grid_height:
                self.cells.setY(self.height() - self.grid_height)


class LBFScene(GridScene):
    def __init__(self, width=400, height=250):
        super().__init__(width, height)
        self.agent_disp = LBFAgentDisplay(0, 0, CELL_SIZE, self.cells)
        self.items_disp = [LBFItemDisplay(0, 0, CELL_SIZE, self.cells)
                           for _ in range(NUM_ITEMS)]

    def update_state(self, agent_x, agent_y, items_x, items_y, item_exists):
        for item, x, y, exist in zip(self.items_disp, items_x, items_y, item_exists):
            item.update_pos(x, y)
            item.update_existence(exist)
        self.agent_disp.update_pos(agent_x, agent_y)


class LBFAgentDisplay(QGraphicsRectItem):
    def __init__(self, x, y, size, parent=None):
        super().__init__(parent)
        self.setBrush(Qt.red)
        self.setRect(0, 0, size, size)
        self.setPos(x * size, y * size)
        self.setZValue(1)
        self.size = size
    
    def update_pos(self, x, y):
        self.setPos(x * self.size, y * self.size)


class LBFItemDisplay(QGraphicsRectItem):
    def __init__(self, x, y, size, parent=None):
        super().__init__(parent)
        self.setBrush(Qt.blue)
        self.setRect(0, 0, size, size)
        self.setPos(x * size, y * size)
        self.size = size
    
    def update_pos(self, x, y):
        self.setPos(x * self.size, y * self.size)

    def update_existence(self, exist):
        self.setVisible(exist)


def run(agent):
    app = QApplication([])
    gui = Gui(EnvControl(agent, LBFScene(WIDTH, HEIGHT)))
    gui.show()
    app.exec()


if __name__ == '__main__':
    run()
