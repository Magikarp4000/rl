import random

import numpy as np

from imports import *
from gui import Gui, EnvScene

from utils import flatten


class GridSquare(QGraphicsRectItem):
    def __init__(self, x, y, size):
        super().__init__()
        self.setPen(QPen(Qt.white))
        self.setBrush(Qt.black)
        self.setRect(x * size, y * size, size, size)


class GridScene(EnvScene):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.grid_width = CELL_SIZE * NUM_CELLS_X
        self.grid_height = CELL_SIZE * NUM_CELLS_Y
        self.grid = [[GridSquare(x, y, CELL_SIZE) for x in range(NUM_CELLS_X)]
                     for y in range(NUM_CELLS_Y)]
        self.cells = self.createItemGroup(flatten(self.grid))
        self.agent = AgentView(2, 3, CELL_SIZE, self.cells)
    
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


class AgentView(QGraphicsRectItem):
    def __init__(self, x, y, size, parent=None):
        super().__init__(parent)
        self.setBrush(Qt.red)
        self.setRect(0, 0, size, size)
        self.setPos(x * size, y * size)
        
        self.xpos = x
        self.ypos = y
        self.size = size

        self.timer = QTimer()
        self.timer.timeout.connect(self.move)
        self.timer.start(1000 / FPS)
    
    def move(self):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        self.xpos = np.clip(self.xpos + dx, 0, NUM_CELLS_X - 1)
        self.ypos = np.clip(self.ypos + dy, 0, NUM_CELLS_Y - 1)
        self.setPos(self.xpos * self.size, self.ypos * self.size)


if __name__ == '__main__':
    app = QApplication([])
    gui = Gui(GridScene(WIDTH, HEIGHT))
    gui.show()
    app.exec()
