from imports import *
from gui import Gui, EnvScene

from utils import flatten


class GridSquare(QGraphicsRectItem):
    def __init__(self, x, y, size):
        super().__init__()
        self.setPen(QPen(Qt.white))
        self.setBrush(QBrush(Qt.black))
        self.setRect(x * size, y * size, size, size)


class GridScene(EnvScene):
    def __init__(self, width, height):
        super().__init__(width, height)
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


if __name__ == '__main__':
    app = QApplication([])
    gui = Gui(GridScene(WIDTH, HEIGHT))
    gui.show()
    app.exec()
