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
        self.grid = [[GridSquare(x, y, CELL_SIZE) for x in range(NUM_CELLS_X)]
                     for y in range(NUM_CELLS_Y)]
        self.cells = self.createItemGroup(flatten(self.grid))
        self.mouse_pos = None
    
    def mouseMoveEvent(self, event):
        new_pos = event.scenePos()
        if self.mouse_pos != None and Qt.MouseButton.LeftButton in event.buttons():
            dx = (new_pos - self.mouse_pos).x()
            dy = (new_pos - self.mouse_pos).y()
            self.cells.moveBy(dx, dy)
        self.mouse_pos = new_pos


if __name__ == '__main__':
    app = QApplication([])
    gui = Gui(GridScene(WIDTH, HEIGHT))
    gui.show()
    app.exec()
