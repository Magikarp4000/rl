from imports import *
from gui import Gui, EnvScene


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
        for row in self.grid:
            for cell in row:
                self.addItem(cell)


if __name__ == '__main__':
    app = QApplication([])
    gui = Gui(GridScene(WIDTH, HEIGHT))
    gui.show()
    app.exec()
