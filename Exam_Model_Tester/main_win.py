#
import sys
import multiprocessing
from copy import deepcopy
import time
import json

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class MainWindow(QWidget):
    """메인 윈도우"""
    def __init__(self, parnet=None, mem=None):
        super(MainWindow, self).__init__()
        self.top_window = parnet
        self.mem = mem
        # --------------------------------------------------------------------------------------------------------------
        self.setGeometry(300, 300, 1000, 900)    # initial window size
        self.setObjectName('MainWin')
        # 요소 선언 -----------------------------------------------------------------------------------------------------
        # 1] 그래프
        self.fig = plt.Figure(tight_layout=True)
        gs = GridSpec(8, 3, figure=self.fig)

        self.axs = [
            self.fig.add_subplot(gs[0:2, 0:1]),
            self.fig.add_subplot(gs[2:4, 0:1]),
            self.fig.add_subplot(gs[4:6, 0:1]),
            self.fig.add_subplot(gs[6:8, 0:1]),

            self.fig.add_subplot(gs[0:2, 1:2]),
            self.fig.add_subplot(gs[2:4, 1:2]),
            self.fig.add_subplot(gs[4:6, 1:2]),
            self.fig.add_subplot(gs[6:8, 1:2]),

            self.fig.add_subplot(gs[0:2, 2:3]),
            self.fig.add_subplot(gs[2:4, 2:3]),
            self.fig.add_subplot(gs[4:6, 2:3]),
            self.fig.add_subplot(gs[6:8, 2:3]),
        ]
        self.title_pack = [
            # 'CH', 'SH', 'S1', 'DH',
            # 'SY2', 'SJ2', 'CJ', '',
            'SY', 'S1', 'DH', 'SH',
            'CH', 'CJ', 'SJ', '',
            '', '', '', '',
        ]
        self.fig.canvas.draw()
        self.canvas = FigureCanvasQTAgg(self.fig)

        # Main 프레임 모양 정의
        window_vbox = QVBoxLayout()
        window_vbox.addWidget(self.canvas)

        self.setLayout(window_vbox)
        self.setContentsMargins(0, 0, 0, 0)

        if self.mem != None:
            # timer section
            timer = QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._update)
            timer.start()

    def _update(self):
        m = self.mem.get_model_out()

        if m != {}:
            for i in range(len(m)):
                self.axs[i].clear()

                getx = m[i]
                gety = [i for i in range(len(getx))]
                self.axs[i].plot(gety, getx)
                self.axs[i].legend(['Normal', 'LOCA', 'SGTR', 'MSLB', 'MSLB-non'], loc=1, fontsize=7)
                self.axs[i].grid()
                self.axs[i].set_title(self.title_pack[i])
        self.canvas.draw()


if __name__ == '__main__':
    # Board_Tester
    app = QApplication(sys.argv)
    window = MainWindow(None, None)
    window.show()
    app.exec_()