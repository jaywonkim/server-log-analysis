from PyQt4 import QtGui, QtCore
import sys, os
import numpy
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

def main():
    app = QtGui.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())

class meh(QtGui.QLabel):
    def __init__(self):
        super(meh, self).__init__()
        fcPixmap = QtGui.QPixmap(os.getcwd() + '/fc.png')
        self.fcPixmapScaled = fcPixmap.scaled(500, 500, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(self.fcPixmapScaled)

class mainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.setWindowTitle("R E A L -  T I M E   A N A L Y S I S")
        self.navMenu = navMenu()
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.navMenu)
        self.majorScreen = majorScreen()
        self.setCentralWidget(self.majorScreen)
        self.navMenu.connect(self.navMenu.gta, QtCore.SIGNAL('triggered()'), self.changeTogta)
        self.navMenu.connect(self.navMenu.ids, QtCore.SIGNAL('triggered()'), self.changeToIDS)
        self.navMenu.connect(self.navMenu.fc, QtCore.SIGNAL('triggered()'), self.changeToFC)

        self.setFixedSize(1300, 700)
# Functions to change the page according the navigation menu

    def changeTogta(self):
        self.majorScreen.stacked.setCurrentIndex(0)

    def changeToIDS(self):
        self.majorScreen.stacked.setCurrentIndex(1)

    def changeToFC(self):
        self.majorScreen.stacked.setCurrentIndex(2)


class navMenu(QtGui.QToolBar):
    def __init__(self):
        super(navMenu, self).__init__()

        # General Traffic Analysis
        gtaPixmap = QtGui.QPixmap(os.getcwd() + '/gta.png')
        gtaIcon = QtGui.QIcon(gtaPixmap)
        self.gta = QtGui.QAction(self)
        self.gta.setIcon(gtaIcon)
        self.addAction(self.gta)

        # Intrusion Detection
        idsPixmap = QtGui.QPixmap(os.getcwd() + '/ids.png')
        idsIcon = QtGui.QIcon(idsPixmap)
        self.ids = QtGui.QAction(self)
        self.ids.setIcon(idsIcon)
        self.addAction(self.ids)

        # Forecasts
        fcPixmap = QtGui.QPixmap(os.getcwd() + '/fc.png')
        self.fcPixmapScaled = fcPixmap.scaledToHeight(1500, QtCore.Qt.SmoothTransformation)
        fcIcon = QtGui.QIcon(self.fcPixmapScaled)
        self.fc = QtGui.QAction(self)
        self.fc.setIcon(fcIcon)
        self.addAction(self.fc)

        for action in self.actions():
            widget = self.widgetForAction(action)
            widget.setFixedSize(100, 100)
        self.setOrientation(QtCore.Qt.Vertical)

def readName(fileName):
    data = numpy.genfromtxt(fileName, delimiter=',', dtype='|U12')
    return data

def readNumber(fileName):
    data = numpy.genfromtxt(fileName, delimiter=',')
    return data

# General Traffic Analysis Graphs page
class gtaScreen(QtGui.QWidget):
    def __init__(self):
        super(gtaScreen, self).__init__()
        self.title = QtGui.QLabel("G E N E R A L   T R A F F I C   A N A L Y S I S")
        self.title.setFixedSize(1200, 50)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        frequency_file = os.getcwd() + '/frequency.csv'
        ipAdd = readName(frequency_file)
        freq = readNumber(frequency_file)
        reply_file = os.getcwd() + '/replysize.csv'
        reply = readName(reply_file)
        self.fx = []
        self.fy = []
        self.rx = []
        self.ry = []
        self.flen = len(ipAdd)
        self.rlen = len(reply)
        for i in range(1, self.flen):
            self.fx.append(ipAdd[i][0])
            self.fy.append(freq[i][1])
        for i in range(1, self.rlen):
            self.rx.append(reply[i][0])
            self.ry.append(reply[i][1])

        # Timer which refreshes every 1 second
        self.timer = QtCore.QTimer(self)
        self.timer.connect(self.timer, QtCore.SIGNAL('timeout()'), self.updateIndices)
        self.timer.start(1000)

        self.fstart = 0
        self.fend = 10
        self.rstart = 0
        self.rend = 10
        self.fterm = self.flen - (self.flen%10)
        self.rterm = self.rlen - (self.rlen%10)
        self.mapOne = mapObject(self.fx[0:10], self.fy[0:10], 'bar', self.flen, "REQUEST FREQUENCY PER USER")
        self.mapTwo = mapObject(self.rx[0:10], self.ry[0:10], 'line', self.rlen, "SERVER REPLY SIZE")
        self.graphLayout = QtGui.QHBoxLayout()
        self.graphLayout.addWidget(self.mapOne)
        self.graphLayout.addWidget(self.mapTwo)
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.title)
        self.layout.addLayout(self.graphLayout)
        self.setLayout(self.layout)

    def updateIndices(self):
        if self.fstart < (self.fterm-10):
            self.fstart += 1
            self.fend += 1
        else:
            self.fstart = 0
            self.fend = 10

        if self.rstart < (self.rterm-10):
            self.rstart += 1
            self.rend += 1

        else:
            self.rstart = 0
            self.rend = 10

        self.mapOne.x = self.fx[self.fstart:self.fend]
        self.mapOne.y = self.fy[self.fstart:self.fend]
        self.mapTwo.x = self.rx[self.rstart:self.rend]
        self.mapTwo.y = self.ry[self.rstart:self.rend]
        self.mapOne.updateGraph()
        self.mapTwo.updateGraph()

# Intrusion Detection Graphs
class idsScreen(QtGui.QWidget):
    def __init__(self):
        super(idsScreen, self).__init__()
        self.title = QtGui.QLabel("I N T R U S I O N    D E T E C T I O N   ")
        self.title.setFixedSize(1200, 50)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        #self.mapOne = mapObject()
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.title)
        #self.layout.addWidget(self.mapOne)
        self.setLayout(self.layout)

# Forecasts Graphs
class fcScreen(QtGui.QWidget):
    def __init__(self):
        super(fcScreen, self).__init__()
        self.title = QtGui.QLabel("F O R E C A S T S  ")
        self.title.setFixedSize(1200, 50)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        #self.mapOne = mapObject()
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.title)
        #self.layout.addWidget(self.mapOne)
        self.setLayout(self.layout)

# Main window that will contain the graphs
class majorScreen(QtGui.QWidget):
    def __init__(self):
        super(majorScreen, self).__init__()
        self.stacked = QtGui.QStackedWidget()
        self.gta = gtaScreen()
        self.ids = idsScreen()
        self.fc = fcScreen()
        self.stacked.addWidget(self.gta)
        self.stacked.addWidget(self.ids)
        self.stacked.addWidget(self.fc)
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.stacked)
        self.setLayout(self.layout)

class mapObject(QtGui.QWidget):
    def __init__(self, x, y, type, len, title):
        super(mapObject, self).__init__()
        self.plotObject = QtGui.QWidget(self)
        self.plotObject.setGeometry(200, 200, 300, 300)
        self.dpi = 100
        self.x = x
        self.y = y
        self.type = type
        self.len = len
        self.title = QtGui.QLabel(title)
        self.title.setAlignment(QtCore.Qt.AlignCenter)

        self.figureOne = Figure((6.5, 6.5), dpi=self.dpi)
        self.canvasOne = FigureCanvas(self.figureOne)
        self.canvasOne.setParent(self)
        self.axesOne = self.figureOne.add_subplot(111)

        self.plotLayout = QtGui.QVBoxLayout()
        self.plotLayout.addWidget(self.canvasOne)
        self.plotObject.setLayout(self.plotLayout)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.plotObject)
        self.setLayout(self.layout)


    def updateGraph(self):
        if self.type == 'line':
            axis = numpy.arange(0, 10, 1)
            self.axesOne.cla()
            self.axesOne.plot(axis, self.y, 'R')
            self.axesOne.set_xticks(axis)
            self.axesOne.set_xticklabels(self.x)
            for tick in self.axesOne.get_xticklabels():
                tick.set_rotation(45)
                tick.set_fontsize(8)
            self.canvasOne.draw()

        elif self.type == 'bar':
            axis = numpy.arange(0, 10, 1)
            self.axesOne.cla()
            self.axesOne.bar(axis, self.y)
            self.axesOne.set_xticks(axis)
            self.axesOne.set_xticklabels(self.x)
            for tick in self.axesOne.get_xticklabels():
                tick.set_rotation(45)
                tick.set_fontsize(6.5)
            self.canvasOne.draw()

if __name__ == "__main__":
    main()