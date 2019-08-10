# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_loader.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(773, 662)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setObjectName("tabWidget")
        self.origin_view = QtWidgets.QWidget()
        self.origin_view.setObjectName("origin_view")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.origin_view)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.video_frame1 = QtWidgets.QLabel(self.origin_view)
        self.video_frame1.setObjectName("video_frame1")
        self.verticalLayout_2.addWidget(self.video_frame1)
        self.tabWidget.addTab(self.origin_view, "")
        self.processing_view = QtWidgets.QWidget()
        self.processing_view.setObjectName("processing_view")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.processing_view)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.video_frame2 = QtWidgets.QLabel(self.processing_view)
        self.video_frame2.setObjectName("video_frame2")
        self.verticalLayout_3.addWidget(self.video_frame2)
        self.tabWidget.addTab(self.processing_view, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalSlider = QtWidgets.QSlider(self.frame)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayout.addWidget(self.horizontalSlider)
        self.timeLabel = QtWidgets.QLabel(self.frame)
        self.timeLabel.setEnabled(True)
        self.timeLabel.setMinimumSize(QtCore.QSize(100, 18))
        self.timeLabel.setMaximumSize(QtCore.QSize(300, 18))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.timeLabel.setFont(font)
        self.timeLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.timeLabel.setInputMethodHints(QtCore.Qt.ImhTime)
        self.timeLabel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.timeLabel.setObjectName("timeLabel")
        self.verticalLayout.addWidget(self.timeLabel)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.startButton = QtWidgets.QPushButton(self.frame)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(self.frame)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 1, 1, 1)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.frame)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.bar_graph = QtWidgets.QWidget()
        self.bar_graph.setObjectName("bar_graph")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.bar_graph)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.bar_graph)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.bar_graph)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.tabWidget_2.addTab(self.bar_graph, "")
        self.line_graph = QtWidgets.QWidget()
        self.line_graph.setObjectName("line_graph")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.line_graph)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_4 = QtWidgets.QLabel(self.line_graph)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_5.addWidget(self.label_4)
        self.tabWidget_2.addTab(self.line_graph, "")
        self.gridLayout.addWidget(self.tabWidget_2, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Analysis_player"))
        self.video_frame1.setText(_translate("MainWindow", "origin_frame"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.origin_view), _translate("MainWindow", "origin"))
        self.video_frame2.setText(_translate("MainWindow", "processing_frame"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.processing_view), _translate("MainWindow", "processing"))
        self.timeLabel.setText(_translate("MainWindow", "run_time : / full_time :"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.label_2.setText(_translate("MainWindow", "team_A"))
        self.label_3.setText(_translate("MainWindow", "team_B"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.bar_graph), _translate("MainWindow", "team"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.line_graph), _translate("MainWindow", "player"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
