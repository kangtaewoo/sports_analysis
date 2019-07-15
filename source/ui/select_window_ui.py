# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'select_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SelectWindow(object):
    def setupUi(self, SelectWindow):
        SelectWindow.setObjectName("SelectWindow")
        SelectWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(SelectWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)
        SelectWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SelectWindow)
        QtCore.QMetaObject.connectSlotsByName(SelectWindow)

    def retranslateUi(self, SelectWindow):
        _translate = QtCore.QCoreApplication.translate
        SelectWindow.setWindowTitle(_translate("SelectWindow", "SelectWindow"))
        self.pushButton.setText(_translate("SelectWindow", "축구"))
        self.pushButton_2.setText(_translate("SelectWindow", "농구"))
        self.pushButton_3.setText(_translate("SelectWindow", "핸드볼"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SelectWindow = QtWidgets.QMainWindow()
    ui = Ui_SelectWindow()
    ui.setupUi(SelectWindow)
    SelectWindow.show()
    sys.exit(app.exec_())

