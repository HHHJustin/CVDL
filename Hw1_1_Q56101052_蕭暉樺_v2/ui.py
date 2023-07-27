# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

def show():
    print("123")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(821, 457)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 40, 221, 191))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(40, 30, 131, 31))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 80, 131, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 130, 131, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(270, 40, 231, 351))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 40, 131, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(40, 90, 131, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 130, 211, 101))
        self.groupBox_3.setObjectName("groupBox_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(70, 30, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems([str(1),str(2),str(3),str(4),str(5),str(6),str(7),str(8),str(9),
                               str(10),str(11),str(12),str(13),str(14),str(15)])
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(30, 60, 131, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(40, 250, 131, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(40, 300, 131, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(510, 40, 211, 181))
        self.groupBox_4.setObjectName("groupBox_4")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit.setGeometry(QtCore.QRect(40, 40, 131, 31))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_9.setGeometry(QtCore.QRect(30, 90, 161, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 130, 161, 31))
        self.pushButton_10.setObjectName("pushButton_10")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(510, 240, 221, 151))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 70, 161, 31))
        self.pushButton_11.setObjectName("pushButton_11")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 821, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.pushButton.setText(_translate("MainWindow", "Load Folder"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Image_L"))
        self.pushButton_3.setText(_translate("MainWindow", "Load Image_R"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.pushButton_4.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.pushButton_5.setText(_translate("MainWindow", "1.2 Find Instrinsic"))
        self.groupBox_3.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_8.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_6.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.pushButton_7.setText(_translate("MainWindow", "1.5 Show Result"))
        self.groupBox_4.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.pushButton_9.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.pushButton_10.setText(_translate("MainWindow", "2.2 Show Words Vertically"))
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.pushButton_11.setText(_translate("MainWindow", "3.1 Sterro Disparity Map"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

