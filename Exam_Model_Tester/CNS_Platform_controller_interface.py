# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './/Interface\CNS_Platform_controller_interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(269, 383)
        self.Run = QtWidgets.QPushButton(Form)
        self.Run.setGeometry(QtCore.QRect(10, 10, 120, 23))
        self.Run.setObjectName("Run")
        self.Freeze = QtWidgets.QPushButton(Form)
        self.Freeze.setGeometry(QtCore.QRect(140, 10, 120, 23))
        self.Freeze.setObjectName("Freeze")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(10, 140, 251, 171))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(5, 5, 130, 16))
        self.label.setObjectName("label")
        self.Go_mal = QtWidgets.QPushButton(self.frame)
        self.Go_mal.setGeometry(QtCore.QRect(10, 140, 231, 23))
        self.Go_mal.setObjectName("Go_mal")
        self.Mal_list = QtWidgets.QListWidget(self.frame)
        self.Mal_list.setGeometry(QtCore.QRect(10, 80, 231, 51))
        self.Mal_list.setObjectName("Mal_list")
        self.Mal_nub = QtWidgets.QLineEdit(self.frame)
        self.Mal_nub.setGeometry(QtCore.QRect(70, 30, 50, 20))
        self.Mal_nub.setObjectName("Mal_nub")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 60, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(130, 30, 60, 20))
        self.label_3.setObjectName("label_3")
        self.Mal_type = QtWidgets.QLineEdit(self.frame)
        self.Mal_type.setGeometry(QtCore.QRect(190, 30, 50, 20))
        self.Mal_type.setObjectName("Mal_type")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(10, 50, 60, 20))
        self.label_4.setObjectName("label_4")
        self.Mal_time = QtWidgets.QLineEdit(self.frame)
        self.Mal_time.setGeometry(QtCore.QRect(70, 50, 50, 20))
        self.Mal_time.setObjectName("Mal_time")
        self.frame_2 = QtWidgets.QFrame(Form)
        self.frame_2.setGeometry(QtCore.QRect(10, 40, 251, 41))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.Initial = QtWidgets.QPushButton(self.frame_2)
        self.Initial.setGeometry(QtCore.QRect(10, 10, 75, 22))
        self.Initial.setObjectName("Initial")
        self.Initial_list = QtWidgets.QComboBox(self.frame_2)
        self.Initial_list.setGeometry(QtCore.QRect(90, 10, 151, 22))
        self.Initial_list.setObjectName("Initial_list")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Initial_list.addItem("")
        self.Go_db = QtWidgets.QPushButton(Form)
        self.Go_db.setGeometry(QtCore.QRect(20, 320, 231, 23))
        self.Go_db.setObjectName("Go_db")
        self.Show_main_win = QtWidgets.QPushButton(Form)
        self.Show_main_win.setGeometry(QtCore.QRect(20, 350, 231, 23))
        self.Show_main_win.setObjectName("Show_main_win")
        self.frame_3 = QtWidgets.QFrame(Form)
        self.frame_3.setGeometry(QtCore.QRect(10, 90, 251, 41))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_5 = QtWidgets.QLabel(self.frame_3)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 60, 20))
        self.label_5.setObjectName("label_5")
        self.Se_SP = QtWidgets.QLineEdit(self.frame_3)
        self.Se_SP.setGeometry(QtCore.QRect(120, 10, 61, 20))
        self.Se_SP.setObjectName("Se_SP")
        self.Apply_Sp = QtWidgets.QPushButton(self.frame_3)
        self.Apply_Sp.setGeometry(QtCore.QRect(190, 10, 51, 23))
        self.Apply_Sp.setObjectName("Apply_Sp")
        self.Cu_SP = QtWidgets.QLabel(self.frame_3)
        self.Cu_SP.setGeometry(QtCore.QRect(60, 10, 41, 20))
        self.Cu_SP.setObjectName("Cu_SP")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.Run.setText(_translate("Form", "Run"))
        self.Freeze.setText(_translate("Form", "Freeze"))
        self.label.setText(_translate("Form", "Malfunction"))
        self.Go_mal.setText(_translate("Form", "Go Malfunction"))
        self.label_2.setText(_translate("Form", "Mal_nub"))
        self.label_3.setText(_translate("Form", "Mal_type"))
        self.label_4.setText(_translate("Form", "Mal_time"))
        self.Initial.setText(_translate("Form", "(1) Initial"))
        self.Initial_list.setItemText(0, _translate("Form", "1. 100%"))
        self.Initial_list.setItemText(1, _translate("Form", "2. 80%"))
        self.Initial_list.setItemText(2, _translate("Form", "3. 70%"))
        self.Initial_list.setItemText(3, _translate("Form", "4. 60% (??)"))
        self.Initial_list.setItemText(4, _translate("Form", "5. 13% (??)"))
        self.Initial_list.setItemText(5, _translate("Form", "6. 7% (??)"))
        self.Initial_list.setItemText(6, _translate("Form", "7. 2% (??)"))
        self.Initial_list.setItemText(7, _translate("Form", "8. Reactor Critical"))
        self.Initial_list.setItemText(8, _translate("Form", "9. Hotstandby"))
        self.Initial_list.setItemText(9, _translate("Form", "10. Hotshutdown"))
        self.Initial_list.setItemText(10, _translate("Form", "11. Cold shutdown"))
        self.Initial_list.setItemText(11, _translate("Form", "12. Hot standby"))
        self.Initial_list.setItemText(12, _translate("Form", "13. Cold shutdown 1"))
        self.Initial_list.setItemText(13, _translate("Form", "14. Hot showdown(colddown)"))
        self.Initial_list.setItemText(14, _translate("Form", "15. Hot shut to Hot standby"))
        self.Initial_list.setItemText(15, _translate("Form", "16. Test 70%"))
        self.Initial_list.setItemText(16, _translate("Form", "17. 2%(real)"))
        self.Initial_list.setItemText(17, _translate("Form", "18. d"))
        self.Initial_list.setItemText(18, _translate("Form", "19. REAL2%"))
        self.Initial_list.setItemText(19, _translate("Form", "20. 2% to ALL"))
        self.Go_db.setText(_translate("Form", "Save_DB"))
        self.Show_main_win.setText(_translate("Form", "Show Interface"))
        self.label_5.setText(_translate("Form", "Speed"))
        self.Apply_Sp.setText(_translate("Form", "적용"))
        self.Cu_SP.setText(_translate("Form", "1"))