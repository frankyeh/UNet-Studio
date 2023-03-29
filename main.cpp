#include "mainwindow.h"
#include "zlib.h"
#include "TIPL/tipl.hpp"

#include <QApplication>

void check_cuda(std::string& error_msg);
int main(int argc, char *argv[])
{
    tipl::show_prog = true;
    std::string msg;
    check_cuda(msg);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
