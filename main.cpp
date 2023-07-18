#include "mainwindow.h"
#include "zlib.h"
#include "TIPL/tipl.hpp"

#include <QApplication>
#include <QSettings>
#include <QTextEdit>
#include "console.h"
QSettings settings("settings.ini",QSettings::IniFormat);
extern console_stream console;
void check_cuda(std::string& error_msg);

int run_action_with_wildcard(tipl::program_option<tipl::out>& po)
{
    return 1;
}
int main(int argc, char *argv[])
{
    tipl::show_prog = true;
    console.attach();
    std::string msg;

    if constexpr (tipl::use_cuda)
        check_cuda(msg);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
