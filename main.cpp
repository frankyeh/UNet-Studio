#include "mainwindow.h"
#include "zlib.h"
#include "TIPL/tipl.hpp"

#include <QApplication>
#include <QSettings>
#include <QTextEdit>
QSettings settings(QSettings::IniFormat, QSettings::UserScope,"LabSolver", "UNet Studio");


console_stream console;

void console_stream::show_output(void)
{
    if(!tipl::is_main_thread<0>() || !log_window || !has_output)
        return;
    QStringList strSplitted;
    {
        std::lock_guard<std::mutex> lock(edit_buf);
        strSplitted = buf.split("\n");
        buf = strSplitted.back();
    }
    for(int i = 0; i+1 < strSplitted.size(); i++)
        log_window->append(strSplitted.at(i));
    QApplication::processEvents();
    has_output = false;
}
std::basic_streambuf<char>::int_type console_stream::overflow(std::basic_streambuf<char>::int_type v)
{
    {
        std::lock_guard<std::mutex> lock(edit_buf);
        buf.push_back(char(v));
    }

    if (v == '\n')
    {
        has_output = true;
        show_output();
    }
    return v;
}

std::streamsize console_stream::xsputn(const char *p, std::streamsize n)
{
    std::lock_guard<std::mutex> lock(edit_buf);
    buf += p;
    return n;
}




void check_cuda(std::string& error_msg);
int main(int argc, char *argv[])
{
    tipl::show_prog = true;
    console.attach();
    std::string msg;
    check_cuda(msg);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
