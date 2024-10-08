#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include "console.h"
#include "ui_console.h"
#include "zlib.h"
#include "TIPL/tipl.hpp"

console_stream console;

void console_stream::show_output(void)
{
    if(!tipl::is_main_thread() || !log_window || !has_output)
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

Console::Console(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Console)
{
    ui->setupUi(this);
    ui->pwd->setText(QString("[%1]$ ./unet_studio ").arg(QDir().current().absolutePath()));
    console.log_window = ui->console;
    console.show_output();

}

Console::~Console()
{
    console.log_window = nullptr;
    delete ui;
}

extern tipl::program_option<tipl::out> po;
int run_cmd(void);
void Console::on_run_cmd_clicked()
{
    if(ui->cmd_line->text().startsWith("unet_studio "))
        ui->cmd_line->setText(ui->cmd_line->text().remove("unet_studio "));
    if(!po.parse(ui->cmd_line->text().toStdString()))
    {
        QMessageBox::critical(this,"ERROR",po.error_msg.c_str());
        return;
    }
    if (!po.has("action"))
    {
        std::cout << "ERROR: invalid command" << std::endl;
        return;
    }
    run_cmd();
}


void Console::on_set_dir_clicked()
{
    QString dir =
        QFileDialog::getExistingDirectory(this,"Browse Directory","");
    if ( dir.isEmpty() )
        return;
    QDir::setCurrent(dir);
    ui->pwd->setText(QString("[%1]$ ./unet_studio ").arg(QDir().current().absolutePath()));

}

