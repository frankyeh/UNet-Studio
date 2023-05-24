#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QSettings>
#include <QMovie>
#include <QMessageBox>
#include "console.h"
#include "TIPL/tipl.hpp"

extern QSettings settings;
extern std::vector<std::string> gpu_names;
void gen_list(std::vector<std::string>& network_list);
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),ui(new Ui::MainWindow)

{
    ui->setupUi(this);
    ui->option_widget_layout->addWidget(option = new OptionTableWidget(*this,ui->option_widget,":/options.txt"));
    ui->postproc_widget_layout->addWidget(eval_option = new OptionTableWidget(*this,ui->postproc_widget,":/postproc.txt"));
    connect(eval_option,SIGNAL(run_action(QString)),this,SLOT(run_action(QString)));
    ui->postproc_widget->hide();
    ui->option_widget->hide();

    ui->tabWidget->setCurrentIndex(0);
    ui->training_gif->setMovie(new QMovie(":/icons/icons/processing.gif"));
    ui->evaluate_gif->setMovie(new QMovie(":/icons/icons/processing.gif"));
    ui->eval_label_slider->setVisible(false);
    ui->view1->setScene(&train_scene1);
    ui->view2->setScene(&train_scene2);
    ui->eval_view1->setScene(&eval_scene1);
    ui->eval_view2->setScene(&eval_scene2);
    ui->error_view->setScene(&error_scene);
    v2c1.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    v2c1.set_range(0.0f,1.0f);
    v2c2.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    v2c2.set_range(0.0f,1.0f);
    eval_v2c1.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    eval_v2c1.set_range(0.0f,1.0f);
    eval_v2c2.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    eval_v2c2.set_range(0.0f,1.0f);

    ui->eval_contrast->hide();

    // populate device list
    {
        QStringList device_list;
        device_list << "CPU";
        torch::set_num_threads(std::thread::hardware_concurrency());
        if (torch::cuda::is_available())
        {
            for(int i = 0;i < gpu_names.size();++i)
                device_list << gpu_names[i].c_str();
        }

        ui->train_device->addItems(device_list);
        ui->evaluate_device->addItems(device_list);
        ui->train_device->setCurrentIndex(torch::cuda::is_available() ? 1:0);
        ui->evaluate_device->setCurrentIndex(ui->evaluate_device->count()-1);
    }
    // populate networks
    {
        ui->evaluate_builtin_networks->addItem("select or open...");
        QDir dir(QCoreApplication::applicationDirPath() + "/network");
        dir.setNameFilters(QStringList() << "*.net.gz");
        QFileInfoList files = dir.entryInfoList(QDir::Files);
        for (const QFileInfo& fileInfo : files)
            ui->evaluate_builtin_networks->addItem(fileInfo.fileName().remove(".net.gz"));
        ui->evaluate_builtin_networks->setCurrentText(settings.value("eval_network").toString());
    }

    timer = new QTimer(this);
    timer->setInterval(1500);
    connect(timer, SIGNAL(timeout()), this, SLOT(training()));

    eval_timer = new QTimer(this);
    eval_timer->setInterval(1500);
    connect(eval_timer, SIGNAL(timeout()), this, SLOT(evaluating()));

}

MainWindow::~MainWindow()
{
    settings.setValue("eval_network",ui->evaluate_builtin_networks->currentText());
    delete ui;
}

void MainWindow::on_eval_view_dim_currentIndexChanged(int index)
{
    auto currentRow = ui->evaluate_list->currentRow();
    if(currentRow >= 0 && currentRow < eval_I1_buffer.size() && !eval_I1_buffer[currentRow].empty())
    {
        ui->eval_pos->setMaximum(eval_I1_buffer[currentRow].shape()[index]-1);
        ui->eval_pos->setValue(ui->eval_pos->maximum()/2);
    }
}
void MainWindow::on_view_dim_currentIndexChanged(int index)
{
    ui->pos->setMaximum(I1.shape()[index]-1);
    ui->pos->setValue(ui->pos->maximum()/2);
}
void MainWindow::on_label_slider_valueChanged(int value)
{
    on_pos_valueChanged(ui->pos->value());
}
void MainWindow::on_eval_label_slider_valueChanged(int value)
{
    on_eval_pos_valueChanged(ui->eval_pos->value());
}
void MainWindow::on_error_x_size_valueChanged(int arg1){plot_error();}
void MainWindow::on_error_y_size_valueChanged(int arg1){plot_error();}

void MainWindow::on_actionConsole_triggered()
{
    static auto* con = new Console(this);
    con->showNormal();
}














