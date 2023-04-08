#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QSettings>
#include <QMovie>
#include <QMessageBox>
#include "TIPL/tipl.hpp"

extern QSettings settings;
extern std::vector<std::string> gpu_names;
void gen_list(std::vector<std::string>& network_list);
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),ui(new Ui::MainWindow)

{

    ui->setupUi(this);
    ui->input_dim_panel->hide();
    ui->tabWidget->setCurrentIndex(0);
    console.log_window = ui->console;
    console.show_output();

    ui->training->setMovie(new QMovie(":/icons/icons/processing.gif"));
    ui->evaluating->setMovie(new QMovie(":/icons/icons/processing.gif"));
    ui->view1->setScene(&train_scene1);
    ui->view2->setScene(&train_scene2);
    ui->eval_view1->setScene(&eval_scene1);
    ui->eval_view2->setScene(&eval_scene2);
    ui->error_view->setScene(&error_scene);
    v2c1.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    v2c1.set_range(0.0f,4.0f);
    v2c2.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    v2c2.set_range(0.0f,2.0f);
    eval_v2c1.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    eval_v2c1.set_range(0.0f,2.0f);
    eval_v2c2.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));
    eval_v2c2.set_range(0.0f,2.0f);



    QStringList device_list;
    device_list << "CPU";
    if (torch::cuda::is_available())
    {
        for(int i = 0;i < gpu_names.size();++i)
            device_list << gpu_names[i].c_str();
    }

    ui->gpu->addItems(device_list);
    ui->evaluate_device->addItems(device_list);
    ui->gpu->setCurrentIndex(torch::cuda::is_available() ? 1:0);
    ui->evaluate_device->setCurrentIndex(torch::cuda::is_available() ? 1:0);

    timer = new QTimer(this);
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(training()));

    eval_timer = new QTimer(this);
    eval_timer->setInterval(1000);
    connect(eval_timer, SIGNAL(timeout()), this, SLOT(evaluating()));
}

MainWindow::~MainWindow()
{

    delete ui;
}

void MainWindow::on_show_advanced_clicked(){ui->input_dim_panel->show();}
void MainWindow::on_eval_view_dim_currentIndexChanged(int index){   on_eval_pos_valueChanged(ui->eval_pos->value()); }
void MainWindow::on_view_dim_currentIndexChanged(int index){  on_pos_valueChanged(ui->pos->value()); }
void MainWindow::on_label_slider_valueChanged(int value){    on_pos_valueChanged(ui->pos->value());}
void MainWindow::on_eval_label_slider_valueChanged(int value){    on_eval_pos_valueChanged(ui->eval_pos->value());}
void MainWindow::on_error_x_size_valueChanged(int arg1){error_view_epoch = 0;}
void MainWindow::on_error_y_size_valueChanged(int arg1){error_view_epoch = 0;}



