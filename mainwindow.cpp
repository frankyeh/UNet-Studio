#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include "TIPL/tipl.hpp"
extern std::vector<std::string> gpu_names;
void gen_list(std::vector<std::string>& network_list);
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),ui(new Ui::MainWindow)

{

    ui->setupUi(this);
    ui->view1->setScene(&train_scene1);
    ui->view2->setScene(&train_scene2);
    ui->eval_view1->setScene(&eval_scene1);
    ui->eval_view2->setScene(&eval_scene2);

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

void MainWindow::on_train_from_scratch_clicked()
{
    train.model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
    QMessageBox::information(this,"","A new network loaded");
}
void MainWindow::on_load_network_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,"Open network file","","Network files (*net.gz);;All files (*)");
    if(file.isEmpty() || !load_from_file(train.model,file.toStdString().c_str()))
        return;
    ui->feature_string->setText(train.model->feature_string.c_str());
    ui->network->setText(QString("UNet %1->%2->%3").arg(train.model->in_count).arg(train.model->feature_string.c_str()).arg(train.model->out_count));
    QMessageBox::information(this,"","Loaded");
}
void MainWindow::on_save_network_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save network file","","Network files (*net.gz);;All files (*)");
    if(!file.isEmpty() && save_to_file(train.model,file.toStdString().c_str()))
        QMessageBox::information(this,"","Network Saved");
}
void MainWindow::on_start_training_clicked()
{
    if(train.running)
    {
        train.pause = !train.pause;
        ui->start_training->setText(train.pause ? "Resume":"Pause");
        ui->train_prog->setEnabled(!train.pause);
        return;
    }
    if(!ready_to_train)
    {
        QMessageBox::critical(this,"Error","Please specify training image and/or labels");
        return;
    }
    if(train.model->feature_string.empty())
        train.model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
    else
    if(out_count != train.model->out_count || ui->feature_string->text().toStdString() != train.model->feature_string)
    {
        tipl::out() << "ccopy pretrained model" << std::endl;
        auto new_model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
        new_model->copy_from(*train.model.get());
        train.model = new_model;
    }
    ui->network->setText(QString("UNet %1->%2->%3").arg(train.model->in_count).arg(train.model->feature_string.c_str()).arg(train.model->out_count));
    train.model->show_structure(std::cout);

    TrainParam param;
    param.batch_size = ui->batch_size->value();
    param.epoch = ui->epoch->value();
    param.learning_rate = ui->learning_rate->value();
    param.dim = tipl::shape<3>(ui->dim_x->value(),ui->dim_y->value(),ui->dim_z->value());
    param.device = ui->gpu->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->gpu->currentIndex()-1):torch::Device(torch::kCPU);
    for(size_t i = 0;i < image_list.size();++i)
    {
        param.image_file_name.push_back(image_list[i].toStdString());
        param.label_file_name.push_back(label_list[i].toStdString());
    }
    train.start(param);
    ui->train_prog->setMaximum(ui->epoch->value());
    ui->train_prog->setValue(1);
    timer->start();
    ui->start_training->setText(train.pause ? "Resume":"Pause");

}
void MainWindow::on_eval_from_train_clicked()
{
    if(train.model->feature_string.empty())
    {
        QMessageBox::critical(this,"Error","No trained network");
        return;
    }
    evaluate.model = UNet3d(train.model->in_count,train.model->out_count,train.model->feature_string);
    evaluate.model->train();
    evaluate.model->copy_from(*train.model.get());
    ui->evaluate_network->setText(QString("UNet %1->%2->%3").arg(evaluate.model->in_count).arg(evaluate.model->feature_string.c_str()).arg(evaluate.model->out_count));
    ui->evaluate->setEnabled(true);
    QMessageBox::information(this,"","Copied");
}


void MainWindow::on_eval_from_file_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,"Open Network File","","Network files (*net.gz);;All files (*)");
    if(file.isEmpty() || !load_from_file(evaluate.model,file.toStdString().c_str()))
        return;
    ui->evaluate_network->setText(QString("UNet %1->%2->%3").arg(evaluate.model->in_count).arg(evaluate.model->feature_string.c_str()).arg(evaluate.model->out_count));
    ui->evaluate->setEnabled(true);
    QMessageBox::information(this,"","Loaded");
}

void MainWindow::on_evaluate_clicked()
{
    if(evaluate.running)
    {
        evaluate.stop();
        eval_timer->stop();
        ui->evaluate->setText("Start");
        return;
    }
    EvaluateParam param;
    param.dim = tipl::shape<3>(ui->dim_x->value(),ui->dim_y->value(),ui->dim_z->value());
    param.device = ui->evaluate_device->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->evaluate_device->currentIndex()-1):torch::Device(torch::kCPU);
    for(auto s : evaluate_list)
        param.image_file_name.push_back(s.toStdString());

    ui->evaluate->setText("Stop");
    ui->eval_prog->setEnabled(true);
    ui->eval_prog->setMaximum(evaluate_list.size());
    ui->eval_label_slider->setMaximum(evaluate.model->out_count-1);
    ui->eval_label_slider->setValue(0);

    evaluate.start(param);
    eval_timer->start();

}

void MainWindow::training()
{
    ui->train_prog->setValue(train.cur_epoch+1);
    ui->train_prog->setFormat(QString( "epoch: %1/%2 error: %3" ).arg(train.cur_epoch).arg(ui->train_prog->maximum()).arg(train.cur_epoch ? std::to_string(train.error[train.cur_epoch-1]).c_str():"pending"));
    if(!train.running)
    {
        timer->stop();
        if(!train.error_msg.empty())
            QMessageBox::critical(this,"Error",train.error_msg.c_str());
        ui->start_training->setText("Start");
        ui->train_prog->setValue(0);
    }
}
void MainWindow::evaluating()
{
    ui->eval_prog->setValue(evaluate.cur_output);
    ui->train_prog->setFormat(QString("%1/%2").arg(evaluate.cur_output).arg(evaluate_list.size()));
    if(!evaluate.running)
    {
        eval_timer->stop();
        if(!evaluate.error_msg.empty())
            QMessageBox::critical(this,"Error",evaluate.error_msg.c_str());
        ui->evaluate->setText("Start");
    }
}
void MainWindow::on_end_training_clicked()
{
    train.stop();
}
void MainWindow::update_list(void)
{
    auto index = ui->list1->currentRow();
    ui->list1->clear();
    ready_to_train = false;
    for(size_t i = 0;i < image_list.size();++i)
    {
        if(!QFileInfo(label_list[i]).exists())
            label_list[i].clear();
        if(label_list[i].isEmpty())
            ui->list1->addItem(QFileInfo(image_list[i]).fileName() + " (label to be assigned)");
        else
        {
            ui->list1->addItem(QFileInfo(image_list[i]).fileName() + "->" + QFileInfo(label_list[i]).fileName());
            ready_to_train = true;
            ui->start_training->setEnabled(true);
        }
    }
    if(index >=0 && index < ui->list1->count())
        ui->list1->setCurrentRow(index);
    else
        ui->list1->setCurrentRow(0);
}
void MainWindow::on_open_files_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images","","NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    for(auto s : fileNames)
    {
        image_list << s;
        label_list << QString();
    }
    update_list();
    ui->open_labels->setEnabled(true);
}

void MainWindow::on_clear_clicked()
{
    image_list.clear();
    label_list.clear();
    ui->list1->clear();
    ui->open_labels->setEnabled(false);
    update_list();
}


void MainWindow::on_evaluate_clear_clicked()
{
    evaluate_list.clear();
    ui->evaluate_list->clear();
}



void MainWindow::on_open_labels_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images","","NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    auto index = ui->list1->currentRow();
    for(int i = 0;i < fileNames.size() && index < label_list.size();++i,++index)
        label_list[index] = fileNames[i];


    if(fileNames.size() == 1 && image_list.size() > 1)
    {
        auto ans = QMessageBox::warning(this, "", "Apply to other labels?",QMessageBox::Yes | QMessageBox::No,QMessageBox::Yes);
        if(ans == QMessageBox::Cancel)
            return;
        for(;index < label_list.size();++index)
        {
            std::string result;
            if(tipl::match_files(image_list[ui->list1->currentRow()].toStdString(),label_list[ui->list1->currentRow()].toStdString(),
                              image_list[index].toStdString(),result) && QFileInfo(result.c_str()).exists())
                label_list[index] = result.c_str();
        }
    }
    update_list();

    if(!(out_count = get_label_out_count(label_list[0].toStdString())))
    {
        out_count = 1;
        QMessageBox::critical(this,"Error","Not a valid label image");
        return;
    }
    ui->label_slider->setMaximum(out_count-1);
}


void MainWindow::on_list1_currentRowChanged(int currentRow)
{
    auto pos_index = ui->pos->value();
    if(currentRow >= 0 && currentRow < image_list.size())
    {
        tipl::vector<3> vs;
        ui->image_name->setText(image_list[currentRow]);
        ui->label_name->setText(label_list[currentRow]);
        if(!read_image_and_label(image_list[currentRow].toStdString(),label_list[currentRow].toStdString(),I1,I2,out_count,vs))
            I2.clear();
        load_image_and_label(I1,I2,vs,tipl::shape<3>(ui->dim_x->value(),ui->dim_y->value(),ui->dim_z->value()));
        v2c1.set_range(0,tipl::max_value_mt(I1));
        ui->pos->setMaximum(I1.shape()[ui->view_dim->currentIndex()]-1);
    }
    else
    {
        I1.clear();
        I2.clear();
        ui->image_name->setText(QString());
        ui->label_name->setText(QString());
        ui->pos->setMaximum(0);
    }
    ui->pos->setValue(pos_index);
    on_pos_valueChanged(ui->pos->value());
}



void MainWindow::on_evaluate_list_currentRowChanged(int currentRow)
{
    auto pos_index = ui->eval_pos->value();
    if(currentRow >= 0 && currentRow < evaluate_list.size())
    {
        tipl::vector<3> vs;
        if(!tipl::io::gz_nifti::load_from_file(evaluate_list[currentRow].toStdString().c_str(),eval_I1,vs))
            return;
        if(currentRow < evaluate.cur_output)
            eval_I2 = evaluate.evaluate_output[currentRow];
        else
        {
            eval_I2.resize(eval_I1.shape().multiply(tipl::shape<3>::z,evaluate.model->out_count));
            eval_I2 = 0;
        }
        eval_v2c1.set_range(0,tipl::max_value_mt(eval_I1));
        ui->eval_pos->setMaximum(eval_I1.shape()[ui->eval_view_dim->currentIndex()]-1);
    }
    else
    {
        eval_I1.clear();
        eval_I2.clear();
        ui->eval_pos->setMaximum(0);
    }
    ui->eval_pos->setValue(pos_index);
    on_eval_pos_valueChanged(ui->eval_pos->value());
}

void MainWindow::on_pos_valueChanged(int value)
{
    if(I1.empty())
        return ;
    train_scene1 << (QImage() << v2c1[tipl::volume2slice_scaled(I1,ui->view_dim->currentIndex(),value,2.0f)]);
    if(I2.size() == I1.size()*out_count)
        train_scene2 << (QImage() << v2c2[tipl::volume2slice_scaled(
                            I2.sub_image(I1.size()*ui->label_slider->value(),I1.shape()),
                            ui->view_dim->currentIndex(),value,2.0f)]);
    else
        train_scene2 << QImage();
}


void MainWindow::on_eval_pos_valueChanged(int value)
{
    if(eval_I1.empty())
        return;


    eval_scene1 << (QImage() << eval_v2c1[tipl::volume2slice_scaled(eval_I1,ui->eval_view_dim->currentIndex(),value,2.0f)]);

    if(eval_I2.size() == eval_I1.size()*evaluate.model->out_count)
        eval_scene2 << (QImage() << eval_v2c2[tipl::volume2slice_scaled(
                           eval_I2.sub_image(eval_I1.size()*ui->eval_label_slider->value(),eval_I1.shape()),
                           ui->eval_view_dim->currentIndex(),value,2.0f)]);
    else
        eval_scene2 << QImage();

}

void MainWindow::update_evaluate_list(void)
{
    auto index = ui->evaluate_list->currentRow();
    ui->evaluate_list->clear();
    for(auto s: evaluate_list)
        ui->evaluate_list->addItem(QFileInfo(s).fileName());
    if(index >=0 && index < ui->evaluate_list->count())
        ui->evaluate_list->setCurrentRow(index);
    else
        ui->evaluate_list->setCurrentRow(0);
}
void MainWindow::on_open_evale_image_clicked()
{
    QStringList file = QFileDialog::getOpenFileNames(this,"Open Image","","NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    evaluate_list << file;
    update_evaluate_list();
}


void MainWindow::on_eval_view_dim_currentIndexChanged(int index){    }
void MainWindow::on_view_dim_currentIndexChanged(int index){   }
void MainWindow::on_label_slider_valueChanged(int value){    on_pos_valueChanged(ui->pos->value());}
void MainWindow::on_eval_label_slider_valueChanged(int value){    on_eval_pos_valueChanged(ui->eval_pos->value());}



















