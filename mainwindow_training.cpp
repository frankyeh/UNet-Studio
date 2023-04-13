#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QSettings>
#include <QMessageBox>
#include <QMovie>
#include "TIPL/tipl.hpp"
extern QSettings settings;

void MainWindow::on_actionOpen_Training_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,"Open Training Setting",settings.value("on_actionOpen_Training_triggered").toString(),"INI files (*.ini);;All files (*)"
    );
    if (fileName.isEmpty())
        return;
    settings.setValue("on_actionOpen_Training_triggered",fileName);

    QSettings ini(fileName,QSettings::IniFormat);
    image_list = ini.value("image",image_list).toStringList();
    label_list = ini.value("label",label_list).toStringList();
    update_list();
    ui->feature_string->setText(ini.value("feature_string",ui->feature_string->text()).toString());
    ui->epoch->setValue(ini.value("epoch",ui->epoch->value()).toInt());
    ui->batch_size->setValue(ini.value("batch_size",ui->batch_size->value()).toInt());
    ui->learning_rate->setValue(ini.value("learning_rate",ui->learning_rate->value()).toFloat());
    option->load(ini);
}

void MainWindow::on_actionSave_Training_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(
        this,"Save Training Setting",settings.value("on_actionSave_Training_triggered").toString(),"INI files (*.ini);;All files (*)"
    );
    if (fileName.isEmpty())
        return;
    settings.setValue("on_actionSave_Training_triggered",fileName);

    QSettings ini(fileName,QSettings::IniFormat);
    ini.setValue("image",image_list);
    ini.setValue("label",label_list);
    ini.setValue("feature_string",ui->feature_string->text());
    ini.setValue("epoch",ui->epoch->value());
    ini.setValue("batch_size",ui->batch_size->value());
    ini.setValue("learning_rate",ui->learning_rate->value());
    option->save(ini);
    ini.sync();

}

void MainWindow::update_list(void)
{
    auto index = ui->list1->currentRow();
    ui->list1->clear();
    ui->list2->clear();
    ready_to_train = false;
    for(size_t i = 0;i < image_list.size();++i)
    {
        if(!QFileInfo(label_list[i]).exists())
            label_list[i].clear();
        ui->list1->addItem(QFileInfo(image_list[i]).fileName());
        ui->list2->addItem(label_list[i].isEmpty() ? QString("(to be assigned)") : QFileInfo(label_list[i]).fileName());
        ready_to_train = true;
        ui->start_training->setEnabled(true);
    }
    if(index >=0 && index < ui->list1->count())
        ui->list1->setCurrentRow(index);
    else
        ui->list1->setCurrentRow(0);

    if(ui->list2->currentRow() == 0 && !label_list[0].isEmpty())
    {
        if(!get_label_info(label_list[0].toStdString(),out_count,is_label))
        {
            QMessageBox::critical(this,"Error",QString("%1 is not a valid label image").arg(QFileInfo(label_list[0]).fileName()));
            label_list[0] = QString();
            return;
        }
        ui->output_info->setText(QString("dim: %1 type: %2").arg(out_count).arg(is_label?"label":"scalar"));
        ui->label_slider->setMaximum(out_count-1);
    }
}

void MainWindow::on_open_files_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images",settings.value("on_open_files_clicked").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    settings.setValue("on_open_files_clicked",fileNames[0]);
    for(auto s : fileNames)
    {
        image_list << s;
        label_list << QString();
    }
    update_list();
    ui->open_labels->setEnabled(true);
}

void MainWindow::on_open_labels_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images",settings.value("on_open_labels_clicked").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    settings.setValue("on_open_labels_clicked",fileNames[0]);
    auto index = ui->list2->currentRow();
    for(int i = 0;i < fileNames.size() && index < label_list.size();++i,++index)
        label_list[index] = fileNames[i];

    update_list();

}


void MainWindow::on_autofill_clicked()
{
    if(ui->list2->currentRow() == -1)
        return;
    if(label_list[ui->list2->currentRow()].isEmpty())
    {
        QMessageBox::critical(this,"Error","At least assign the first label file");
        return;
    }
    for(int index = 0;index < label_list.size();++index)
        if(label_list[index].isEmpty())
        {
            std::string result;
            if(tipl::match_files(image_list[ui->list1->currentRow()].toStdString(),label_list[ui->list1->currentRow()].toStdString(),
                              image_list[index].toStdString(),result) && QFileInfo(result.c_str()).exists())
                label_list[index] = result.c_str();
        }
    update_list();
}

void MainWindow::on_clear_clicked()
{
    image_list.clear();
    label_list.clear();
    ui->list1->clear();
    ui->open_labels->setEnabled(false);
    ui->start_training->setEnabled(false);
    update_list();
}

void MainWindow::on_train_from_scratch_clicked()
{
    torch::manual_seed(0);
    train.model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
    QMessageBox::information(this,"","A new network loaded");
}
void MainWindow::on_load_network_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,"Open network file",settings.value("on_load_network_clicked").toString(),"Network files (*net.gz);;All files (*)");
    if(file.isEmpty() || !load_from_file(train.model,file.toStdString().c_str()))
        return;

    ui->feature_string->setText(train.model->feature_string.c_str());
    QMessageBox::information(this,"","Loaded");
    settings.setValue("on_load_network_clicked",file);
}
void MainWindow::on_save_network_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save network file",settings.value("on_save_network_clicked").toString(),"Network files (*net.gz);;All files (*)");
    if(!file.isEmpty() && save_to_file(train.model,file.toStdString().c_str()))
    {
        QMessageBox::information(this,"","Network Saved");
        settings.setValue("on_save_network_clicked",file);
    }

}
#include <ATen/Context.h>
void MainWindow::on_start_training_clicked()
{
    tipl::progress p("initiate training");
    torch::manual_seed(0);
    at::globalContext().setDeterministicCuDNN(true);
    qputenv("CUDNN_DETERMINISTIC", "1");
    bool from_scratch = false;

    train.param.epoch = ui->epoch->value();
    train.param.batch_size = ui->batch_size->value();
    train.param.learning_rate = ui->learning_rate->value();

    train.option = option;
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
    {
        train.model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
        from_scratch = true;
    }
    else
    if(out_count != train.model->out_count || ui->feature_string->text().toStdString() != train.model->feature_string)
    {
        tipl::out() << "copy pretrained model" << std::endl;
        auto new_model = UNet3d(1,out_count,ui->feature_string->text().toStdString());
        new_model->copy_from(*train.model.get(),torch::kCPU);
        train.model = new_model;
        from_scratch = true;
    }
    //tipl::out() << show_structure(train.model);

    train.param.dim = tipl::shape<3>(option->get<int>("dim_x"),
                                     option->get<int>("dim_y"),
                                     option->get<int>("dim_z"));
    train.param.device = ui->gpu->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->gpu->currentIndex()-1):torch::Device(torch::kCPU);
    train.param.from_scratch = from_scratch;

    train.param.image_file_name.clear();
    train.param.label_file_name.clear();
    for(size_t i = 0;i < image_list.size();++i)
    {
        train.param.image_file_name.push_back(image_list[i].toStdString());
        train.param.label_file_name.push_back(label_list[i].toStdString());
    }

    train.param.test_image_file_name.clear();
    train.param.test_label_file_name.clear();
    {
        train.param.test_image_file_name = train.param.image_file_name;
        train.param.test_label_file_name = train.param.label_file_name;
    }
    train.start();
    ui->train_prog->setMaximum(ui->epoch->value());
    ui->train_prog->setValue(1);
    timer->start();
    ui->start_training->setText(train.pause ? "Resume":"Pause");
    error_view_epoch = 0;
    error_scene << QImage();
}

void MainWindow::on_end_training_clicked()
{
    train.stop();
}

void MainWindow::training()
{
    console.show_output();
    ui->batch_size->setEnabled(!train.running || train.pause);
    ui->epoch->setEnabled(!train.running || train.pause);
    ui->learning_rate->setEnabled(!train.running || train.pause);

    ui->open_files->setEnabled(!train.running);
    ui->clear->setEnabled(!train.running);
    ui->open_labels->setEnabled(!train.running);
    ui->autofill->setEnabled(!train.running);
    ui->load_network->setEnabled(!train.running);
    ui->train_from_scratch->setEnabled(!train.running);
    ui->gpu->setEnabled(!train.running);
    ui->feature_string->setEnabled(!train.running);
    ui->save_error->setEnabled(train.cur_epoch);
    ui->training->setEnabled(train.running);

    if(!train.running)
        ui->training->movie()->stop();
    else
    {
        if(train.pause)
            ui->training->movie()->stop();
        else
            ui->training->movie()->start();
    }
    ui->train_prog->setValue(train.cur_epoch+1);
    ui->train_prog->setFormat(QString( "epoch: %1/%2 error: %3" ).arg(train.cur_epoch).arg(ui->train_prog->maximum()).arg(train.cur_epoch ? std::to_string(train.error[train.cur_epoch-1]).c_str():"pending"));
    ui->statusbar->showMessage(train.status.c_str());
    ui->end_training->setEnabled(train.running);

    if(train.cur_epoch > 1 && train.cur_epoch >= error_view_epoch)
    {
        auto x_scale = std::min<float>(5.0f,float(ui->error_x_size->value())/float(train.cur_epoch+1));
        size_t s = std::min<int>(train.cur_epoch,train.error.size());
        size_t s2 = std::min<int>(train.cur_epoch,train.test_error.size());
        size_t s3 = std::min<int>(loaded_error1.size(),(ui->error_x_size->value()-10)/x_scale);
        size_t s4 = std::min<int>(loaded_error2.size(),(ui->error_x_size->value()-10)/x_scale);

        QImage image(ui->error_x_size->value(),ui->error_y_size->value(),QImage::Format_RGB32);
        QPainter painter(&image);
        painter.fillRect(image.rect(), Qt::white);
        painter.setPen(QPen(Qt::black, 2));
        painter.drawRect(QRectF(5, 5, image.width() - 10, image.height() - 10));

        std::vector<float> y_value(train.error.begin(),train.error.begin()+s);
        if(s2)
            y_value.insert(y_value.end(),train.test_error.begin(),train.test_error.begin()+s2);
        if(s3)
            y_value.insert(y_value.end(),loaded_error1.begin(),loaded_error1.begin()+s3);
        if(s4)
            y_value.insert(y_value.end(),loaded_error2.begin(),loaded_error2.begin()+s4);
        if(y_value.empty())
            return;
        for(auto& v : y_value)
            v = -std::log10(v);
        tipl::normalize_upper_lower(y_value,(image.height()-10));

        auto y_value1 = std::vector<float>(y_value.begin(),y_value.begin()+s);
        auto y_value2 = std::vector<float>(y_value.begin()+s,y_value.begin()+s+s2);
        auto y_value3 = std::vector<float>(y_value.begin()+s+s2,y_value.begin()+s+s2+s3);
        auto y_value4 = std::vector<float>(y_value.begin()+s+s2+s3,y_value.end());

        QVector<QPointF> p1,p2,p3,p4;
        for(size_t i = 0;i < y_value1.size();++i)
            p1 << QPointF(i*x_scale+5,y_value1[i]+5);
        for(size_t i = 0;i < y_value2.size();++i)
            p2 << QPointF(i*x_scale+5,y_value2[i]+5);
        for(size_t i = 0;i < y_value3.size();++i)
            p3 << QPointF(i*x_scale+5,y_value3[i]+5);
        for(size_t i = 0;i < y_value4.size();++i)
            p4 << QPointF(i*x_scale+5,y_value4[i]+5);

        if(!p1.empty())
            painter.drawPolyline(p1);
        if(!p3.empty())
        {
            painter.setPen(QPen(Qt::black, 1));
            painter.drawPolyline(p3);
        }
        if(!p2.empty())
        {
            painter.setPen(QPen(Qt::red, 2));
            painter.drawPolyline(p2);
        }
        if(!p4.empty())
        {
            painter.setPen(QPen(Qt::red, 1));
            painter.drawPolyline(p4);
        }
        error_view_epoch = train.cur_epoch;
        error_scene << image;
    }


    if(!train.running)
    {
        timer->stop();
        if(!train.error_msg.empty())
            QMessageBox::critical(this,"Error",train.error_msg.c_str());
        ui->start_training->setText("Start");
        ui->train_prog->setValue(0);
    }
}


void fuzzy_labels(tipl::image<3>& label,const std::vector<size_t>& weights);
std::vector<size_t> get_label_count(const tipl::image<3>& label,size_t out_count);
void MainWindow::on_list1_currentRowChanged(int currentRow)
{
    if(ui->list2->currentRow() != currentRow)
        ui->list2->setCurrentRow(currentRow);
    auto pos_index = ui->pos->value();
    if(currentRow >= 0 && currentRow < image_list.size())
    {
        tipl::vector<3> vs;
        if(!read_image_and_label(image_list[currentRow].toStdString(),label_list[currentRow].toStdString(),I1,I2,vs))
            I2.clear();
        if(ui->show_transform->isChecked())
            load_image_and_label(*option,I1,I2,vs,tipl::shape<3>(option->get<int>("dim_x"),
                                                                 option->get<int>("dim_y"),
                                                                 option->get<int>("dim_z")),time(0));
        if(!I2.empty())
            fuzzy_labels(I2,get_label_count(I2,out_count));
        ui->pos->setMaximum(I1.shape()[ui->view_dim->currentIndex()]-1);
    }
    else
    {
        I1.clear();
        I2.clear();
        ui->pos->setMaximum(0);
    }
    ui->pos->setValue(pos_index);
    on_pos_valueChanged(ui->pos->value());
}

void MainWindow::on_list2_currentRowChanged(int currentRow)
{
    if(ui->list1->currentRow() != currentRow)
        ui->list1->setCurrentRow(currentRow);
}


void MainWindow::on_pos_valueChanged(int value)
{
    if(I1.empty())
        return ;
    train_scene1 << (QImage() << v2c1[tipl::volume2slice_scaled(I1,ui->view_dim->currentIndex(),value,2.0f)]);
    if(I2.size() == I1.size()*out_count)
        train_scene2 << (QImage() << v2c2[tipl::volume2slice_scaled(
                            I2.alias(I1.size()*ui->label_slider->value(),I1.shape()),
                            ui->view_dim->currentIndex(),value,2.0f)]);
    else
        train_scene2 << QImage();
}




void MainWindow::on_show_transform_clicked()
{
    on_list1_currentRowChanged(ui->list1->currentRow());
}

void MainWindow::on_save_error_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Error",settings.value("on_save_error_clicked").toString(),"Text values (*.txt);;All files (*)");
    if(file.isEmpty())
        return;
    std::ofstream out(file.toStdString());
    std::copy(train.error.begin(),train.error.begin()+train.cur_epoch,std::ostream_iterator<float>(out," "));
    if(train.cur_epoch)
    {
        out << std::endl;
        std::copy(train.test_error.begin(),train.test_error.begin()+train.cur_epoch,std::ostream_iterator<float>(out," "));
    }
    if(out.is_open())
        QMessageBox::information(this,"","Saved");
}

void MainWindow::on_open_error_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,"Open Error",settings.value("on_open_error_clicked").toString(),"Text values (*.txt);;All files (*)");
    if(file.isEmpty())
        return;
    std::ifstream in(file.toStdString());
    std::string line1,line2;
    std::getline(in,line1);
    std::getline(in,line2);
    std::istringstream in1(line1),in2(line2);
    loaded_error1 = std::vector<float>(std::istream_iterator<float>(in1),std::istream_iterator<float>());
    loaded_error2 = std::vector<float>(std::istream_iterator<float>(in2),std::istream_iterator<float>());
}


void MainWindow::on_clear_error_clicked()
{
    loaded_error1.clear();
    loaded_error2.clear();
}

