#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QMessageBox>
#include <QMovie>
#include "TIPL/tipl.hpp"

extern QSettings settings;

void MainWindow::on_show_advanced_clicked()
{
    if(ui->option_widget->isVisible())
        ui->option_widget->hide();
    else
        ui->option_widget->show();
}

void MainWindow::on_actionOpen_Training_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,"Open Training Setting",settings.value("work_dir").toString() + "/" +
                                     settings.value("work_file").toString() + ".ini","INI files (*.ini);;All files (*)"
    );
    if (fileName.isEmpty())
        return;

    QSettings ini(fileName,QSettings::IniFormat);
    image_list = ini.value("image",image_list).toStringList();
    label_list = ini.value("label",label_list).toStringList();
    update_list();
    ui->epoch->setValue(ini.value("epoch",ui->epoch->value()).toInt());
    ui->batch_size->setValue(ini.value("batch_size",ui->batch_size->value()).toInt());
    ui->learning_rate->setValue(ini.value("learning_rate",ui->learning_rate->value()).toFloat());
    option->load(ini);

    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",QFileInfo(fileName.remove(".ini")).fileName());

}

void MainWindow::on_actionSave_Training_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(
        this,"Save Training Setting",settings.value("work_dir").toString() + "/" +
                                     settings.value("work_file").toString() + ".ini","INI files (*.ini);;All files (*)"
    );
    if (fileName.isEmpty())
        return;

    QSettings ini(fileName,QSettings::IniFormat);
    ini.setValue("image",image_list);
    ini.setValue("label",label_list);
    ini.setValue("epoch",ui->epoch->value());
    ini.setValue("batch_size",ui->batch_size->value());
    ini.setValue("learning_rate",ui->learning_rate->value());
    option->save(ini);
    ini.sync();

    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",QFileInfo(fileName.remove(".ini")).fileName());

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
    on_list1_currentRowChanged(ui->list1->currentRow());
}

void MainWindow::on_open_files_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    settings.setValue("work_dir",QFileInfo(fileNames[0]).absolutePath());

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
        this,"Select NIFTI images",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    settings.setValue("work_dir",QFileInfo(fileNames[0]).absolutePath());
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
    auto feature = QInputDialog::getText(this,"","Please Specify Network Structure",QLineEdit::Normal,"8x8+16x16+32x32+64x64+128x128");
    if(feature.isEmpty())
        return;
    torch::manual_seed(0);
    train.model = UNet3d(1,out_count,feature.toStdString());
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    ui->batch_size->setValue(1);
}
void MainWindow::on_load_network_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open network file",
                                                settings.value("work_dir").toString() + "/" +
                                                settings.value("work_file").toString() + ".net.gz","Network files (*net.gz);;All files (*)");
    if(fileName.isEmpty())
        return;

    if(!load_from_file(train.model,fileName.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error","Invalid file format");
        return;
    }
    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",train_name = QFileInfo(fileName.remove(".net.gz")).fileName());

    ui->batch_size->setValue(std::pow(4,std::log10(train.model->total_training_count)));
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());


}
void MainWindow::on_save_network_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save network file",
                                                settings.value("work_dir").toString() + "/" +
                                                train_name + ".net.gz","Network files (*net.gz);;All files (*)");
    if(!fileName.isEmpty() && save_to_file(train.model,fileName.toStdString().c_str()))
    {
        QMessageBox::information(this,"","Network Saved");
        settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
        settings.setValue("work_file",train_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    }

}
#include <ATen/Context.h>
void MainWindow::on_start_training_clicked()
{
    tipl::progress p("initiate training");
    torch::manual_seed(0);
    at::globalContext().setDeterministicCuDNN(true);
    qputenv("CUDNN_DETERMINISTIC", "1");

    train.param.epoch = ui->epoch->value();
    train.param.batch_size = ui->batch_size->value();
    train.param.learning_rate = ui->learning_rate->value();

    train.option = option;
    if(train.running)
    {
        train.pause = !train.pause;
        return;
    }
    if(!ready_to_train)
    {
        QMessageBox::critical(this,"Error","Please specify training image and/or labels");
        return;
    }
    if(train.model->feature_string.empty())
    {
        on_train_from_scratch_clicked();
        if(train.model->feature_string.empty())
            return;
    }

    if(out_count != train.model->out_count)
    {
        tipl::out() << "copy pretrained model" << std::endl;
        auto new_model = UNet3d(1,out_count,train.model->feature_string);
        new_model->copy_from(*train.model.get(),torch::kCPU);
        train.model = new_model;
    }

    //tipl::out() << show_structure(train.model);

    train.model->dim = tipl::shape<3>(option->get<int>("dim_x"),
                                     option->get<int>("dim_y"),
                                     option->get<int>("dim_z"));
    train.param.device = ui->gpu->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->gpu->currentIndex()-1):torch::Device(torch::kCPU);
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

void MainWindow::plot_error()
{
    if(train.cur_epoch > 1)
    {
        size_t x_size = ui->error_x_size->value();
        size_t y_size = ui->error_y_size->value();
        auto x_scale = std::min<float>(5.0f,float(x_size)/float(train.cur_epoch+1));
        size_t s = std::min<int>(train.cur_epoch,train.error.size());
        size_t s2 = std::min<int>(train.cur_epoch,train.test_error.size());
        size_t s3 = std::min<int>(loaded_error1.size(),float(x_size)/x_scale);
        size_t s4 = std::min<int>(loaded_error2.size(),float(x_size)/x_scale);

        QImage image(x_size+10,y_size+10,QImage::Format_RGB32);
        QPainter painter(&image);
        painter.fillRect(image.rect(), Qt::white);
        painter.setPen(QPen(Qt::black, 2));
        painter.drawRect(QRectF(5, 5, x_size,y_size));

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
            p1 << QPointF(float(i)*x_scale+5,y_value1[i]+5);

        for(size_t i = 0;i < y_value2.size();++i)
            p2 << QPointF(float(i)*x_scale+5,y_value2[i]+5);
        for(size_t i = 0;i < y_value3.size();++i)
            p3 << QPointF(float(i)*x_scale+5,y_value3[i]+5);
        for(size_t i = 0;i < y_value4.size();++i)
            p4 << QPointF(float(i)*x_scale+5,y_value4[i]+5);

        if(!p1.empty())
        {
            painter.setPen(QPen(Qt::black, 2));
            painter.drawPolyline(p1);
            painter.setPen(QPen(Qt::black, 1));
            painter.drawLine(5,p1.back().y(),p1.back().x(),p1.back().y());
        }
        if(!p3.empty())
        {
            painter.setPen(QPen(Qt::black, 1));
            painter.drawPolyline(p3);
        }
        if(!p2.empty())
        {
            painter.setPen(QPen(Qt::red, 2));
            painter.drawPolyline(p2);
            painter.setPen(QPen(Qt::red, 1));
            painter.drawLine(5,p2.back().y(),p2.back().x(),p2.back().y());
        }
        if(!p4.empty())
        {
            painter.setPen(QPen(Qt::red, 1));
            painter.drawPolyline(p4);
        }
        error_view_epoch = train.cur_epoch;
        error_scene << image;
    }

}

void MainWindow::training()
{
    if(!train.running)
        timer->stop();

    if(!train.error_msg.empty())
    {
        QMessageBox::critical(this,"Error",train.error_msg.c_str());
        train.error_msg.clear();
    }

    ui->start_training->setText(train.running ? (train.pause ? "Resume":"Pause") : "Start");
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
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
    ui->save_error->setEnabled(!train.error.empty());
    ui->training->setEnabled(train.running);

    if(!train.running || train.pause)
        ui->training->movie()->stop();
    else
        ui->training->movie()->start();

    if(train.running)
    {
        ui->train_prog->setEnabled(!train.pause);
        ui->train_prog->setValue(train.cur_epoch+1);
        ui->train_prog->setFormat(QString( "epoch: %1/%2 error: %3" ).arg(train.cur_epoch).arg(ui->train_prog->maximum()).arg(train.cur_epoch ? std::to_string(train.error[train.cur_epoch-1]).c_str():"pending"));
    }
    else
        ui->train_prog->setValue(0);


    ui->end_training->setEnabled(train.running);

    if(train.cur_epoch >= error_view_epoch)
        plot_error();

    if(ui->tabWidget->currentIndex() == 0)
        ui->statusbar->showMessage(train.status.c_str());
}


void fuzzy_labels(tipl::image<3>& label,const std::vector<size_t>& weights);
std::vector<size_t> get_label_count(const tipl::image<3>& label,size_t out_count);
void MainWindow::on_list1_currentRowChanged(int currentRow)
{
    if(ui->list2->currentRow() != currentRow)
        ui->list2->setCurrentRow(currentRow);
    auto pos_index = float(ui->pos->value())/float(ui->pos->maximum());
    if(pos_index == 0.0f)
        pos_index = 0.5f;
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
        v2c1.set_range(0,tipl::max_value_mt(I1));
        ui->pos->setMaximum(I1.shape()[ui->view_dim->currentIndex()]-1);
    }
    else
    {
        I1.clear();
        I2.clear();
        ui->pos->setMaximum(0);
    }
    ui->pos->setValue(pos_index*float(ui->pos->maximum()));

    on_pos_valueChanged(ui->pos->value());
}

void MainWindow::on_list2_currentRowChanged(int currentRow)
{
    if(ui->list1->currentRow() != currentRow)
        ui->list1->setCurrentRow(currentRow);
}

void label_on_images(QImage& I,
                     const tipl::image<3>& I2,
                     const tipl::shape<3>& dim,
                     int slice_pos,
                     int cur_label_index,
                     int out_count)
{
    float display_ratio = float(I.width())/float(I2.width());
    std::vector<tipl::image<2,char> > region_masks(out_count);
    if(I2.depth() != dim.depth())
    {
        tipl::par_for(out_count,[&](size_t i)
        {
            auto slice = I2.alias(dim.size()*i,dim).slice_at(slice_pos);
            tipl::image<2,char> mask(slice.shape());
            for(size_t pos = 0;pos < slice.size();++pos)
                mask[pos] = (slice[pos] == 1.0f ? 1:0);

            region_masks[i] = std::move(mask);
        });
    }
    else
    {
        for(auto& mask : region_masks)
            mask.resize(tipl::shape<2>(I2.width(),I2.height()));
        size_t base = slice_pos*I2.plane_size();
        for(size_t pos = 0;pos < I2.plane_size();++pos,++base)
        {
            int id = I2[base];
            if(id && id <= out_count)
                region_masks[id-1][pos] = 1;
        }
    }

    std::vector<tipl::rgb> colors(out_count);
    for(size_t i = 0;i < out_count;++i)
        colors[i] = tipl::rgb(111,111,255);

    {
        QPainter painter(&I);
        painter.setCompositionMode(QPainter::CompositionMode_Screen);
        painter.drawImage(0,0,tipl::qt::draw_regions(region_masks,colors,
                                                     false, // roi_fill_region
                                                     true, // roi_draw_edge
                                                     std::max<int>(1,display_ratio/5), // roi_edge_width
                                                     -1,display_ratio));
    }
}

void MainWindow::on_pos_valueChanged(int slice_pos)
{
    if(I1.empty())
        return ;

    auto d = ui->view_dim->currentIndex();
    auto sizes = tipl::space2slice<tipl::vector<2,int> >(d,I1.shape());
    float display_ratio = std::min<float>((ui->view1->width()-10)/sizes[0],(ui->view1->height()-10)/sizes[1]);
    if(display_ratio < 1.0f)
        display_ratio = 1.0f;


    QImage train_image;
    train_image << v2c1[tipl::volume2slice_scaled(I1,d,slice_pos,display_ratio)];

    if(I2.size() == I1.size()*out_count)
    {
        if(d == 2 && is_label)
            label_on_images(train_image,I2,I1.shape(),slice_pos,ui->label_slider->value(),out_count);

        train_scene2 << (QImage() << v2c2[tipl::volume2slice_scaled(
                            I2.alias(I1.size()*ui->label_slider->value(),I1.shape()),
                            ui->view_dim->currentIndex(),slice_pos,display_ratio)]).mirrored(d,d!=2);
    }
    else
        train_scene2 << QImage();
    train_scene1 << train_image.mirrored(d,d!=2);

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

