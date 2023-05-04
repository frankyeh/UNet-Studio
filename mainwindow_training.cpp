#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QInputDialog>
#include <QClipboard>
#include <QSettings>
#include <QMessageBox>
#include <QMovie>
#include "TIPL/tipl.hpp"
#include "console.h"
extern QSettings settings;

void MainWindow::on_action_train_options_triggered()
{
    if(ui->option_widget->isVisible())
        ui->option_widget->hide();
    else
        ui->option_widget->show();
}

void MainWindow::on_action_open_training_setting_triggered()
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

    ui->action_train_open_labels->setEnabled(true);
    ui->train_open_labels->setEnabled(true);

}

void MainWindow::on_action_save_training_setting_triggered()
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
    if(!label_list.empty() && QFileInfo(label_list[0]).exists())
    {
         if(get_label_info(label_list[0].toStdString(),out_count,is_label))
         {
             ui->output_info->setText(QString("num label: %1 type: %2").arg(out_count).arg(is_label?"label":"scalar"));
             ui->label_slider->setMaximum(out_count-1);
         }
         else
         {
             QMessageBox::critical(this,"Error",QString("%1 is not a valid label image").arg(QFileInfo(label_list[0]).fileName()));
             label_list[0].clear();
         }
    }

    auto index = ui->list1->currentRow();
    ui->list1->clear();
    ui->list2->clear();
    bool ready_to_train = false;
    for(size_t i = 0;i < image_list.size();++i)
    {
        if(!QFileInfo(label_list[i]).exists())
            label_list[i].clear();
        ui->list1->addItem(QFileInfo(image_list[i]).fileName());
        auto item = ui->list1->item(i);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        ui->list2->addItem(label_list[i].isEmpty() ? QString("(to be assigned)") : QFileInfo(label_list[i]).fileName());
        ready_to_train = true;
    }
    ui->train_start->setEnabled(ready_to_train);

    if(index >=0 && index < ui->list1->count())
        ui->list1->setCurrentRow(index);
    else
        ui->list1->setCurrentRow(0);

    on_list1_currentRowChanged(ui->list1->currentRow());
}

void MainWindow::on_action_train_open_files_triggered()
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

    ui->action_train_open_labels->setEnabled(true);
    ui->train_open_labels->setEnabled(true);

}

void MainWindow::on_action_train_open_labels_triggered()
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


void MainWindow::on_action_train_auto_match_label_files_triggered()
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

void MainWindow::on_action_train_clear_all_triggered()
{
    image_list.clear();
    label_list.clear();
    ui->list1->clear();
    ui->action_train_open_labels->setEnabled(false);
    ui->train_open_labels->setEnabled(false);
    ui->train_start->setEnabled(false);
    update_list();
}

void MainWindow::has_network(void)
{
    ui->action_evaluate_copy_trained_network->setEnabled(true);
    ui->action_train_save_network->setEnabled(true);
    ui->train_save_network->setEnabled(true);
}
void MainWindow::on_action_train_new_network_triggered()
{
    auto feature = QInputDialog::getText(this,"","Please Specify Network Structure",QLineEdit::Normal,"8x8+16x16+32x32+64x64+128x128");
    if(feature.isEmpty())
        return;
    torch::manual_seed(0);
    at::globalContext().setDeterministicCuDNN(true);
    qputenv("CUDNN_DETERMINISTIC", "1");
    train.model = UNet3d(1,out_count,feature.toStdString());
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    ui->batch_size->setValue(1);
    has_network();

}
void MainWindow::on_action_train_open_network_triggered()
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

    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());

    has_network();
}
void MainWindow::on_action_train_save_network_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save network file",
                                                settings.value("work_dir").toString() + "/" +
                                                train_name + ".net.gz","Network files (*net.gz);;All files (*)");
    if(!fileName.isEmpty() && save_to_file(train.output_model,fileName.toStdString().c_str()))
    {
        QMessageBox::information(this,"","Network Saved");
        settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
        settings.setValue("work_file",train_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    }

}
#include <ATen/Context.h>
tipl::shape<3> unet_inputsize(const tipl::shape<3>& s);
void MainWindow::on_train_start_clicked()
{


    tipl::progress p("initiate training");


    // those parameters can be modified using training pause
    train.param.batch_size = ui->batch_size->value();
    train.param.learning_rate = ui->learning_rate->value();
    train.param.output_model_type = ui->training_output->currentIndex();

    if(train.running)
    {
        train.pause = !train.pause;
        return;
    }


    if(train.model->feature_string.empty())
    {
        on_action_train_new_network_triggered();
        if(train.model->feature_string.empty())
            return;
    }
    if(out_count != train.model->out_count)
    {
        tipl::out() << "copy pretrained model" << std::endl;
        auto new_model = UNet3d(1,out_count,train.model->feature_string);
        new_model->copy_from(*train.model.get());
        train.model = new_model;
    }

    //tipl::out() << show_structure(train.model);


    train.param.image_file_name.clear();
    train.param.label_file_name.clear();
    train.param.test_image_file_name.clear();
    train.param.test_label_file_name.clear();
    for(size_t i = 0;i < image_list.size();++i)
    {
        if(ui->list1->item(i)->checkState() == Qt::Checked)
        {
            train.param.image_file_name.push_back(image_list[i].toStdString());
            train.param.label_file_name.push_back(label_list[i].toStdString());
        }
        train.param.test_image_file_name.push_back(image_list[i].toStdString());
        train.param.test_label_file_name.push_back(label_list[i].toStdString());
    }
    if(train.param.image_file_name.empty())
    {
        QMessageBox::critical(this,"Error","Please specify the training data");
        return;
    }


    {
        tipl::io::gz_nifti in;
        if(!in.load_from_file(train.param.image_file_name[0]))
        {
            QMessageBox::critical(this,"Error","Invalid NIFTI format");
            return;
        }
        in.toLPS();
        in.get_image_dimension(train.model->dim);
        train.model->dim = unet_inputsize(train.model->dim);
        tipl::out() << "network input sizes: " << train.model->dim << std::endl;
    }

    train.param.device = ui->train_device->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->train_device->currentIndex()-1):torch::Device(torch::kCPU);
    train.param.epoch = ui->epoch->value();
    train.param.is_label = is_label;
    train.option = option;
    train.start();

    ui->train_prog->setMaximum(ui->epoch->value());
    ui->train_prog->setValue(1);
    timer->start();
    ui->train_start->setText(train.pause ? "Resume":"Pause");
    error_view_epoch = 0;
    error_scene << QImage();

    has_network();
}

void MainWindow::on_train_stop_clicked()
{
    train.stop();
}



void MainWindow::plot_error()
{
    if(train.cur_epoch > 1)
    {
        size_t x_size = ui->error_x_size->value();
        size_t y_size = ui->error_y_size->value();
        QImage image(x_size+10,y_size+10,QImage::Format_RGB32);
        QPainter painter(&image);
        painter.fillRect(image.rect(), Qt::white);
        painter.setPen(QPen(Qt::black, 2));
        painter.drawRect(QRectF(5, 5, x_size,y_size));

        std::vector<std::vector<float> > all_errors;

        all_errors.push_back(std::vector<float>(train.error.begin(),train.error.begin()+train.cur_epoch));
        for(size_t i = 0;i < train.test_error_foreground.size();++i)
        {
            all_errors.push_back(std::vector<float>(train.test_error_foreground[i].begin(),train.test_error_foreground[i].begin()+train.cur_epoch));
            all_errors.push_back(std::vector<float>(train.test_error_background[i].begin(),train.test_error_background[i].begin()+train.cur_epoch));
        }


        {
            std::vector<float> data;
            for(const auto& d : all_errors)
                data.insert(data.end(),d.begin(),d.end());

            for(auto& v : data)
                v = -std::log10(v);
            tipl::normalize_upper_lower(data,image.height()-10);

            size_t pos = 0;
            for(auto& d : all_errors)
            {
                std::copy(data.begin()+pos,data.begin()+pos+d.size(),d.begin());
                pos += d.size();
            }
        }


        std::vector<QColor> colors = {QColor(0,0,0),QColor(244,177,131),QColor(197,90,17),QColor(142,170,219),QColor(47,84,150)};
        auto x_scale = std::min<float>(5.0f,float(x_size)/float(train.cur_epoch+1));
        painter.setRenderHint(QPainter::Antialiasing, true);
        for(size_t i = 0;i < all_errors.size() && i < colors.size();++i)
        {
            if(all_errors[i].empty())
                continue;
            QVector<QPointF> points;
            for(size_t j = 0;j < all_errors[i].size();++j)
                points << QPointF(float(j)*x_scale+5,all_errors[i][j]+5);

            painter.setPen(QPen(colors[i],1.5f));
            painter.drawPolyline(points);
            //painter.setPen(QPen(Qt::black, 1));
            //painter.drawLine(5,p1.back().y(),p1.back().x(),p1.back().y());
        }

        error_view_epoch = train.cur_epoch;
        error_scene << image;
    }

}

void MainWindow::training()
{
    console.show_output();
    if(!train.running)
        timer->stop();

    if(!train.error_msg.empty())
    {
        QMessageBox::critical(this,"Error",train.error_msg.c_str());
        train.error_msg.clear();
    }

    ui->train_start->setText(train.running ? (train.pause ? "Resume":"Pause") : "Start");
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    ui->batch_size->setEnabled(!train.running || train.pause);
    ui->learning_rate->setEnabled(!train.running || train.pause);
    ui->training_output->setEnabled(!train.running || train.pause);
    ui->epoch->setEnabled(!train.running);

    ui->action_train_open_files->setEnabled(!train.running);
    ui->train_open_files->setEnabled(!train.running);
    ui->action_train_open_labels->setEnabled(!train.running);
    ui->train_open_labels->setEnabled(!train.running);
    ui->action_train_clear_all->setEnabled(!train.running);
    ui->train_clear_all->setEnabled(!train.running);
    ui->action_train_open_network->setEnabled(!train.running);
    ui->train_open_network->setEnabled(!train.running);
    ui->action_train_new_network->setEnabled(!train.running);
    ui->train_new_network->setEnabled(!train.running);
    ui->action_train_auto_match_label_files->setEnabled(!train.running);

    ui->train_device->setEnabled(!train.running);
    ui->save_error->setEnabled(!train.error.empty());
    ui->training_gif->setEnabled(train.running);


    if(!train.running || train.pause)
        ui->training_gif->movie()->stop();
    else
        ui->training_gif->movie()->start();

    if(train.running)
    {
        ui->train_prog->setEnabled(!train.pause);
        ui->train_prog->setValue(train.cur_epoch+1);
        ui->train_prog->setFormat(QString( "epoch: %1/%2 error: %3" ).arg(train.cur_epoch).arg(ui->train_prog->maximum()).arg(train.cur_epoch ? std::to_string(train.error[train.cur_epoch-1]).c_str():"pending"));
    }
    else
        ui->train_prog->setValue(0);


    ui->train_stop->setEnabled(train.running);

    if(train.cur_epoch >= error_view_epoch)
        plot_error();

    if(ui->tabWidget->currentIndex() == 0)
        ui->statusbar->showMessage(train.status.c_str());
}

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
        if(ui->train_view_transform->isChecked())
            load_image_and_label(*option,I1,I2,is_label,vs,vs,I1.shape(),time(0));
        if(!I2.empty())
        {
            if(out_count != 1 && !ui->train_view_3d_label->isChecked())
                tipl::expand_label_to_dimension(I2,out_count);
            else
                tipl::normalize(I2);
        }
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
void MainWindow::get_train_views(QImage& view1,QImage& view2)
{
    auto d = ui->view_dim->currentIndex();
    auto sizes = tipl::space2slice<tipl::vector<2,int> >(d,I1.shape());
    float display_ratio = std::min<float>(float((ui->view1->width()-10))/float(sizes[0]),float(ui->view1->height()-10)/float(sizes[1]));
    if(display_ratio < 1.0f)
        display_ratio = 1.0f;

    int slice_pos = ui->pos->value();
    view1 << v2c1[tipl::volume2slice_scaled(I1,d,slice_pos,display_ratio)];

    if(I2.size() == I1.size()*out_count)
    {
        if(d == 2 && is_label && ui->train_view_contour->isChecked())
            label_on_images(view1,I2,I1.shape(),slice_pos,ui->label_slider->value(),out_count);

        view2 << v2c2[tipl::volume2slice_scaled(
                            I2.alias(I1.size()*ui->label_slider->value(),I1.shape()),
                            ui->view_dim->currentIndex(),slice_pos,display_ratio)];
    }
    else
    {
        if(!I2.empty())
            view2 << v2c2[tipl::volume2slice_scaled(I2,ui->view_dim->currentIndex(),slice_pos,display_ratio)];
        else
            view2 = QImage();
    }
    view1 = view1.mirrored(d,d!=2);
    view2 = view2.mirrored(d,d!=2);
}
void MainWindow::on_pos_valueChanged(int slice_pos)
{
    if(I1.empty())
        return ;
    QImage view1,view2;
    get_train_views(view1,view2);
    train_scene1 << view1;
    train_scene2 << view2;
}


void MainWindow::on_train_view_transform_clicked()
{
    on_list1_currentRowChanged(ui->list1->currentRow());
}


void MainWindow::on_save_error_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Error",settings.value("on_save_error_clicked").toString(),"Text values (*.txt);;All files (*)");
    if(file.isEmpty())
        return;
    std::ofstream out(file.toStdString());
    out << "trainning_error ";
    for(size_t j = 0;j < train.test_error_foreground.size();++j)
        out << "test_foreground_error test_background_error ";
    for(size_t i = 0;i < train.error.size() && i < train.cur_epoch;++i)
    {
        out << train.error[i] << " ";
        for(size_t j = 0;j < train.test_error_foreground.size();++j)
        {
            out << train.test_error_foreground[j][i] << " ";
            out << train.test_error_background[j][i] << " ";
        }
        out << std::endl;
    }
    if(out.is_open())
        QMessageBox::information(this,"","Saved");
}


void MainWindow::on_action_train_copy_view_triggered()
{
    QImage view1,view2;
    get_train_views(view1,view2);
    QImage concatenatedImage(view1.width()+view2.width(),view1.height(),QImage::Format_ARGB32);
    QPainter painter(&concatenatedImage);
    painter.drawImage(0, 0, view1);
    painter.drawImage(view1.width(), 0, view2);
    painter.end();
    QApplication::clipboard()->setImage(concatenatedImage);
}
