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

    in_count = ini.value("in_count",in_count).toInt();
    out_count = ini.value("out_count",out_count).toInt();
    ui->epoch->setValue(ini.value("epoch",ui->epoch->value()).toInt());
    ui->batch_size->setValue(ini.value("batch_size",ui->batch_size->value()).toInt());
    ui->learning_rate->setValue(ini.value("learning_rate",ui->learning_rate->value()).toFloat());

    if(!ini.value("network_dir").toString().isEmpty())
    {
        auto fileName = ini.value("network_dir").toString() + "/" + ini.value("network_file").toString() + ".net.gz";
        if(QFileInfo(fileName).exists())
            load_network(fileName);
    }


    option->load(ini);

    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",QFileInfo(fileName.remove(".ini")).fileName());

    ui->tabWidget->setCurrentIndex(1);
    update_list();

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
    ini.setValue("in_count",in_count);
    ini.setValue("out_count",out_count);
    ini.setValue("epoch",ui->epoch->value());
    ini.setValue("batch_size",ui->batch_size->value());
    ini.setValue("learning_rate",ui->learning_rate->value());
    if(train.model->total_training_count)
    {
        ini.setValue("network_dir",settings.value("network_dir").toString());
        ini.setValue("network_file",settings.value("network_file").toString());
    }
    option->save(ini);
    ini.sync();

    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",QFileInfo(fileName.remove(".ini")).fileName());

}

void MainWindow::on_actionAdd_Relation_triggered()
{
    auto relation_str = QInputDialog::getText(this,"","Please Specify Related Labels",QLineEdit::Normal,"0 1 2 3 4");
    if(relation_str.isEmpty())
        return;
    std::istringstream in(relation_str.toStdString());
    std::vector<size_t> relation((std::istream_iterator<int>(in)),std::istream_iterator<int>());
    if(relation.size() < 2 || tipl::max_value(relation) >= out_count)
    {
        QMessageBox::critical(this,"ERROR","Invalid Relation");
        return;
    }
    relations.push_back(relation);
    update_list();
}

void MainWindow::update_list(void)
{
    auto index = ui->list1->currentRow();
    ui->list1->clear();
    ui->list2->clear();
    bool ready_to_train = false;
    for(size_t i = 0;i < image_list.size();++i)
    {
        ui->list1->addItem(QFileInfo(image_list[i]).fileName());
        auto item = ui->list1->item(i);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);

        auto label_text = QFileInfo(label_list[i]).fileName();
        if(!relations.empty())
        {
            for(const auto& each: relations)
            {
                label_text += ".r";
                for(const auto& v: each)
                    label_text += QString::number(v);
            }
        }
        ui->list2->addItem(label_list[i].isEmpty() ? QString("(to be assigned)") : label_text);
        ready_to_train = true;
    }
    ui->train_start->setEnabled(ready_to_train);

    if(index >=0 && index < ui->list1->count())
        ui->list1->setCurrentRow(index);
    else
        ui->list1->setCurrentRow(0);

    on_list1_currentRowChanged(ui->list1->currentRow());
    ui->action_train_open_labels->setEnabled(image_list.size());
    ui->train_open_labels->setEnabled(image_list.size());
    ui->train_start->setEnabled(ready_to_train);

    ui->view_channel->setMaximum(in_count-1);
    ui->output_info->setText(QString("num label: %1 type: %2").arg(out_count).arg(is_label?"label":"scalar"));
    ui->label_slider->setMaximum(out_count-1);


}

void MainWindow::on_action_train_open_files_triggered()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    {
         tipl::io::gz_nifti nii;
         if(!nii.open(fileNames[0].toStdString(),std::ios::in))
         {
             QMessageBox::critical(this,"ERROR","Cannot read the NIFTI file");
             return;
         }
         in_count = nii.dim(4);    
    }    

    settings.setValue("work_dir",QFileInfo(fileNames[0]).absolutePath());
    image_last_added_indices.clear();
    for(auto s : fileNames)
    {
        image_last_added_indices.push_back(image_list.size());
        image_list << s;
        label_list << QString();
    }
    update_list();

}

bool get_label_info(const std::string& label_name,std::vector<int>& out_count,bool& is_label)
{
    tipl::io::gz_nifti nii;
    if(!nii.open(label_name,std::ios::in))
        return false;
    if(nii.dim(4) != 1)
        out_count.resize(nii.dim(4));
    if(nii.is_integer())
    {
        is_label = true;
        if(nii.dim(4) == 1)
        {
            tipl::image<3,short> labels;
            nii >> labels;
            out_count.resize(tipl::max_value(labels));
            for(size_t i = 0;i < labels.size();++i)
                if(labels[i])
                    out_count[labels[i]-1]++;
        }
    }
    else
    {
        tipl::image<3,float> labels;
        nii >> labels;
        is_label = tipl::is_label_image(labels);
        if(nii.dim(4) == 1)
            out_count.resize(is_label ? tipl::max_value(labels) : 1);
        for(size_t i = 0;i < labels.size();++i)
            if(labels[i] != 0.0f)
                out_count[int(labels[i])-1]++;
    }

    if(out_count.size() > 128)
    {
        out_count.resize(1);
        is_label = false;
    }
    return true;
}


void MainWindow::on_action_train_open_labels_triggered()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this,"Select NIFTI images",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)"
    );
    if (fileNames.isEmpty())
        return;
    settings.setValue("work_dir",QFileInfo(fileNames[0]).absolutePath());

    if(!get_label_info(fileNames[0].toStdString(),label_count,is_label))
    {
        QMessageBox::critical(this,"Error",QString("%1 is not a valid label image").arg(QFileInfo(fileNames[0]).fileName()));
        return;
    }
    out_count = std::max<int>(out_count,label_count.size());

    // auto complete
    auto entry_index = ui->list2->currentRow();
    auto index = entry_index;
    for(int i = 0;i < fileNames.size() && index < label_list.size();++i,++index)
        label_list[index] = fileNames[i];

    image_last_added_indices.clear();
    image_last_added_indices.push_back(entry_index);

    for(int index = 0;index < label_list.size();++index)
        if(label_list[index].isEmpty())
        {
            std::string result;
            if(tipl::match_files(image_list[entry_index].toStdString(),label_list[entry_index].toStdString(),
                                  image_list[index].toStdString(),result) && QFileInfo(result.c_str()).exists())
            {
                label_list[index] = result.c_str();
                image_last_added_indices.push_back(index);
            }
        }
    update_list();

}


void MainWindow::on_action_train_clear_all_triggered()
{
    image_list.clear();
    label_list.clear();
    relations.clear();
    in_count = out_count = 1;
    is_label = true;
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
    if(image_list.empty())
    {
        QMessageBox::critical(this,"ERROR","Please specify training images");
        return;
    }
    auto feature = QInputDialog::getText(this,"","Please Specify Network Structure",QLineEdit::Normal,
                                         out_count <= 6 ? "8x8+16x16+32x32+64x64+128x128" :
                                         (out_count <= 10 ? "16x16+32x32+64x64+128x128+128x128" : "32x32+64x64+128x128+128x128+256x256"));
    if(feature.isEmpty())
        return;
    torch::manual_seed(0);
    at::globalContext().setDeterministicCuDNN(true);
    qputenv("CUDNN_DETERMINISTIC", "1");
    train.model = UNet3d(in_count,out_count,feature.toStdString());
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    has_network();

}
void MainWindow::load_network(QString fileName)
{
    if(!load_from_file(train.model,fileName.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error","Invalid file format");
        return;
    }
    settings.setValue("network_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("network_file",train_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    has_network();
}
void MainWindow::on_action_train_open_network_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open network file",
                                                settings.value("network_dir").toString() + "/" +
                                                settings.value("network_file").toString() + ".net.gz","Network files (*net.gz);;All files (*)");
    if(fileName.isEmpty())
        return;
    load_network(fileName);

}
void MainWindow::on_action_train_save_network_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save network file",
                                                settings.value("network_dir").toString() + "/" +
                                                train_name + ".net.gz","Network files (*net.gz);;All files (*)");
    if(!fileName.isEmpty() && save_to_file(train.get_model(),fileName.toStdString().c_str()))
    {
        QMessageBox::information(this,"","Network Saved");
        settings.setValue("network_dir",QFileInfo(fileName).absolutePath());
        settings.setValue("network_file",train_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    }
}


void MainWindow::on_actionApply_Label_Weights_triggered()
{
    QString default_weight = "0.1 0.1 0.1 0.1 0.1";
    if(!label_count.empty())
    {
        std::vector<float> v(label_count.size());
        for(size_t i = 0;i < v.size();++i)
            v[i] += 1.0f/(1.0f+float(label_count[i]));
        tipl::multiply_constant(v,1.0f/tipl::sum(v));
        std::string str;
        for(auto& each : v)
        {
            if(!str.empty())
                str += " ";
            str += std::to_string(float(std::max<int>(1,each*100.0f))/100.0f);
            while(str.back() == '0')
                str.pop_back();
        }
        default_weight = str.c_str();
    }

    default_weight = QInputDialog::getText(this,"","Specify back propagation weights for each label",QLineEdit::Normal,default_weight);
    if(default_weight.isEmpty())
        return;
    train.param.set_weight(default_weight.toStdString());
}



#include <ATen/Context.h>
extern tipl::program_option<tipl::out> po;
void MainWindow::on_train_start_clicked()
{


    train.param.batch_size = ui->batch_size->value();
    train.param.learning_rate = ui->learning_rate->value();
    train.param.epoch = ui->epoch->value();
    train.param.is_label = is_label;
    train.param.relations = relations;

    train.param.options.clear();
    for(auto& each : option->treemodel->name_data_mapping)
        train.param.options[each.first.toStdString()] = each.second->getValue().toFloat();

    if(train.running)
    {
        train.update_epoch_count();
        train.pause = !train.pause;
        return;
    }
    tipl::progress p("initiate training");
    if(train.model->feature_string.empty())
    {
        on_action_train_new_network_triggered();
        if(train.model->feature_string.empty())
            return;
    }
    if(out_count != train.model->out_count)
    {
        tipl::out() << "different output channel noted. padding model output..." << std::endl;
        auto new_model = UNet3d(in_count,out_count,train.model->feature_string);
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
        if(!std::filesystem::exists(image_list[i].toStdString()) ||
           !std::filesystem::exists(label_list[i].toStdString()))
            continue;
        if(ui->list1->item(i)->checkState() == Qt::Checked)
        {
            train.param.image_file_name.push_back(image_list[i].toStdString());
            train.param.label_file_name.push_back(label_list[i].toStdString());
        }
    }
    train.param.test_image_file_name = train.param.image_file_name;
    train.param.test_label_file_name = train.param.label_file_name;

    train.param.device = ui->train_device->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->train_device->currentIndex()-1):torch::Device(torch::kCPU);
    train.start();
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
        const int left_border = 40;
        const int upper_border = 10;
        QImage image(x_size+left_border+5,y_size+upper_border+20,QImage::Format_RGB32);
        QPainter painter(&image);
        painter.fillRect(image.rect(), Qt::white);
        painter.setPen(QPen(Qt::gray, 2));
        painter.drawRect(left_border,upper_border + y_size/4, x_size,y_size/2);
        painter.drawLine(left_border,upper_border + y_size/2,left_border + x_size,upper_border + y_size/2);
        painter.drawText(QRect(0,upper_border-8,left_border-4,16), Qt::AlignRight,"1");
        painter.drawText(QRect(0,upper_border-8+y_size/4,left_border-4,16), Qt::AlignRight,"0.1");
        painter.drawText(QRect(0,upper_border-8+y_size/2,left_border-4,16), Qt::AlignRight,"0.01");
        painter.drawText(QRect(0,upper_border-8+y_size*3/4,left_border-4,16), Qt::AlignRight,"0.001");
        painter.drawText(QRect(0,upper_border-8+y_size,left_border-4,16), Qt::AlignRight,"0.0001");
        painter.drawText(left_border + x_size/2-16,upper_border+y_size+16,"Epoch");
        painter.setPen(QPen(Qt::red, 2));

        painter.setPen(QPen(Qt::black, 2));
        painter.drawRect(left_border,upper_border,x_size ,y_size);

        std::vector<std::vector<float> > all_errors;
        for(size_t i = 0;i < train.test_error.size();++i)
            all_errors.push_back(std::vector<float>(train.test_error[i].begin(),train.test_error[i].begin()+train.cur_epoch));

        std::vector<QColor> colors = {QColor(0,0,0),QColor(244,177,131),QColor(197,90,17),QColor(142,170,219),QColor(47,84,150)};
        auto x_scale = std::min<float>(5.0f,float(x_size)/float(train.cur_epoch+1));
        painter.setRenderHint(QPainter::Antialiasing, true);
        for(size_t i = 0;i < all_errors.size() && i < colors.size();++i)
        {
            if(all_errors[i].empty())
                continue;
            QVector<QPointF> points;
            for(size_t j = 0;j < all_errors[i].size();++j)
                points << QPointF(float(j)*x_scale+left_border,-std::log10(all_errors[i][j])*y_size/4.0f+upper_border);

            painter.setPen(QPen(colors[i],1.5f));
            painter.drawPolyline(points);
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

    ui->train_start->setText(train.running ? (train.pause ? "Resume":"Pause") : "Start");
    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
    ui->batch_size->setEnabled(!train.running || train.pause);
    ui->learning_rate->setEnabled(!train.running || train.pause);
    ui->epoch->setEnabled(!train.running || train.pause);

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
    ui->save_error->setEnabled(!train.test_error.empty());
    ui->training_gif->setEnabled(train.running);


    if(!train.running || train.pause)
        ui->training_gif->movie()->stop();
    else
        ui->training_gif->movie()->start();

    if(train.running)
    {
        ui->train_prog->setEnabled(!train.pause);
        ui->train_prog->setMaximum(ui->epoch->value());
        ui->train_prog->setValue(train.cur_epoch+1);
        if(!train.test_error.empty())
            ui->train_prog->setFormat(QString( "epoch: %1/%2 error: %3" ).arg(train.cur_epoch).arg(ui->train_prog->maximum()).arg(train.cur_epoch ? std::to_string(train.test_error[0][train.cur_epoch-1]).c_str():"pending"));
    }
    else
        ui->train_prog->setValue(0);


    ui->train_stop->setEnabled(train.running);

    if(train.cur_epoch >= error_view_epoch)
        plot_error();

    if(ui->tabWidget->currentIndex() == 1)
        ui->statusbar->showMessage((train.get_status() + "|" + train.reading_status+"/"+train.augmentation_status+"/"+train.training_status).c_str());
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
        tipl::shape<3> shape;
        if(!read_image_and_label(image_list[currentRow].toStdString(),label_list[currentRow].toStdString(),in_count,I1,I2,shape))
            I2.clear();
        if(!is_label)
            tipl::normalize(I2);
        if(ui->train_view_transform->isChecked())
        {
            std::unordered_map<std::string,float> options;
            for(auto& each : option->treemodel->name_data_mapping)
                options[each.first.toStdString()] = each.second->getValue().toFloat();

            visual_perception_augmentation(options,I1,I2,is_label,shape,ui->seed->value());
        }
        if(ui->view_channel->value())
            std::copy(I1.begin()+shape.size()*ui->view_channel->value(),I1.begin()+shape.size()*(ui->view_channel->value()+1),I1.begin());
        I1.resize(shape);
        v2c1.set_range(0,tipl::max_value(I1));
        v2c2.set_range(0,is_label ? out_count : 1);
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
void MainWindow::on_train_view_transform_clicked()
{
    on_list1_currentRowChanged(ui->list1->currentRow());
}

void MainWindow::on_seed_valueChanged(int arg1)
{
    ui->train_view_transform->setChecked(true);
    on_list1_currentRowChanged(ui->list1->currentRow());
}


void MainWindow::on_list2_currentRowChanged(int currentRow)
{
    if(ui->list1->currentRow() != currentRow)
        ui->list1->setCurrentRow(currentRow);
}

void label_on_images(QImage& I,float display_ratio,
                     const tipl::image<3>& I2,
                     const tipl::shape<3>& raw_image_shape,
                     unsigned char dim,
                     int slice_pos,
                     int cur_label_index,
                     int out_count)
{
    std::vector<tipl::image<2,char> > region_masks(out_count);
    if(I2.depth() != raw_image_shape.depth())
    {
        tipl::par_for(out_count,[&](size_t i)
        {
            auto slice = tipl::volume2slice(I2.alias(raw_image_shape.size()*i,raw_image_shape),dim,slice_pos);
            tipl::image<2,char> mask(slice.shape());
            for(size_t pos = 0;pos < slice.size();++pos)
                mask[pos] = (slice[pos] == 1.0f ? 1:0);
            region_masks[i] = std::move(mask);
        },out_count);
    }
    else
    {
        auto slice = tipl::volume2slice(I2,dim,slice_pos);
        for(auto& mask : region_masks)
            mask.resize(slice.shape());
        for(size_t pos = 0;pos < slice.size();++pos)
        {
            int id = slice[pos];
            if(id && id <= out_count)
                region_masks[id-1][pos] = 1;
        }
    }

    std::vector<tipl::rgb> colors(std::max<int>(5,out_count));
    colors[0] = tipl::rgb(255,255,255);
    colors[1] = tipl::rgb(80,170,255);
    colors[2] = tipl::rgb(255,170,0);
    colors[3] = tipl::rgb(244,126,113);
    colors[4] = tipl::rgb(255,255,255);
    for(size_t i = 5;i < out_count;++i)
        colors[i] = tipl::rgb(111,111,255);

    {
        QPainter painter(&I);
        painter.setCompositionMode(QPainter::CompositionMode_Screen);
        painter.drawImage(0,0,tipl::qt::draw_regions(region_masks,colors,
                                                     true, // roi_draw_edge
                                                     2, // roi_edge_width
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
    if(!I2.empty() && is_label && ui->train_view_contour->isChecked())
        label_on_images(view1,display_ratio,I2,I1.shape(),d,slice_pos,out_count,out_count);

    view2 = QImage();
    if(!I2.empty())
        view2 << v2c2[tipl::volume2slice_scaled(
                            I2.alias((I2.size() == I1.size()*out_count) ? I1.size()*ui->label_slider->value() : 0,I1.shape()),
                            ui->view_dim->currentIndex(),slice_pos,display_ratio)];

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



void MainWindow::on_save_error_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Error",settings.value("on_save_error_clicked").toString(),"Text values (*.txt);;All files (*)");
    if(file.isEmpty())
        return;
    if(train.save_error_to(file.toStdString().c_str()))
        QMessageBox::information(this,"","Saved");
}



void MainWindow::on_action_train_reorder_output_triggered()
{
    if(train.running)
    {
        QMessageBox::critical(this,"","Cannot change output during training");
        return;
    }
    std::string cur_order;
    for(size_t i = 0;i < train.model->out_count;++i)
    {
        if(i)
            cur_order += " ";
        cur_order += std::to_string(i);
    }
    bool ok;
    QString new_order_text = QInputDialog::getText(nullptr, "Input", "Enter order:", QLineEdit::Normal, cur_order.c_str(), &ok);
    if (!ok || new_order_text.isEmpty())
        return;

    std::vector<int> new_order;
    {
        QStringList stringList = new_order_text.split(' ');
        std::transform(stringList.begin(), stringList.end(), std::back_inserter(new_order),
                       [](const QString& str) { return str.toInt(); });
    }

    train.model->to(torch::kCPU);

    auto old_model = UNet3d(train.model->in_count,train.model->out_count,train.model->feature_string);
    old_model->copy_from(*train.model);

    train.model = UNet3d(train.model->in_count,new_order.size(),train.model->feature_string);
    train.model->copy_from(*old_model);

    auto tensor_from = old_model->parameters();
    auto tensor_to = train.model->parameters();

    {
        auto tensor_buf_from = tensor_from.back().data_ptr<float>();
        auto tensor_buf_to = tensor_to.back().data_ptr<float>();
        for(size_t i = 0;i < new_order.size();++i)
            tensor_buf_to[i] = tensor_buf_from[new_order[i]];
    }

    {
        auto tensor_buf_from = tensor_from[tensor_from.size()-2].data_ptr<float>();
        auto tensor_buf_to = tensor_to[tensor_to.size()-2].data_ptr<float>();
        size_t total_size = tensor_to[tensor_to.size()-2].numel();
        size_t length =  total_size/new_order.size();
        for(size_t i = 0;i < new_order.size();++i)
        {
            std::copy(tensor_buf_from + length*new_order[i],
                      tensor_buf_from + length*new_order[i] + length,
                      tensor_buf_to + i*length);
        }
    }

    ui->train_network_info->setText(QString("name: %1\n").arg(train_name) + train.model->get_info().c_str());
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

void MainWindow::on_action_train_copy_view_left_triggered()
{
    QImage view1,view2;
    get_train_views(view1,view2);
    QApplication::clipboard()->setImage(view1);

}

void MainWindow::on_action_train_copy_view_right_triggered()
{
    QImage view1,view2;
    get_train_views(view1,view2);
    QApplication::clipboard()->setImage(view2);

}




void MainWindow::on_action_train_copy_all_view_triggered()
{
    bool ok = true;
    int num_col = QInputDialog::getInt(nullptr,"", "Specify number of columns:",5,1,20,1&ok);
    if(!ok)
        return;
    int num_image = QInputDialog::getInt(nullptr,"", "Specify number of images:",5,1,100,1&ok);
    if(!ok)
        return;
    ui->train_view_transform->setChecked(true);
    std::vector<QImage> images;
    tipl::progress p("generating",true);
    for(size_t i = 0;p(i,num_image);++i)
    {
        ui->seed->setValue(i);
        on_list1_currentRowChanged(ui->list1->currentRow());
        QImage view1,view2;
        get_train_views(view1,view2);
        QImage concatenatedImage(view1.width()+view2.width(),view1.height(),QImage::Format_RGB32);
        QPainter painter(&concatenatedImage);
        painter.drawImage(0, 0, view1);
        painter.drawImage(view1.width(), 0, view2);
        painter.end();
        images.push_back(concatenatedImage);
    }
    QApplication::clipboard()->setImage(tipl::qt::create_mosaic(images,num_col));
}



void MainWindow::on_action_train_open_options_triggered()
{
    QString file = QFileDialog::getOpenFileName(this,"Open Settings","settings.ini","Settings (*.ini);;All files (*)");
    if(file.isEmpty())
        return;
    QSettings s(file,QSettings::IniFormat);
    option->load(s);
    QMessageBox::information(this,"","Setting Loaded");
}


void MainWindow::on_action_train_save_options_triggered()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Error","settings.ini","Settings (*.ini);;All files (*)");
    if(file.isEmpty())
        return;
    {
        QSettings s(file,QSettings::IniFormat);
        option->save(s);
    }
    QMessageBox::information(this,"","Setting Saved");
}
