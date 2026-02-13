#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QMessageBox>
#include <QClipboard>
#include <QMovie>
#include "TIPL/tipl.hpp"
#include "console.h"


extern QSettings settings;

void MainWindow::on_action_evaluate_option_triggered()
{
    if(ui->postproc_widget->isVisible())
        ui->postproc_widget->hide();
    else
        ui->postproc_widget->show();
}

void MainWindow::update_evaluate_list(void)
{
    eval_I1_buffer.resize(evaluate_list.size());
    eval_I1_buffer_max.resize(evaluate_list.size());
    auto index = ui->evaluate_list->currentRow();
    ui->evaluate_list->clear();
    for(auto s: evaluate_list)
        ui->evaluate_list->addItem(QFileInfo(s).fileName());
    if(index >=0 && index < ui->evaluate_list->count())
        ui->evaluate_list->setCurrentRow(index);
    else
        ui->evaluate_list->setCurrentRow(0);
}
void MainWindow::on_action_evaluate_open_images_triggered()
{
    QStringList file = QFileDialog::getOpenFileNames(this,"Open Image",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    settings.setValue("work_dir",QFileInfo(file[0]).absolutePath());
    evaluate_list << file;
    update_evaluate_list();
    ui->evaluate->setEnabled(!evaluate.model->feature_string.empty());
}


void MainWindow::on_action_evaluate_copy_trained_network_triggered()
{

    auto& model = train.get_model();
    if(model->feature_string.empty())
    {
        QMessageBox::critical(this,"Error","No trained network");
        return;
    }
    ui->evaluate_builtin_networks->setCurrentIndex(0);
    evaluate.model = UNet3d(model->in_count,model->out_count,model->feature_string);
    evaluate.model->copy_from(*model);
    evaluate.model->eval();
    ui->evaluate_network_info->setText(QString("name: %1\n").arg(eval_name = train_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(evaluate_list.size());
    ui->evaluate_builtin_networks->setCurrentIndex(0);
}


void MainWindow::on_action_evaluate_open_network_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open Network File",settings.value("network_dir").toString() + "/" +
                                                    settings.value("network_file").toString() + ".net.gz","Network files (*net.gz);;All files (*)");
    if(fileName.isEmpty())
        return;
    if(!load_from_file(evaluate.model,fileName.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error","Invalid file format");
        return;
    }
    settings.setValue("network_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("network_file",eval_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    ui->evaluate_network_info->setText(QString("name: %1\n").arg(eval_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(evaluate_list.size());
    ui->evaluate_builtin_networks->setCurrentIndex(0);
}

void MainWindow::on_evaluate_builtin_networks_currentIndexChanged(int index)
{
    if(index > 0)
    {
        QString fileName =  QCoreApplication::applicationDirPath() + "/network/" + ui->evaluate_builtin_networks->currentText() + ".net.gz";
        if(!load_from_file(evaluate.model,fileName.toStdString().c_str()))
        {
            QMessageBox::critical(this,"Error","Failed to load network");
            return;
        }
        ui->evaluate_network_info->setText(QString("name: %1\n").arg(eval_name = ui->evaluate_builtin_networks->currentText()) + evaluate.model->get_info().c_str());
        ui->evaluate->setEnabled(evaluate_list.size());
    }
}

void MainWindow::on_evaluate_clicked()
{
    tipl::progress p("initiate evaluation");
    if(evaluate.running)
    {
        evaluate.stop();
        return;
    }
    evaluate.param.device = ui->evaluate_device->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->evaluate_device->currentIndex()-1):torch::Device(torch::kCPU);
    evaluate.param.image_file_name.clear();
    evaluate.param.prob_threshold = ui->postproc_prob_threshold->value();
    for(auto s : evaluate_list)
        evaluate.param.image_file_name.push_back(s.toStdString());

    ui->evaluate->setText("Stop");
    ui->evaluate_progress->setEnabled(true);
    ui->evaluate_progress->setMaximum(evaluate_list.size());
    ui->evaluate_list2->clear();
    evaluate.proc_strategy.match_resolution = ui->match_resolution->isChecked() | ui->match_orientation->isChecked();
    evaluate.proc_strategy.match_fov = ui->match_fov->isChecked() | ui->match_orientation->isChecked();
    evaluate.proc_strategy.match_orientation = ui->match_orientation->isChecked();
    evaluate.proc_strategy.output_format = ui->evaluate_output->currentIndex();
    // find template
    if(ui->match_orientation->isChecked())
    {
        /*
        evaluate.proc_strategy.template_file_name.clear();
        QDir dir(QCoreApplication::applicationDirPath() + "/template");
        dir.setNameFilters(QStringList() << "*.nii.gz");
        QFileInfoList files = dir.entryInfoList(QDir::Files);
        for (const QFileInfo& fileInfo : files)
        {
            if(eval_name.contains(fileInfo.fileName().remove(".nii.gz")))
            {
                evaluate.proc_strategy.template_file_name = (QCoreApplication::applicationDirPath() + "/template/" + fileInfo.fileName()).toStdString();
                tipl::out() << "rotation template: " << evaluate.proc_strategy.template_file_name << std::endl;
            }
        }
        */
        if(evaluate.proc_strategy.template_file_name.empty())
        {
            QMessageBox::critical(this,"ERROR",QString("cannot find a rotation template for ") + eval_name);
            return;
        }
    }
    evaluate.start();
    eval_timer->start();

}



void MainWindow::evaluating()
{
    console.show_output();
    if(!evaluate.running)
        eval_timer->stop();
    if(!evaluate.error_msg.empty())
    {
        QMessageBox::critical(this,"Error",evaluate.error_msg.c_str());
        evaluate.error_msg.clear();
    }
    ui->evaluate->setText(evaluate.running ? "Stop" : "Start");

    ui->action_evaluate_open_images->setEnabled(!evaluate.running);
    ui->evaluate_open_images->setEnabled(!evaluate.running);
    ui->action_evaluate_clear_all->setEnabled(!evaluate.running);
    ui->evaluate_clear_all->setEnabled(!evaluate.running);
    ui->action_evaluate_open_network->setEnabled(!evaluate.running);
    ui->evaluate_open_network->setEnabled(!evaluate.running);
    ui->action_evaluate_save_results->setEnabled(ui->evaluate_list2->count());
    ui->evaluate_save_results->setEnabled(ui->evaluate_list2->count());


    ui->evaluate_device->setEnabled(!evaluate.running);
    ui->evaluate_gif->setEnabled(evaluate.running);

    if(!evaluate.running)
        ui->evaluate_gif->movie()->stop();
    else
        ui->evaluate_gif->movie()->start();

    if(train.running)
    {
        ui->evaluate_progress->setValue(evaluate.cur_output);
        ui->evaluate_progress->setFormat(QString("%1/%2").arg(evaluate.cur_output).arg(evaluate_list.size()));
    }
    else
        ui->evaluate_progress->setValue(0);

    while(ui->evaluate_list2->count() < evaluate.cur_output)
    {
        ui->evaluate_list2->addItem(ui->evaluate_list->item(ui->evaluate_list2->count())->text());
        if(ui->evaluate_list2->currentRow() != ui->evaluate_list->currentRow())
            ui->evaluate_list2->setCurrentRow(ui->evaluate_list->currentRow());
        on_eval_pos_valueChanged(ui->eval_pos->value());
    }

    if(ui->tabWidget->currentIndex() == 0)
        ui->statusbar->showMessage(evaluate.status.c_str());

}


void MainWindow::on_action_evaluate_clear_all_triggered()
{
    evaluate.clear();
    evaluate_list.clear();
    eval_I1_buffer.clear();
    eval_I1_buffer_max.clear();
    ui->evaluate_list->clear();
    ui->evaluate_list2->clear();
    ui->evaluate->setEnabled(false);
    ui->action_evaluate_save_results->setEnabled(false);
    ui->evaluate_save_results->setEnabled(false);
    eval_scene1 << QImage();
    eval_scene2 << QImage();
}


void MainWindow::on_evaluate_list_currentRowChanged(int currentRow)
{
    float pos_index = 0.5f;
    if(ui->eval_pos->value() && ui->eval_pos->maximum())
        pos_index = float(ui->eval_pos->value())/float(ui->eval_pos->maximum());

    if(currentRow >= 0 && currentRow < eval_I1_buffer.size())
    {
        if(eval_I1_buffer[currentRow].empty())
        {
            tipl::vector<3> vs;
            tipl::io::gz_nifti in;
            if(!in.open(evaluate_list[currentRow].toStdString(),std::ios::in))
                return;
            in >> eval_I1_buffer[currentRow];
            eval_I1_buffer_max[currentRow] = tipl::max_value(eval_I1_buffer[currentRow]);
        }
        ui->eval_image_max->setMaximum(eval_I1_buffer_max[currentRow]);
        ui->eval_image_max->setSingleStep(eval_I1_buffer_max[currentRow]/20.0f);
        ui->eval_pos->setMaximum(eval_I1_buffer[currentRow].shape()[ui->eval_view_dim->currentIndex()]-1);
        ui->eval_pos->setValue(std::round(pos_index*float(ui->eval_pos->maximum())));
        ui->eval_image_max->setValue(tipl::max_value(tipl::volume2slice(eval_I1_buffer[currentRow],ui->eval_view_dim->currentIndex(),ui->eval_pos->value())));
    }
    else
        ui->eval_pos->setMaximum(0);
    on_eval_pos_valueChanged(ui->eval_pos->value());
    if(currentRow >= 0 && currentRow != ui->evaluate_list2->currentRow())
        ui->evaluate_list2->setCurrentRow(currentRow);
}

void MainWindow::on_evaluate_list2_currentRowChanged(int currentRow)
{
    if(currentRow >= 0 && currentRow != ui->evaluate_list->currentRow())
        ui->evaluate_list->setCurrentRow(currentRow);
}
void label_on_images(QImage& I,float display_ratio,
                     const tipl::image<3>& I2,
                     const tipl::shape<3>& shape,
                     unsigned char dim,
                     int slice_pos,
                     int cur_label_index,
                     int out_count);

void MainWindow::get_evaluate_views(QImage& view1,QImage& view2,float display_ratio)
{
    auto currentRow = ui->evaluate_list->currentRow();
    if(currentRow < 0 || currentRow >= eval_I1_buffer.size() || eval_I1_buffer[currentRow].empty())
        return;
    auto d = ui->eval_view_dim->currentIndex();
    auto sizes = tipl::space2slice<tipl::vector<2,int> >(d,eval_I1_buffer[currentRow].shape());
    if(display_ratio == 0.0f)
        display_ratio = std::min<float>(float((ui->eval_view1->width()-10))/float(sizes[0]),float(ui->eval_view1->height()-10)/float(sizes[1]));
    if(display_ratio < 1.0f)
        display_ratio = 1.0f;

    int slice_pos = std::max<int>(0,std::min<int>(ui->eval_pos->value(),eval_I1_buffer[currentRow].shape()[d]-1));
    view1 << eval_v2c1[tipl::volume2slice_scaled(eval_I1_buffer[currentRow],d,slice_pos,display_ratio)];

    if(currentRow < evaluate.cur_output)
    {
        const auto& eval_I2 = evaluate.label_prob[currentRow];
        auto eval_output_count = eval_I2.depth()/evaluate.raw_image_shape[currentRow][2];
        eval_v2c2.set_range(0,1);
        if(evaluate.is_label[currentRow] && eval_output_count == 1)
            eval_v2c2.set_range(0,evaluate.model->out_count);
        if(evaluate.is_label[currentRow] && eval_output_count >= 1)
            label_on_images(view1,display_ratio,
                            eval_I2,
                            evaluate.raw_image_shape[currentRow],
                            d,
                            slice_pos,
                            evaluate.model->out_count,
                            evaluate.model->out_count);

        ui->eval_label_slider->setMaximum(eval_output_count-1);
        ui->eval_label_slider->setVisible(eval_output_count > 1);
        view2 << eval_v2c2[tipl::volume2slice_scaled(
                           eval_I2.alias(
                            eval_I1_buffer[currentRow].size()*ui->eval_label_slider->value(),eval_I1_buffer[currentRow].shape()),
                           ui->eval_view_dim->currentIndex(),slice_pos,display_ratio)];

    }
    else
    {
        view2 = QImage();
        ui->eval_label_slider->setVisible(false);
    }
    view1 = view1.mirrored(d,d!=2);
    view2 = view2.mirrored(d,d!=2);
}

void MainWindow::on_eval_pos_valueChanged(int slice_pos)
{
    auto currentRow = ui->evaluate_list->currentRow();
    if(currentRow < 0 || currentRow >= eval_I1_buffer.size() || eval_I1_buffer[currentRow].empty())
        return;

    QImage view1,view2;
    get_evaluate_views(view1,view2);
    eval_scene1 << view1;
    eval_scene2 << view2;

    ui->action_evaluate_save_results->setEnabled(currentRow < evaluate.cur_output);
    ui->evaluate_save_results->setEnabled(currentRow < evaluate.cur_output);

}
void MainWindow::on_action_evaluate_save_results_triggered()
{
    auto currentRow = ui->evaluate_list2->currentRow();
    if(currentRow < 0 || currentRow >= evaluate.label_prob.size())
        return;
    QString file = evaluate_list[ui->evaluate_list->currentRow()];
    file = file.remove(".nii").remove(".gz") + ".result.nii.gz";
    file = QFileDialog::getSaveFileName(this,"Save Image",file,"NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;

    if(!evaluate.save_to_file(currentRow,file.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error","Cannot save file");
        return;
    }
    if(evaluate_list.size() <= 1 ||
       QMessageBox::question(this, "", "Save others?",QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
        return;

    tipl::progress p("saving files");
    for(int index = 0;p(index,evaluate_list.size());++index)
        if(index != currentRow)
        {
            std::string result;
            if(tipl::match_files(evaluate_list[currentRow].toStdString(),file.toStdString(),
                                 evaluate_list[index].toStdString(),result))
            {
                if(!evaluate.save_to_file(index,result.c_str()))
                {
                    QMessageBox::critical(this,"Error",QString("Cannot save ") + result.c_str());
                    return;
                }
            }
            else
            {
                QMessageBox::critical(this,"Error",QString("Cannot match file name for ") + evaluate_list[index]);
                return;
            }
        }
    if(!p.aborted())
        QMessageBox::information(this,"","Files saved");
}


void MainWindow::on_action_evaluate_copy_view_left_triggered()
{
    QImage view1,view2;
    get_evaluate_views(view1,view2);
    QApplication::clipboard()->setImage(view1);
    QMessageBox::information(this,"","Copied");
}


void MainWindow::on_action_evaluate_copy_view_right_triggered()
{
    QImage view1,view2;
    get_evaluate_views(view1,view2);
    QApplication::clipboard()->setImage(view2);
    QMessageBox::information(this,"","Copied");
}

void MainWindow::copy_to_clipboard(bool left,bool cropped)
{
    bool ok = true;
    int num_col = QInputDialog::getInt(nullptr,"", "Specify number of columns:",5,1,20,1&ok);
    if(!ok)
        return;
    int margin = 0;
    if(cropped)
    {
        margin = QInputDialog::getInt(nullptr,"", "Specify margin ",20,0,100,1&ok);
        if(!ok)
            return;
    }
    std::vector<QImage> images;
    tipl::progress p("generating",true);
    for(size_t i = 0;p(i,ui->evaluate_list2->count());++i)
    {
        ui->evaluate_list2->setCurrentRow(i);
        QImage view1,view2;
        get_evaluate_views(view1,view2);
        if(cropped)
        {
            tipl::vector<2,int> min,max;
            tipl::bounding_box(tipl::image<2,int32_t>(reinterpret_cast<const uint32_t*>(view2.constBits()),
                                                      tipl::shape<2>(view2.width(),view2.height())),min,max,0,margin);
            max -= min;
            view1 = view1.copy(min[0],min[1],max[0],max[1]);
            view2 = view2.copy(min[0],min[1],max[0],max[1]);
        }
        images.push_back(left ? view1 : view2);
    }
    QApplication::clipboard()->setImage(tipl::qt::create_mosaic(images,num_col));
}

void MainWindow::on_action_evaluate_copy_all_right_view_triggered()
{
    copy_to_clipboard(false,false);
}


void MainWindow::on_action_evaluate_copy_all_left_view_triggered()
{
    copy_to_clipboard(true,false);
}



void MainWindow::on_action_evaluate_copy_all_right_view_cropped_triggered()
{
    copy_to_clipboard(false,true);
}


void MainWindow::on_action_evaluate_copy_all_left_view_cropped_triggered()
{
    copy_to_clipboard(true,true);

}

void MainWindow::on_eval_show_contrast_panel_clicked()
{
    ui->eval_contrast->show();
}
void MainWindow::on_eval_image_max_valueChanged(double arg1)
{
    eval_v2c1.set_range(ui->eval_image_min->value(),ui->eval_image_max->value());
    on_eval_pos_valueChanged(ui->eval_pos->value());
}

void MainWindow::on_eval_image_min_valueChanged(double arg1)
{
    eval_v2c1.set_range(ui->eval_image_min->value(),ui->eval_image_max->value());
    on_eval_pos_valueChanged(ui->eval_pos->value());
}

void MainWindow::run_action(QString command)
{
    auto cur_index = ui->evaluate_list2->currentRow();
    if(cur_index < 0)
        return;
    float param1(0),param2(0);
    if(command == "defragment")
    {
        param1 = eval_option->get<float>("defragment_threshold");
        param2 = eval_option->get<float>("defragment_smoothing");
    }
    if(command == "soft_max")
        param1 = eval_option->get<float>("soft_max_prob");
    if(command == "defragment_each")
        param1 = eval_option->get<float>("defragment_threshold");
    if(command == "upper_threshold")
        param1 = eval_option->get<float>("upper_threshold_threshold");
    if(command == "lower_threshold")
        param1 = eval_option->get<float>("lower_threshold_threshold");
    if(command == "minus")
        param1 = eval_option->get<float>("minus_value");
    evaluate.proc_actions(command.toStdString().c_str(),param1,param2);
    on_eval_pos_valueChanged(ui->eval_pos->value());
}

