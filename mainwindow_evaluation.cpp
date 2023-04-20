#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QSettings>
#include <QMessageBox>
#include <QMovie>
#include "TIPL/tipl.hpp"

extern QSettings settings;

void MainWindow::on_advance_eval_clicked()
{
    if(ui->postproc_widget->isVisible())
        ui->postproc_widget->hide();
    else
        ui->postproc_widget->show();
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
    QStringList file = QFileDialog::getOpenFileNames(this,"Open Image",settings.value("work_dir").toString(),"NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    evaluate_list << file;
    update_evaluate_list();
    ui->evaluate->setEnabled(!evaluate.model->feature_string.empty());
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
    evaluate.model->copy_from(*train.model.get(),torch::kCPU);
    ui->evaluate_network->setText(QString("name: %1\n").arg(eval_name = train_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(evaluate_list.size());
}


void MainWindow::on_eval_from_file_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open Network File",                                                settings.value("work_dir").toString() + "/" +
                                                    settings.value("work_file").toString() + ".net.gz","Network files (*net.gz);;All files (*)");
    if(fileName.isEmpty())
        return;
    if(!load_from_file(evaluate.model,fileName.toStdString().c_str()))
    {
        QMessageBox::critical(this,"Error","Invalid file format");
        return;
    }
    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",eval_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    ui->evaluate_network->setText(QString("name: %1\n").arg(eval_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(evaluate_list.size());
    ui->eval_networks->setCurrentIndex(0);
}

void MainWindow::on_eval_networks_currentIndexChanged(int index)
{
    if(index > 0)
    {
        QString fileName =  QCoreApplication::applicationDirPath() + "/network/" + ui->eval_networks->currentText() + ".net.gz";
        if(!load_from_file(evaluate.model,fileName.toStdString().c_str()))
        {
            QMessageBox::critical(this,"Error","Failed to load network");
            return;
        }
        ui->evaluate_network->setText(QString("name: %1\n").arg(ui->eval_networks->currentText()) + evaluate.model->get_info().c_str());
        ui->evaluate->setEnabled(evaluate_list.size());
    }
}

void MainWindow::on_evaluate_clicked()
{
    tipl::progress p("initiate evaluation");
    if(evaluate.running)
    {
        evaluate.stop();
        eval_timer->stop();
        ui->evaluate->setText("Start");
        return;
    }
    EvaluateParam param;
    param.device = ui->evaluate_device->currentIndex() >= 1 ? torch::Device(torch::kCUDA, ui->evaluate_device->currentIndex()-1):torch::Device(torch::kCPU);
    for(auto s : evaluate_list)
        param.image_file_name.push_back(s.toStdString());

    ui->evaluate->setText("Stop");
    ui->eval_prog->setEnabled(true);
    ui->eval_prog->setMaximum(evaluate_list.size());
    ui->evaluate_list2->clear();
    evaluate.option = eval_option;
    evaluate.preproc_strategy = ui->preproc->currentIndex();
    evaluate.postproc_strategy = ui->postproc->currentIndex();
    evaluate.start(param);
    eval_timer->start();

}



void MainWindow::evaluating()
{
    if(!evaluate.running)
        eval_timer->stop();
    if(!evaluate.error_msg.empty())
    {
        QMessageBox::critical(this,"Error",evaluate.error_msg.c_str());
        evaluate.error_msg.clear();
    }
    ui->evaluate->setText(evaluate.running ? "Stop" : "Start");
    ui->open_evale_image->setEnabled(!evaluate.running);
    ui->evaluate_clear->setEnabled(!evaluate.running);
    ui->eval_from_file->setEnabled(!evaluate.running);
    ui->eval_from_train->setEnabled(!evaluate.running);
    ui->evaluate_device->setEnabled(!evaluate.running);
    ui->evaluating->setEnabled(evaluate.running);
    if(!evaluate.running)
        ui->evaluating->movie()->stop();
    else
        ui->evaluating->movie()->start();

    if(train.running)
    {
        ui->eval_prog->setValue(evaluate.cur_output);
        ui->eval_prog->setFormat(QString("%1/%2").arg(evaluate.cur_output).arg(evaluate_list.size()));
    }
    else
        ui->eval_prog->setValue(0);

    while(ui->evaluate_list2->count() < evaluate.cur_output)
    {
        ui->evaluate_list2->addItem(ui->evaluate_list->item(ui->evaluate_list2->count())->text());
        if(ui->evaluate_list2->currentRow() != ui->evaluate_list->currentRow())
            ui->evaluate_list2->setCurrentRow(ui->evaluate_list->currentRow());
        on_eval_pos_valueChanged(ui->eval_pos->value());
    }
    ui->save_evale_image->setEnabled(ui->evaluate_list2->count());

    if(ui->tabWidget->currentIndex() == 1)
        ui->statusbar->showMessage(evaluate.status.c_str());

}


void MainWindow::on_evaluate_clear_clicked()
{
    evaluate.clear();
    evaluate_list.clear();
    ui->evaluate_list->clear();
    ui->evaluate_list2->clear();
    ui->evaluate->setEnabled(false);
    ui->save_evale_image->setEnabled(false);
    eval_scene1 << QImage();
    eval_scene2 << QImage();
}


void MainWindow::on_evaluate_list_currentRowChanged(int currentRow)
{
    auto pos_index = float(ui->eval_pos->value())/float(ui->eval_pos->maximum());
    if(pos_index == 0.0f)
        pos_index = 0.5f;
    eval_I1.clear();
    if(currentRow >= 0 && currentRow < evaluate_list.size())
    {
        tipl::vector<3> vs;
        if(!tipl::io::gz_nifti::load_from_file(evaluate_list[currentRow].toStdString().c_str(),eval_I1,vs))
            return;
        eval_v2c1.set_range(0,tipl::max_value_mt(eval_I1)*0.8f);
        eval_v2c2.set_range(0,1.0f);
        ui->eval_pos->setMaximum(eval_I1.shape()[ui->eval_view_dim->currentIndex()]-1);
    }
    else
        ui->eval_pos->setMaximum(0);
    ui->eval_pos->setValue(pos_index*float(ui->eval_pos->maximum()));
    on_eval_pos_valueChanged(ui->eval_pos->value());
    if(currentRow >= 0 && currentRow != ui->evaluate_list2->currentRow())
        ui->evaluate_list2->setCurrentRow(currentRow);
}

void MainWindow::on_evaluate_list2_currentRowChanged(int currentRow)
{
    if(currentRow >= 0 && currentRow != ui->evaluate_list->currentRow())
        ui->evaluate_list->setCurrentRow(currentRow);
}

void label_on_images(QImage& I,const tipl::image<3>& I2,int slice_pos,int cur_label_index,int out_count);
void MainWindow::on_eval_pos_valueChanged(int slice_pos)
{
    auto currentRow = ui->evaluate_list->currentRow();
    if(eval_I1.empty() || currentRow < 0)
        return;
    auto d = ui->eval_view_dim->currentIndex();

    auto sizes = tipl::space2slice<tipl::vector<2,int> >(d,eval_I1.shape());
    float display_ratio = std::min<float>((ui->eval_view1->width()-10)/sizes[0],(ui->eval_view1->height()-10)/sizes[1]);
    if(display_ratio < 1.0f)
        display_ratio = 1.0f;

    QImage network_input;
    network_input << eval_v2c1[tipl::volume2slice_scaled(eval_I1,ui->eval_view_dim->currentIndex(),slice_pos,display_ratio)];

    if(currentRow < evaluate.cur_output)
    {
        auto eval_output_count = evaluate.label_prob[currentRow].depth()/
                                              evaluate.raw_image_shape[currentRow][2];
        eval_v2c2.set_range(0,1);
        if(evaluate.is_label[currentRow] && eval_output_count == 1)
            eval_v2c2.set_range(0,evaluate.model->out_count);
        if(d == 2 && evaluate.is_label[currentRow] && eval_output_count > 1)
            label_on_images(network_input,evaluate.label_prob[currentRow],slice_pos,
                            ui->eval_label_slider->value(),eval_output_count);

        ui->eval_label_slider->setMaximum(eval_output_count-1);
        ui->eval_label_slider->setVisible(eval_output_count > 1);
        eval_scene2 << (QImage() << eval_v2c2[tipl::volume2slice_scaled(
                           evaluate.label_prob[currentRow].alias(
                               eval_I1.size()*ui->eval_label_slider->value(),eval_I1.shape()),
                           ui->eval_view_dim->currentIndex(),slice_pos,display_ratio)]).mirrored(d,d!=2);
        ui->save_evale_image->setEnabled(true);
    }
    else
    {
        eval_scene2 << QImage();
        ui->eval_label_slider->setVisible(false);
        ui->save_evale_image->setEnabled(false);
    }
    eval_scene1 << network_input.mirrored(d,d!=2);
}
void MainWindow::on_save_evale_image_clicked()
{
    auto currentRow = ui->evaluate_list2->currentRow();
    if(currentRow < 0 || currentRow >= evaluate.label_prob.size())
        return;
    QString file = evaluate_list[ui->evaluate_list->currentRow()].remove(".nii").remove(".gz");
    file += ".result.nii.gz";
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



void postproc_actions(const std::string& command,
                      float value1,float value2,
                      tipl::image<3>& this_image,
                      const tipl::shape<3>& dim,
                      char& is_label);
void MainWindow::runAction(QString command)
{
    auto cur_index = ui->evaluate_list2->currentRow();
    if(cur_index < 0)
        return;
    float param1(0),param2(0);
    if(command == "remove_background")
    {
        param1 = eval_option->get<float>("remove_fragments_smoothing");
        param2 = eval_option->get<float>("remove_fragments_threshold");
    }
    if(command == "")
    {
        param1 = eval_option->get<float>("soft_min_prob");
        param2 = eval_option->get<float>("soft_max_prob");
    }
    postproc_actions(command.toStdString(),param1,param2,
                     evaluate.label_prob[cur_index],
                     evaluate.raw_image_shape[cur_index],
                     evaluate.is_label[cur_index]);
    on_eval_pos_valueChanged(ui->eval_pos->value());
}

