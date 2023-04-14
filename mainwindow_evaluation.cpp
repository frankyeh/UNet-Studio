#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "optiontablewidget.hpp"
#include <QFileDialog>
#include <QSettings>
#include <QMessageBox>
#include <QMovie>
#include "TIPL/tipl.hpp"

extern QSettings settings;


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
    ui->evaluate_network->setText(QString("UNet %1->%2->%3").arg(evaluate.model->in_count).arg(evaluate.model->feature_string.c_str()).arg(evaluate.model->out_count));
    ui->evaluate->setEnabled(true);
    QMessageBox::information(this,"","Copied");
}


void MainWindow::on_eval_from_file_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open Network File",                                                settings.value("work_dir").toString() + "/" +
                                                    settings.value("work_file").toString() + ".net.gz","Network files (*net.gz);;All files (*)");
    if(fileName.isEmpty() || !load_from_file(evaluate.model,fileName.toStdString().c_str()))
        return;
    ui->evaluate_network->setText(QString("UNet %1->%2->%3").arg(evaluate.model->in_count).arg(evaluate.model->feature_string.c_str()).arg(evaluate.model->out_count));
    ui->evaluate->setEnabled(true);
    QMessageBox::information(this,"","Loaded");
    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",QFileInfo(fileName.remove(".net.gz")).fileName());
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
    ui->eval_label_slider->setMaximum(evaluate.model->out_count-1);
    ui->eval_label_slider->setValue(0);
    ui->evaluate_list2->clear();
    evaluate.start(param);
    eval_timer->start();

}



void MainWindow::evaluating()
{
    console.show_output();
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
    ui->eval_prog->setValue(evaluate.cur_output);
    ui->eval_prog->setFormat(QString("%1/%2").arg(evaluate.cur_output).arg(evaluate_list.size()));
    while(ui->evaluate_list2->count() < evaluate.cur_output)
        ui->evaluate_list2->addItem(ui->evaluate_list->item(ui->evaluate_list2->count())->text());
    ui->save_evale_image->setEnabled(ui->evaluate_list2->count());
    if(!evaluate.running)
    {
        eval_timer->stop();
        if(!evaluate.error_msg.empty())
            QMessageBox::critical(this,"Error",evaluate.error_msg.c_str());
        ui->evaluate->setText("Start");
    }
}


void MainWindow::on_evaluate_clear_clicked()
{
    evaluate_list.clear();
    evaluate.evaluate_output.clear();
    ui->evaluate_list->clear();
    ui->evaluate_list2->clear();
    ui->save_evale_image->setEnabled(false);
}


void MainWindow::on_evaluate_list_currentRowChanged(int currentRow)
{
    auto pos_index = ui->eval_pos->value();
    eval_I1.clear();
    if(currentRow >= 0 && currentRow < evaluate_list.size())
    {
        tipl::vector<3> vs;
        if(!tipl::io::gz_nifti::load_from_file(evaluate_list[currentRow].toStdString().c_str(),eval_I1,vs))
            return;
        eval_v2c1.set_range(0,tipl::max_value_mt(eval_I1));
        eval_v2c2.set_range(0,1.0f);
        ui->eval_pos->setMaximum(eval_I1.shape()[ui->eval_view_dim->currentIndex()]-1);
    }
    else
        ui->eval_pos->setMaximum(0);
    ui->eval_pos->setValue(pos_index);
    on_eval_pos_valueChanged(ui->eval_pos->value());
    if(currentRow != ui->evaluate_list2->currentRow())
        ui->evaluate_list2->setCurrentRow(currentRow);
}

void MainWindow::on_evaluate_list2_currentRowChanged(int currentRow)
{
    if(currentRow >= 0 && currentRow != ui->evaluate_list->currentRow())
        ui->evaluate_list->setCurrentRow(currentRow);
}

void MainWindow::on_eval_pos_valueChanged(int value)
{
    if(eval_I1.empty())
        return;


    eval_scene1 << (QImage() << eval_v2c1[tipl::volume2slice_scaled(eval_I1,ui->eval_view_dim->currentIndex(),value,2.0f)]);

    auto currentRow = ui->evaluate_list->currentRow();
    if(currentRow < evaluate.cur_output &&
       currentRow < evaluate.evaluate_output.size() &&
       !evaluate.evaluate_output[currentRow].empty() &&
       evaluate.evaluate_output[currentRow].size() == eval_I1.size()*evaluate.model->out_count)
    {
        eval_scene2 << (QImage() << eval_v2c2[tipl::volume2slice_scaled(
                           evaluate.evaluate_output[currentRow].alias(eval_I1.size()*ui->eval_label_slider->value(),eval_I1.shape()),
                           ui->eval_view_dim->currentIndex(),value,2.0f)]);
        ui->save_evale_image->setEnabled(true);
    }
    else
    {
        eval_scene2 << QImage();
        ui->save_evale_image->setEnabled(false);
    }

}
void MainWindow::on_save_evale_image_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Image",evaluate_list[ui->evaluate_list->currentRow()],"NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    auto currentRow = ui->evaluate_list2->currentRow();
    if(!tipl::io::gz_nifti::save_to_file(file.toStdString().c_str(),
                                         evaluate.evaluate_output[currentRow].alias(0,
                                                tipl::shape<4>(
                                                    evaluate.evaluate_image_shape[currentRow][0],
                                                    evaluate.evaluate_image_shape[currentRow][1],
                                                    evaluate.evaluate_image_shape[currentRow][2],
                                                    evaluate.evaluate_output[currentRow].depth()/
                                                    evaluate.evaluate_image_shape[currentRow][2])),
                                         evaluate.evaluate_image_vs[currentRow],
                                         evaluate.evaluate_image_trans[currentRow]))
    {
        QMessageBox::critical(this,"Error","Cannot save file");
    }
}


void MainWindow::runAction(QString command)
{
    tipl::out() << "run " << command.toStdString();
    /*
    {
        tipl::progress p("processing",true);
        size_t count = 0;
        tipl::par_for(evaluate.evaluate_output.size(),[&](size_t index)
        {
            if(!p(count,evaluate.evaluate_output.size()))
                return;
            auto dim = evaluate.evaluate_image_shape[index];
            size_t out_count = evaluate.evaluate_output[index].depth()/dim.depth();
            for(size_t label = 0;label < out_count;++label)
            {
                auto from = evaluate.evaluate_output[index].alias(dim.size()*label,dim);
                tipl::image<3> smoothed_from(from);
                tipl::filter::mean(smoothed_from);
                tipl::filter::mean(smoothed_from);
                tipl::image<3,char> mask(dim);
                for(size_t i = 0;i < mask.size();++i)
                    mask[i] = smoothed_from[i] > 0.5f ? 1:0;
                tipl::morphology::defragment(mask);
                tipl::morphology::dilation(mask);
                tipl::morphology::dilation(mask);
                for(size_t i = 0;i < mask.size();++i)
                    if(mask[i] == 0)
                        from[i] = 0;
                tipl::lower_threshold(from,0.0f);
                tipl::upper_threshold(from,1.0f);
            }
            ++count;
        });
        if(!p.aborted())
            QMessageBox::information(this,"Done","Completed");
    }*/
}

