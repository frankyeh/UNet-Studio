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
    evaluate.model->total_training_count = train.model->total_training_count;
    ui->evaluate_network->setText(QString("name: %1\n").arg(eval_name = train_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(true);
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
    QMessageBox::information(this,"","Loaded");
    settings.setValue("work_dir",QFileInfo(fileName).absolutePath());
    settings.setValue("work_file",eval_name = QFileInfo(fileName.remove(".net.gz")).fileName());
    ui->evaluate_network->setText(QString("name: %1\n").arg(eval_name) + evaluate.model->get_info().c_str());
    ui->evaluate->setEnabled(true);

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
    {
        ui->evaluate_list2->addItem(ui->evaluate_list->item(ui->evaluate_list2->count())->text());
        if(ui->evaluate_list2->currentRow() != ui->evaluate_list->currentRow())
            ui->evaluate_list2->setCurrentRow(ui->evaluate_list->currentRow());
        on_eval_pos_valueChanged(ui->eval_pos->value());
    }
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
    evaluate.clear();
    evaluate_list.clear();
    ui->evaluate_list->clear();
    ui->evaluate_list2->clear();
    ui->save_evale_image->setEnabled(false);
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
        eval_v2c1.set_range(0,tipl::max_value_mt(eval_I1));
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
    float display_ratio = 2.0f;
    auto d = ui->eval_view_dim->currentIndex();

    QImage evaluate_image;
    evaluate_image << eval_v2c1[tipl::volume2slice_scaled(eval_I1,ui->eval_view_dim->currentIndex(),slice_pos,display_ratio)];

    if(currentRow < evaluate.cur_output)
    {
        auto eval_output_count = evaluate.evaluate_output[currentRow].depth()/
                                              evaluate.evaluate_image_shape[currentRow][2];
        if(d == 2 && evaluate.is_label[currentRow] && eval_output_count > 1)
            label_on_images(evaluate_image,evaluate.evaluate_output[currentRow],slice_pos,
                            ui->eval_label_slider->value(),eval_output_count);

        ui->eval_label_slider->setMaximum(eval_output_count-1);
        eval_scene2 << (QImage() << eval_v2c2[tipl::volume2slice_scaled(
                           evaluate.evaluate_output[currentRow].alias(
                               eval_I1.size()*ui->eval_label_slider->value(),eval_I1.shape()),
                           ui->eval_view_dim->currentIndex(),slice_pos,display_ratio)]).mirrored(d,d!=2);
        ui->save_evale_image->setEnabled(true);
    }
    else
    {
        eval_scene2 << QImage();
        ui->save_evale_image->setEnabled(false);
    }
    eval_scene1 << evaluate_image.mirrored(d,d!=2);
}
void MainWindow::on_save_evale_image_clicked()
{
    QString file = QFileDialog::getSaveFileName(this,"Save Image",evaluate_list[ui->evaluate_list->currentRow()],"NIFTI files (*nii.gz);;All files (*)");
    if(file.isEmpty())
        return;
    auto currentRow = ui->evaluate_list2->currentRow();
    if(evaluate.evaluate_output[currentRow].depth() == evaluate.evaluate_image_shape[currentRow][2])
    {
        if(!tipl::io::gz_nifti::save_to_file(file.toStdString().c_str(),
                                         evaluate.evaluate_output[currentRow].alias(0,
                                                tipl::shape<3>(
                                                    evaluate.evaluate_image_shape[currentRow][0],
                                                    evaluate.evaluate_image_shape[currentRow][1],
                                                    evaluate.evaluate_image_shape[currentRow][2])),
                                         evaluate.evaluate_image_vs[currentRow],
                                         evaluate.evaluate_image_trans2mni[currentRow]))
        QMessageBox::critical(this,"Error","Cannot save file");
    }
    else
    {
        if(!tipl::io::gz_nifti::save_to_file(file.toStdString().c_str(),
                                         evaluate.evaluate_output[currentRow].alias(0,
                                                tipl::shape<4>(
                                                    evaluate.evaluate_image_shape[currentRow][0],
                                                    evaluate.evaluate_image_shape[currentRow][1],
                                                    evaluate.evaluate_image_shape[currentRow][2],
                                                    evaluate.evaluate_output[currentRow].depth()/
                                                    evaluate.evaluate_image_shape[currentRow][2])),
                                         evaluate.evaluate_image_vs[currentRow],
                                         evaluate.evaluate_image_trans2mni[currentRow]))
        QMessageBox::critical(this,"Error","Cannot save file");
    }
}

template<typename T,typename U>
void reduce_mt(const T& in,U& out,size_t gap = 0)
{
    if(gap == 0)
        gap = out.size();
    tipl::par_for(out.size(),[&](size_t j)
    {
        auto v = out[j];
        for(size_t pos = j;pos < in.size();pos += gap)
            v += in[pos];
        out[j] = v;
    });
}


void MainWindow::runAction(QString command)
{
    auto cur_index = ui->evaluate_list2->currentRow();
    if(cur_index < 0)
        return;
    auto this_image = evaluate.evaluate_output[cur_index].alias();
    auto dim = evaluate.evaluate_image_shape[cur_index];
    auto& cur_is_label = evaluate.is_label[cur_index];
    auto eval_out_count = this_image.depth()/dim[2];
    if(this_image.empty())
        return;
    tipl::out() << "run " << command.toStdString();
    if(command == "remove_background")
    {

        tipl::image<3> sum_image(dim);
        reduce_mt(this_image,sum_image);

        for(size_t i = 0;i < postproc->get<float>("remove_fragments_smoothing");++i)
            tipl::filter::gaussian(sum_image);

        auto threshold = tipl::max_value(sum_image)*postproc->get<float>("remove_fragments_threshold");
        tipl::image<3> mask;
        tipl::threshold(sum_image,mask,threshold,1,0);
        tipl::morphology::defragment(mask);

        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            for(size_t pos = 0;pos < dim.size();++pos)
                if(!mask[pos])
                    I[pos] = 0;
        });
    }
    if(command == "defragment")
    {
        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::image<3,char> mask(I.shape());
            for(size_t i = 0;i < I.size();++i)
                mask[i] = (I[i] > 0.5f ? 1:0);
            tipl::morphology::defragment(mask);
            for(size_t i = 0;i < I.size();++i)
                if(!mask[i])
                    I[i] = 0;
        });
    }
    if(command =="normalize_volume")
    {
        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::normalize(I);
        });
        cur_is_label = false;
    }
    if(command =="gaussian_smoothing")
    {
        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::filter::gaussian(I);
        });
        cur_is_label = false;
    }
    if(command =="anisotropic_smoothing")
    {
        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::filter::anisotropic_diffusion(I);
        });
        cur_is_label = false;
    }

    if(command =="cal_prob")
    {
        tipl::image<3> sum_image(dim);
        reduce_mt(this_image,sum_image);

        tipl::par_for(eval_out_count,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            for(size_t pos = 0;pos < dim.size();++pos)
                if(sum_image[pos] != 0.0f)
                    I[pos] /= sum_image[pos];
        });
        cur_is_label = false;
    }
    if(command =="soft_max")
    {
        float soft_max_prob = postproc->get<float>("soft_max_prob");
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            float m = 0.0f;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                if(this_image[i] > m)
                    m = this_image[i];
            if(m == 0.0f)
                return;
            m *= soft_max_prob;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                this_image[i] = (this_image[i] >= m ? 1.0f:0.0f);
        });
        cur_is_label = true;
    }
    if(command =="convert_to_3d")
    {
        tipl::image<3> I(dim);
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            for(size_t i = pos,label = 1;i < this_image.size();i += dim.size(),++label)
                if(this_image[i])
                {
                    I[pos] = label;
                    return;
                }
        });
        I.swap(evaluate.evaluate_output[cur_index]);
        cur_is_label = true;
    }
    on_eval_pos_valueChanged(ui->eval_pos->value());
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

