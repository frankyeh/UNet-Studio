#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QGraphicsScene>
#include <QTimer>
#include <memory>
#include "train.hpp"
#include "evaluate.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::MainWindow *ui;
public: //training
    QGraphicsScene train_scene1,train_scene2;
    tipl::image<3> I1,I2;
    tipl::value_to_color<float> v2c1,v2c2;
public:
    QStringList evaluate_list;
    void update_evaluate_list(void);
public:
    QStringList image_list,label_list;
    bool ready_to_train = false;
    void update_list(void);
public: //evalute
    QGraphicsScene eval_scene1,eval_scene2;
    tipl::image<3> eval_I1;
    tipl::value_to_color<float> eval_v2c1,eval_v2c2;
private:
    int out_count = 1;
public:
    train_unet train;
    evaluate_unet evaluate;
    QTimer *timer,*eval_timer;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:

    void training(void);
    void evaluating(void);
    void on_pos_valueChanged(int value);
    void on_open_files_clicked();
    void on_open_labels_clicked();
    void on_list1_currentRowChanged(int currentRow);
    void on_open_evale_image_clicked();
    void on_eval_pos_valueChanged(int value);
    void on_evaluate_clicked();
    void on_eval_view_dim_currentIndexChanged(int index);
    void on_view_dim_currentIndexChanged(int index);
    void on_label_slider_valueChanged(int value);
    void on_eval_label_slider_valueChanged(int value);
    void on_end_training_clicked();
    void on_start_training_clicked();
    void on_clear_clicked();

    void on_evaluate_list_currentRowChanged(int currentRow);
    void on_eval_from_train_clicked();
    void on_eval_from_file_clicked();
    void on_train_from_scratch_clicked();
    void on_load_network_clicked();
    void on_save_network_clicked();
    void on_evaluate_clear_clicked();
    void on_list2_currentRowChanged(int currentRow);
    void on_autofill_clicked();
    void on_show_transform_clicked();

    void on_evaluate_list2_currentRowChanged(int currentRow);
    void on_save_evale_image_clicked();
};
#endif // MAINWINDOW_H
