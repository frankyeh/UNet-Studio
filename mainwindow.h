#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QGraphicsScene>
#include <QTimer>
#include <memory>
#include "train.hpp"
#include "evaluate.hpp"



class QTextEdit;
class OptionTableWidget;
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::MainWindow *ui;
    OptionTableWidget* option,*eval_option;
public: //training
    QGraphicsScene train_scene1,train_scene2,error_scene;
    tipl::image<3> I1,I2;
    tipl::value_to_color<float> v2c1,v2c2;
    size_t error_view_epoch = 0;
    void get_train_views(QImage& view1,QImage& view2);
    void get_evaluate_views(QImage& view1,QImage& view2,float display_ratio = 0.0f);
public:
    QStringList evaluate_list;
    void update_evaluate_list(void);
public:
    QStringList image_list,label_list;
    std::vector<size_t> image_last_added_indices;
    std::vector<std::vector<size_t> > relations;

    void update_list(void);
public: //evalute
    QGraphicsScene eval_scene1,eval_scene2;
    std::vector<tipl::image<3> > eval_I1_buffer;
    std::vector<float> eval_I1_buffer_max;
    tipl::value_to_color<float> eval_v2c1,eval_v2c2;
private:
    int in_count = 1,out_count = 1;
    bool is_label = true;
    std::vector<int> label_count;
    std::vector<float> label_weight;
    void copy_to_clipboard(bool left,bool cropped);
public:
    train_unet train;
    evaluate_unet evaluate;
    QString train_name,eval_name;
    QTimer *timer,*eval_timer;
    void has_network(void);
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    bool eventFilter(QObject *obj, QEvent *event) override;
private slots:

    void training(void);
    void evaluating(void);
    void on_pos_valueChanged(int value);

    void on_list1_currentRowChanged(int currentRow);
    void on_action_evaluate_open_images_triggered();
    void on_eval_pos_valueChanged(int value);
    void on_evaluate_clicked();
    void on_eval_view_dim_currentIndexChanged(int index);
    void on_view_dim_currentIndexChanged(int index);
    void on_label_slider_valueChanged(int value);
    void on_eval_label_slider_valueChanged(int value);
    void on_train_stop_clicked();
    void on_train_start_clicked();
    void on_action_train_clear_all_triggered();

    void on_evaluate_list_currentRowChanged(int currentRow);
    void on_action_evaluate_copy_trained_network_triggered();
    void on_action_evaluate_open_network_triggered();
    void on_action_evaluate_clear_all_triggered();
    void on_list2_currentRowChanged(int currentRow);

    void on_evaluate_list2_currentRowChanged(int currentRow);
    void on_action_evaluate_save_results_triggered();
    void on_error_x_size_valueChanged(int arg1);
    void on_error_y_size_valueChanged(int arg1);
    void on_save_error_clicked();



    void plot_error();
    void run_action(QString);
    void on_actionConsole_triggered();
    void on_evaluate_builtin_networks_currentIndexChanged(int index);
    void on_action_evaluate_option_triggered();
    void on_action_train_options_triggered();

    // training

    void on_action_save_training_setting_triggered();
    void on_action_open_training_setting_triggered();
    void on_action_train_open_files_triggered();
    void on_action_train_open_labels_triggered();
    void on_action_train_new_network_triggered();
    void on_action_train_open_network_triggered();
    void on_action_train_save_network_triggered();
    void on_train_view_transform_clicked();
    void on_action_train_copy_view_triggered();
    void on_action_evaluate_copy_view_left_triggered();
    void on_action_evaluate_copy_view_right_triggered();
    void on_seed_valueChanged(int arg1);
    void on_action_evaluate_copy_all_right_view_triggered();
    void on_action_evaluate_copy_all_left_view_triggered();
    void on_eval_show_contrast_panel_clicked();
    void on_eval_image_max_valueChanged(double arg1);
    void on_eval_image_min_valueChanged(double arg1);
    void on_action_train_open_options_triggered();
    void on_action_train_save_options_triggered();
    void on_action_train_copy_view_left_triggered();
    void on_action_train_copy_view_right_triggered();
    void on_action_evaluate_copy_all_right_view_cropped_triggered();
    void on_action_evaluate_copy_all_left_view_cropped_triggered();
    void on_action_train_reorder_output_triggered();
    void on_action_train_copy_all_view_triggered();
    void on_actionAdd_Relation_triggered();
    void on_actionApply_Label_Weights_triggered();
};
#endif // MAINWINDOW_H
