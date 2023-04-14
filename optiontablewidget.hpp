#ifndef OptionTABLEWIDGET_H
#define OptionTABLEWIDGET_H

#include <QTreeView>
#include <QItemDelegate>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <QSettings>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>

class OptionDelegate : public QItemDelegate
 {
     Q_OBJECT

 public:
    OptionDelegate(QObject *parent)
         : QItemDelegate(parent)
     {
     }

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                                const QModelIndex &index) const;
          void setEditorData(QWidget *editor, const QModelIndex &index) const;
          void setModelData(QWidget *editor, QAbstractItemModel *model,
                            const QModelIndex &index) const;
private slots:
    void emitCommitData();
 };



class OptionItem
{
private:
    QList<OptionItem*> childItems;
    OptionItem *parentItem;
public:
    QObject* GUI = nullptr;
    QString id;
    QVariant title,type,value;
    QString hint;
public:
    OptionItem(QVariant title_, QVariant type_, QString id_,QVariant value_, OptionItem *parent = nullptr):
        parentItem(parent),id(id_),title(title_),type(type_),value(value_)
    {
        if(parent)
            parent->appendChild(this);
    }

    ~OptionItem(){qDeleteAll(childItems);}

     void appendChild(OptionItem *item)
     {
         childItems.append(item);
     }
     OptionItem *child(int row){return childItems.value(row);}
     int childCount() const {return childItems.count();}
     int row() const
     {
         return (parentItem) ? parentItem->childItems.indexOf(const_cast<OptionItem*>(this)) : 0;
     }
     OptionItem *parent(void) {return parentItem;}
     void setParent(OptionItem *parentItem_) {parentItem = parentItem_;}
     QVariant getValue() const{return value;}
     void setValue(QVariant new_value);
     void setMinMax(float min,float max,float step);
     void setList(QStringList list);

};

class OptionTableWidget;
class TreeModel : public QAbstractItemModel
{
    Q_OBJECT
private:
    std::shared_ptr<OptionItem> root;
    std::map<QString,OptionItem*> root_mapping;
    std::map<QString,OptionItem*> name_data_mapping;
    std::map<QString,QVariant> name_default_values;
public:
    TreeModel(OptionTableWidget *parent);
    ~TreeModel();

    QVariant data(const QModelIndex &index, int role) const;
    bool setData ( const QModelIndex & index, const QVariant & value, int role);

    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
                      const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    void save(QSettings& set);
    void load(QSettings& set);
public:
    void addNode(QString root_name,QString id,QVariant title);
    QModelIndex addItem(QString root_name,QString id,QVariant title, QVariant type, QVariant value, QString hint = QString());
    QVariant getData(QString name) const
    {
        std::map<QString,OptionItem*>::const_iterator iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            throw std::runtime_error(std::string("cannot find ") + name.toStdString());
        return iter->second->value;
    }
    OptionItem& operator[](QString name)
    {
        std::map<QString,OptionItem*>::const_iterator iter = name_data_mapping.find(name);
        if(iter == name_data_mapping.end())
            throw std::runtime_error(std::string("cannot find ") + name.toStdString());
        return *(iter->second);
    }
    QStringList get_param_list(QString root_name)
    {
        QStringList result;
        OptionItem* parent = root_mapping[root_name];
        if(!parent)
            throw std::runtime_error("Cannot find the root node");
        for(int index = 0;index < parent->childCount();++index)
            if(!parent->child(index)->type.isNull()) // second layer tree node has type = QVariant() assigned in AddNode
                result.push_back(parent->child(index)->id);
        return result;
    }
    QStringList getParamList(void)
    {
        QStringList result;
        for(auto& iter : name_data_mapping)
            result << iter.first;
        return result;
    }
    void setDefault(QString);
private:

};
class MainWindow;
class OptionTableWidget : public QTreeView
{
    Q_OBJECT
private:
    MainWindow& mainwindow;
public:
    OptionDelegate* data_delegate;
    TreeModel* treemodel;
public:
    explicit OptionTableWidget(MainWindow& mainwindow_,QWidget *parent,const char* source);
    QVariant getData(QString name) const{return treemodel->getData(name);}
    bool has(QString name) const{return treemodel->getData(name).toBool();}
    template<typename T>
    T get(QString name) const
    {
        if constexpr(std::is_floating_point_v<T>)
            return treemodel->getData(name).toDouble();
        else
        {
            if constexpr(std::is_integral_v<T>)
                return treemodel->getData(name).toInt();
            else
                return treemodel->getData(name);
        }
    }

    void setData(QString name,QVariant data){(*treemodel)[name].setValue(data);}
    void setMinMax(QString name,float min,float max,float step){(*treemodel)[name].setMinMax(min,max,step);}
    void setList(QString name,QStringList list){(*treemodel)[name].setList(list);}
    void initialize(const char* source);
    void load(QSettings& set){treemodel->load(set);}
    void save(QSettings& set){treemodel->save(set);}

Q_SIGNALS:
    void runAction(QString command);
public slots:
    void setDefault(QString parent_id);
    void dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
    void action();
};

#endif // OptionTABLEWIDGET_H
