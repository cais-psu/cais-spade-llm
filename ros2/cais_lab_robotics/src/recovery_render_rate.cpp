// Limit only the interactive viewer; physics and sensor timing remain independent.
#include <gazebo/common/Plugin.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/gui/GuiPlugin.hh>
#include <gazebo/gui/GLWidget.hh>
#include <gazebo/gui/MainWindow.hh>
#include <QTimer>

namespace gazebo
{
class RecoveryRenderRate : public GUIPlugin
{
public:
  RecoveryRenderRate() : GUIPlugin()
  {
    this->setFixedSize(0, 0);
    this->hide();
  }

  void Load(sdf::ElementPtr settings) override
  {
    const double rate = settings->HasElement("render_rate") ?
      settings->Get<double>("render_rate") : 30.0;
    if (rate != 15.0 && rate != 30.0 && rate != 60.0)
    {
      gzerr << "Recovery viewer render_rate must be 15, 30 or 60 Hz\n";
      return;
    }
    auto *timer = new QTimer(this);
    QObject::connect(timer, &QTimer::timeout, this, [timer, rate]()
    {
      auto *window = gui::get_main_window();
      auto *widget = window ? window->findChild<gui::GLWidget *>() : nullptr;
      if (widget && widget->Camera())
      {
        widget->SetRenderRate(rate);
        gzmsg << "Recovery Gazebo viewer render rate: " << rate << " Hz\n";
        timer->stop();
      }
    });
    timer->start(100);
  }
};
GZ_REGISTER_GUI_PLUGIN(RecoveryRenderRate)
}
