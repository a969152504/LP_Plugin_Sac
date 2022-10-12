#ifndef LP_PLUGIN_SAC_H
#define LP_PLUGIN_SAC_H

#include <torch/torch.h>
#include <torch/script.h>
#include <tensorboard_logger.h>

#include "LP_Plugin_Sac_global.h"

#include "plugin/lp_actionplugin.h"

#include <QObject>
#include <QOpenGLBuffer>
#include <QCheckBox>
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>

#include "opencv2/aruco.hpp"
#include "opencv2/aruco/charuco.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"

#include <yolo_v2_class.hpp>

#include <math.h>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QPushButton>

class QLabel;
class LP_ObjectImpl;
class QOpenGLShaderProgram;

/**
 * @brief The LP_Plugin_Sac class
 * Sac RL
 */

#define LP_Plugin_Sac_iid "cpii.rp5.SmartFashion.LP_Plugin_Sac/0.1"

class LP_Plugin_Sac : public LP_ActionPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID LP_Plugin_Sac_iid)
    Q_INTERFACES(LP_ActionPlugin)

public:
    virtual ~LP_Plugin_Sac();

        // LP_Functional interface
        QWidget *DockUi() override;
        bool Run() override;
        bool eventFilter(QObject *watched, QEvent *event) override;

        // LP_ActionPlugin interface
        QString MenuName();
        QAction *Trigger();

signals:


        // LP_Functional interface
public slots:

        void FunctionalRender_L(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options) override;
        void FunctionalRender_R(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options) override;
        void Reinforcement_Learning();
        void savedata(QString fileName, std::vector<float> datas);
        void loaddata(std::string fileName, std::vector<float> &datas);
        void drawcart(torch::Tensor &state, int thickness, int step);

private:
        bool mInitialized_L = false;
        bool mInitialized_R = false;
        bool mRunReinforcementLearning = false;
        bool mTraining = false;
        bool gQuit = false;

        std::shared_ptr<QWidget> mWidget;
        QLabel *mLabel = nullptr;
        QLabel *mLabel2 = nullptr;
        QPushButton *mbutton1 = nullptr;
        QOpenGLShaderProgram *mProgram_L = nullptr,
                             *mProgram_R = nullptr;

        double pi = M_PI;

        float total_reward, total_critic_loss, total_policy_loss;
        int episodecount;

        /**
         * @brief initializeGL initalize any OpenGL resource
         */
        void initializeGL_L();
        void initializeGL_R();

};


#endif // LP_PLUGIN_SAC_H
