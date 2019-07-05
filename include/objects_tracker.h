#ifndef PEDESTRIANLOCALISATION_H
#define PEDESTRIANLOCALISATION_H

#include <ros/ros.h>
#include <ros/time.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <mbot_perception_msgs/PoseWithCovarianceArray.h>
#include <mbot_perception_msgs/RecognizedObject3D.h>
#include <mbot_perception_msgs/RecognizedObject3DList.h>
#include <mbot_perception_msgs/TrackedObject3D.h>
#include <mbot_perception_msgs/TrackedObject3DList.h>
#include <mbot_perception_msgs/DeleteObject3D.h>
#include <mbot_perception_msgs/DeleteObject3DRequest.h>
#include <mbot_perception_msgs/DeleteObject3DResponse.h>

#include <std_msgs/ColorRGBA.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <bayes_tracking/BayesFilter/bayesFlt.hpp>

#include <XmlRpcValue.h>
#include <string.h>
#include <vector>
#include <unordered_set>

#include <math.h>
#include "simple_tracking.h"
#include "asso_exception.h"

using namespace mbot_perception_msgs;

class ObjectsTracker {
public:
    ObjectsTracker();

private:
    void trackingThread();

    void detectorCallback(const RecognizedObject3DList::ConstPtr &msg, const std::string &detector);

    bool deleteTrackCallback(mbot_perception_msgs::DeleteObject3D::Request &req, mbot_perception_msgs::DeleteObject3D::Response &res);

    void eventInCallback(const std_msgs::String &msg);

    void destroySubscribers();

    void createSubscribers();

    bool parseParams(ros::NodeHandle);

    ros::Publisher pub_pose_array, pub_tracked_objects_array, pub_pose_with_covariance_array, pub_observation_with_covariance_array;
    std::string target_frame;

    SimpleTracking<EKFilter> *ekf = NULL;
    SimpleTracking<UKFilter> *ukf = NULL;

    std::map<std::pair<std::string, std::string>, ros::Subscriber> subscribers;

    // List of class names
    std::vector<std::string> names;
    std::unordered_set<std::string> names_lookup;
    std::vector<std::vector<int>> confusion_matrix;
    float detection_sampling_radius;

    bool running_requested = false;

};

#endif // PEDESTRIANLOCALISATION_H
