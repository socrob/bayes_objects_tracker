#include "objects_tracker.h"

using namespace mbot_perception_msgs;

ObjectsTracker::ObjectsTracker() {
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    std::string pub_topic_pose_array, pub_topic_tracked_objects_array, pub_topic_pose_with_covariance_array, pub_topic_observation_with_covariance_array;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle("~");

    bool r = parseParams(private_node_handle);

    if (!r) {
        ros::shutdown();
        return;
    }

    auto delete_objects_server = private_node_handle.advertiseService("delete_object", &ObjectsTracker::deleteTrackCallback, this);

    ROS_INFO_STREAM("\n\n\n\n\n\n\n\n\ndelete_objects_server: " << delete_objects_server.getService());

    auto sub = private_node_handle.subscribe("event_in", 1, &ObjectsTracker::eventInCallback, this);

    private_node_handle.param("pose_array", pub_topic_pose_array, std::string("objects_tracker/pose_array"));
    pub_pose_array = n.advertise<geometry_msgs::PoseArray>(pub_topic_pose_array, 10);

    private_node_handle.param("tracked_objects_array", pub_topic_tracked_objects_array,
                              std::string("objects_tracker/tracked_objects"));
    pub_tracked_objects_array = n.advertise<mbot_perception_msgs::TrackedObject3DList>(pub_topic_tracked_objects_array,
                                                                                       10);

    private_node_handle.param("pub_topic_pose_with_covariance_array", pub_topic_pose_with_covariance_array,
                              std::string("objects_tracker/pose_with_covariance_array"));
    pub_pose_with_covariance_array = n.advertise<mbot_perception_msgs::PoseWithCovarianceArray>(
            pub_topic_pose_with_covariance_array, 10);

    private_node_handle.param("pub_topic_observation_with_covariance_array",
                              pub_topic_observation_with_covariance_array,
                              std::string("objects_tracker/observation_with_covariance_array"));
    pub_observation_with_covariance_array = n.advertise<mbot_perception_msgs::PoseWithCovarianceArray>(
            pub_topic_observation_with_covariance_array, 10);

    boost::thread tracking_thread(boost::bind(&ObjectsTracker::trackingThread, this));

    ros::spin();
}

void ObjectsTracker::destroySubscribers() {
    for (auto &it : subscribers) const_cast<ros::Subscriber &>(it.second).shutdown();
}

void ObjectsTracker::createSubscribers() {
    ros::NodeHandle n;
    for (auto &it : subscribers)
        subscribers[it.first] = n.subscribe<mbot_perception_msgs::RecognizedObject3DList>(it.first.second.c_str(), 10,
                                                                                          boost::bind(
                                                                                                  &ObjectsTracker::detectorCallback,
                                                                                                  this, _1,
                                                                                                  it.first.first));
}

void ObjectsTracker::eventInCallback(const std_msgs::String &msg) {

    if (msg.data == "e_start") {
        if (!running_requested) {
            ROS_INFO("START EVENT RECEIVED");
            createSubscribers();
            running_requested = true;
        } else {
            ROS_ERROR("START EVENT RECEIVED, ALREADY STARTED");
        }
    }

    if (msg.data == "e_stop") {
        if (running_requested) {
            ROS_INFO("STOP EVENT RECEIVED");
            destroySubscribers();
            running_requested = false;
        } else {
            ROS_WARN("STOP EVENT RECEIVED, ALREADY STOPPED");
        }
    }

}

bool ObjectsTracker::deleteTrackCallback(mbot_perception_msgs::DeleteObject3D::Request  &req,
                                         mbot_perception_msgs::DeleteObject3D::Response &res) {

    if (ekf != NULL) {
        ekf->deleteTrack(req.uuid);
    } else {
        ekf->deleteTrack(req.uuid);
    }

    return true;
}

bool ObjectsTracker::parseParams(ros::NodeHandle n) {

    n.param("target_frame", target_frame, std::string("/base_link"));
    n.getParam("detection_sampling_radius", detection_sampling_radius);

    n.getParam("names", names);
    names_lookup = std::unordered_set<std::string>(names.begin(), names.end());

    std::stringstream names_string;
    for (auto name : names) names_string << name << ", ";
    ROS_INFO_STREAM("\n\n names: " << names_string.str());

    XmlRpc::XmlRpcValue confusion_matrix_xmlrpc;
    n.getParam("confusion_matrix", confusion_matrix_xmlrpc);
    ROS_ASSERT(confusion_matrix_xmlrpc.getType() == XmlRpc::XmlRpcValue::TypeArray);
    ROS_ASSERT(confusion_matrix_xmlrpc.size() == names.size() + 1);

    confusion_matrix.resize(names.size() + 1);
    for (int i = 0; i < confusion_matrix_xmlrpc.size(); i++) {
        XmlRpc::XmlRpcValue row = confusion_matrix_xmlrpc[i];
        ROS_ASSERT(row.getType() == XmlRpc::XmlRpcValue::TypeArray);
        ROS_ASSERT(row.size() == names.size());
        confusion_matrix[i].resize(names.size(), 0);
        for (int j = 0; j < row.size(); j++) confusion_matrix[i][j] = row[j];
    }

    std::string filter, prediction_type;
    int sequence_size;
    double sequence_time;
    bool print_debug_info = false;

    n.getParam("filter_type", filter);
    n.getParam("prediction_type", prediction_type);
    n.getParam("sequence_size", sequence_size);
    n.getParam("sequence_time", sequence_time);
    n.getParam("print_debug_info", print_debug_info);

    if (sequence_size <= 0) {
        ROS_FATAL("Parameter sequence_size must be greater than 0");
        return false;
    }
    ROS_INFO_STREAM("Using sequence size: " << sequence_size);

    if (sequence_time <= 0.00001) {
        ROS_FATAL("Parameter sequence_time must be greater than 0.0");
        return false;
    }
    ROS_INFO_STREAM("Using sequence time: " << sequence_time);
    ROS_INFO_STREAM("Setting print_debug_info to: " << print_debug_info?"true":"false");


    ROS_INFO_STREAM("Found filter type: " << filter);
    ROS_INFO_STREAM("Found prediction type: " << prediction_type);

    if (filter == "EKF")
        ekf = new SimpleTracking<EKFilter>(names, confusion_matrix, target_frame, detection_sampling_radius,
                                           static_cast<unsigned int>(sequence_size), sequence_time, print_debug_info);
    else if (filter == "UKF")
        ukf = new SimpleTracking<UKFilter>(names, confusion_matrix, target_frame, detection_sampling_radius,
                                           static_cast<unsigned int>(sequence_size), sequence_time, print_debug_info);
    else {
        ROS_FATAL_STREAM("Filter type " << filter << " is not specified. Unable to create the tracker.");
        return false;
    }

    if (prediction_type == "CV") {

        XmlRpc::XmlRpcValue cv_noise;
        n.getParam("cv_noise_params", cv_noise);
        ROS_ASSERT(cv_noise.getType() == XmlRpc::XmlRpcValue::TypeStruct);
        ROS_INFO_STREAM("Constant Velocity Model noise: " << cv_noise);
        if (ukf) {
            ukf->createConstantVelocityModel(cv_noise["x"], cv_noise["y"], cv_noise["z"]);
        } else if (ekf) {
            ekf->createConstantVelocityModel(cv_noise["x"], cv_noise["y"], cv_noise["z"]);
        } else {
            ROS_FATAL("Tracker not initialised!");
            return false;
        }
        ROS_INFO_STREAM("Created " << filter << " based tracker using 3D constant velocity prediction model.");

    } else if (prediction_type == "S") {

        if (ukf) {
            ukf->createStaticModel();
        } else if (ekf) {
            ekf->createStaticModel();
        } else {
            ROS_FATAL("Tracker not initialised!");
            return false;
        }
        ROS_INFO_STREAM("Created " << filter << " based tracker using static prediction model.");

    }

    XmlRpc::XmlRpcValue detectors;
    n.getParam("detectors", detectors);
    ROS_ASSERT(detectors.getType() == XmlRpc::XmlRpcValue::TypeStruct);

    for (XmlRpc::XmlRpcValue::ValueStruct::const_iterator it = detectors.begin(); it != detectors.end(); ++it) {
        string detector_name = static_cast<string>(it->first);
        ROS_INFO_STREAM("Found detector: " << detector_name << " ==> " << detectors[it->first]);
        try {

            association_t association_alg;
                string association_alg_string = detectors[detector_name]["matching_algorithm"];

            if(association_alg_string == "NN")
                    association_alg = NN;
            else if (association_alg_string == "NN_LABELED")
                    association_alg = NN;
            else if (association_alg_string == "NNJPDA")
                    association_alg = NNJPDA;
            else if (association_alg_string == "NNJPDA_LABELED")
                    association_alg = NNJPDA_LABELED;
            else
                throw (asso_exception());

            if (ukf) {
                ukf->addDetectorModel(detector_name, association_alg,
                                      detectors[detector_name]["cartesian_noise_params"]["x"],
                                      detectors[detector_name]["cartesian_noise_params"]["y"],
                                      detectors[detector_name]["cartesian_noise_params"]["z"]);
            } else {
                ekf->addDetectorModel(detector_name, association_alg,
                                      detectors[detector_name]["cartesian_noise_params"]["x"],
                                      detectors[detector_name]["cartesian_noise_params"]["y"],
                                      detectors[detector_name]["cartesian_noise_params"]["z"]);
            }
        } catch (std::exception &e) {
            ROS_FATAL_STREAM(""
                                     << e.what()
                                     << " "
                                     << detectors[detector_name]["matching_algorithm"]
                                     << " is not specified. Unable to add "
                                     << detector_name
                                     << " to the tracker. Please use either NN or NNJPDA as association algorithms."
            );
            return false;
        }

        ros::Subscriber sub;
        subscribers[std::pair<std::string, std::string>(detector_name, detectors[detector_name]["topic"])] = sub;
    }

    return true;
}


void ObjectsTracker::trackingThread() {
    ros::Rate fps(7);
    double time_sec = 0.0;

    while (ros::ok()) {
        fps.sleep();
        if (!running_requested) continue;

        vector<mbot_perception_msgs::TrackedObject3D> object_tracks;
        if (ekf != NULL) {
            object_tracks = ekf->track(&time_sec);
        } else {
            object_tracks = ukf->track(&time_sec);
        }

        geometry_msgs::PoseArray pose_array_msg;
        pose_array_msg.header.stamp.fromSec(time_sec);
        pose_array_msg.header.frame_id = target_frame;

        mbot_perception_msgs::PoseWithCovarianceArray pose_with_covariance_array_msg;
        pose_with_covariance_array_msg.header.stamp.fromSec(time_sec);
        pose_with_covariance_array_msg.header.frame_id = target_frame;

        mbot_perception_msgs::TrackedObject3DList tracked_objects_msg;
        tracked_objects_msg.header.stamp.fromSec(time_sec);
        tracked_objects_msg.header.frame_id = target_frame;

        for (auto tracked_object_msg : object_tracks) {
            tracked_objects_msg.objects.push_back(tracked_object_msg);
            pose_array_msg.poses.push_back(tracked_object_msg.pose.pose);
            pose_with_covariance_array_msg.poses.push_back(tracked_object_msg.pose);
        }

        pub_tracked_objects_array.publish(tracked_objects_msg);
        pub_pose_array.publish(pose_array_msg);
        pub_pose_with_covariance_array.publish(pose_with_covariance_array_msg);

    }
}


void ObjectsTracker::detectorCallback(const mbot_perception_msgs::RecognizedObject3DList::ConstPtr &msg,
                                      const std::string &detector) {

    // Select the detections by class
    PoseWithCovarianceArray observation_with_covariance_array_msg;
    RecognizedObject3DList selected_detections;
    selected_detections.header = msg->header;
    selected_detections.image_header = msg->image_header;
    std::copy_if(msg->objects.begin(), msg->objects.end(), std::back_inserter(selected_detections.objects),
                 [this](mbot_perception_msgs::RecognizedObject3D object) {
                     return names_lookup.find(object.class_name) != names_lookup.end();
                 });

    if (ekf != NULL) {
        ekf->addObservation(detector, selected_detections, observation_with_covariance_array_msg);
    } else {
        ukf->addObservation(detector, selected_detections, observation_with_covariance_array_msg);
    }

    pub_observation_with_covariance_array.publish(observation_with_covariance_array_msg);

}


int main(int argc, char **argv) {
    // Set up ROS.
    ros::init(argc, argv, "bayes_objects_tracker");
    new ObjectsTracker();
    return 0;
}
