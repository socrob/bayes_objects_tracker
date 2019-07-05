/***************************************************************************
 *   Copyright (C) 2011 by Nicola Bellotto                                 *
 *   nbellotto@lincoln.ac.uk                                               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef SIMPLE_TRACKING_H
#define SIMPLE_TRACKING_H

#include <ros/ros.h>
#include <ros/time.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <mbot_perception_msgs/RecognizedObject3DList.h>
#include <mbot_perception_msgs/RecognizedObject3D.h>
#include <mbot_perception_msgs/TrackedObject3D.h>

#include <bayes_tracking/multitracker.h>
#include "models.h"

#include <cstdio>
#include <math.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/optional.hpp>
#include <unordered_map>

#include "ekf.h"
#include "ukf.h"

using namespace std;
using namespace MTRK;
using namespace Models;
using namespace mbot_perception_msgs;
using namespace geometry_msgs;

#define STATE_SPACE_SIZE 6
#define OBSERVATION_SPACE_SIZE 3
#define ROS_COVARIANCE_SIZE 6

// rule to detect lost track
template<class FilterType>
bool MTRK::isLost(const FilterType *filter, double stdLimit) {
    // track lost if var(x)+var(y)+var(z) > 1
    return filter->X(0, 0) + filter->X(2, 2) + filter->X(4, 4) > sqr(1.0);
}

// rule to create new track
template<class FilterType>
bool MTRK::initialize(FilterType *&filter, sequence_t &obsvSeq, observ_model_t om_flag) {
    assert(obsvSeq.size());

    double dt = obsvSeq.back().time - obsvSeq.front().time;
    if (!dt) {
        ROS_WARN("dt == 0 in filter initialisation form sequence");
        return false;
    }


    FM::Vec v((obsvSeq.back().vec - obsvSeq.front().vec) / dt);


//    x, vx, y, vy, z, vz (3D model)
    FM::Vec x(STATE_SPACE_SIZE);
    FM::SymMatrix X(STATE_SPACE_SIZE, STATE_SPACE_SIZE);

    x[0] = obsvSeq.back().vec[0];
    x[1] = v[0];
    x[2] = obsvSeq.back().vec[1];
    x[3] = v[1];
    x[4] = obsvSeq.back().vec[2];
    x[5] = v[2];
    X.clear();
    X(0, 0) = sqr(0.2);
    X(1, 1) = sqr(1.0);
    X(2, 2) = sqr(0.2);
    X(3, 3) = sqr(1.0);
    X(4, 4) = sqr(0.2);
    X(5, 5) = sqr(1.0);

    filter = new FilterType(STATE_SPACE_SIZE);
    filter->init(x, X);

    return true;
}

template<typename FilterType>
class SimpleTracking {
private:
    PoseStamped transformPose(Pose pose, std_msgs::Header header, string target_frame) {
        PoseStamped p_target;
        PoseStamped p_original;
        p_original.header = header;
        p_original.pose = pose;
        listener->waitForTransform(header.frame_id, target_frame, header.stamp, ros::Duration(3.0));
        listener->transformPose(target_frame, ros::Time(0), p_original, header.frame_id, p_target);
        return p_target;
    }

    PoseStamped transformPose(PoseStamped p_original, string target_frame) {
        PoseStamped p_target;
        listener->waitForTransform(p_original.header.frame_id, target_frame, p_original.header.stamp,
                                   ros::Duration(3.0));
        listener->transformPose(target_frame, ros::Time(0), p_original, p_original.header.frame_id, p_target);
        return p_target;
    }

    double euclidean_distance(Point p1, Point p2) {
        return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    }

    template<typename T>
    std::string num_to_str(T num) {
        std::stringstream ss;
        ss << num;
        return ss.str();
    }

    std::string generateUUID(std::string time, long id) {
        boost::uuids::name_generator gen(dns_namespace_uuid);
        time += num_to_str<long>(id);

        return num_to_str<boost::uuids::uuid>(gen(time.c_str()));
    }

    double getTime() {
        return ros::Time::now().toSec();
    }

public:

    SimpleTracking(vector<string> &class_names, vector<vector<int>> confusion_matrix, string &target_frame,
                   float detection_sampling_radius, unsigned int sequence_size, double sequence_time, bool print_debug_info) :
            class_names(class_names), target_frame(target_frame), detection_sampling_radius(detection_sampling_radius),
            sequence_size(sequence_size), sequence_time(sequence_time), print_debug_info(print_debug_info) {

        time = getTime();
        listener = new tf::TransformListener();
        startup_time_str = num_to_str<double>(time);

        int epsilon_sum = 0, total_sum = 0;

        if(print_debug_info) cout << "norm_confusion_matrix" << endl;
        B.resize(confusion_matrix.size());
        for (size_t i = 0; i < confusion_matrix.size(); i++) {
            bool epsilon = i == confusion_matrix.size() - 1;
            vector<int> &row = confusion_matrix[i];
            float sum = 0.0; // sum: sum of the elements of the row
            for (auto &e : row) sum += e;
            for (auto &e : row) {
                B[i].push_back(e / sum);
                if(print_debug_info) cout << e / sum << ", ";
            }
            total_sum += sum;
            epsilon_sum += epsilon ? sum : 0;
            if(print_debug_info) cout << endl;
        }

        epsilon_prior = (float) epsilon_sum / (float) total_sum;

        if(print_debug_info) cout << endl << endl << "epsilon_prior: " << epsilon_prior << endl << endl;
    }

    void createConstantVelocityModel(double vel_noise_x, double vel_noise_y, double vel_noise_z) {
        cvm = new CVModel3D(vel_noise_x, vel_noise_y, vel_noise_z);
    }

    void createStaticModel() {
        cvm = new StaticModel3D();
    }

    void addDetectorModel(string name, association_t alg, double pos_noise_x, double pos_noise_y, double pos_noise_z) {
        ROS_INFO("Adding detector model for: %s.", name.c_str());
        detector_model det;
        det.alg = alg;
        det.ctm = new CartesianModel3D(pos_noise_x, pos_noise_y, pos_noise_y);
        detectors[name] = det;
    }

    PoseWithCovariance::_covariance_type getObservationCovariance(string name) {

        detector_model det = detectors[name];
        PoseWithCovariance::_covariance_type covariance;

        // geometry_msgs covariance matrix
        FM::SymMatrix ros_covariance(ROS_COVARIANCE_SIZE, ROS_COVARIANCE_SIZE);
        ros_covariance.clear();

        // geometry_msgs's covariance is a 6×6 matrix, with components for x, y, z, roll, pitch, yaw; Only x, y, z components are used here
        for (size_t i = 0; i < OBSERVATION_SPACE_SIZE; i++)
            for (size_t j = 0; j < OBSERVATION_SPACE_SIZE; j++)
                ros_covariance(i, j) = det.ctm->Z(i, j);

        // Flatten the geometry_msgs covariance matrix
        size_t k = 0;
        for (size_t i = 0; i < ROS_COVARIANCE_SIZE; i++)
            for (size_t j = 0; j < ROS_COVARIANCE_SIZE; j++, k++)
                covariance[k] = ros_covariance(i, j);

        return covariance;
    }

    void addObservation(string detector_name, RecognizedObject3DList observations,
                        PoseWithCovarianceArray &observations_with_covariance) {

        boost::mutex::scoped_lock lock(mutex);

        if (observations.objects.empty()) return;

        detector_model det;
        try {
            det = detectors.at(detector_name);
        } catch (out_of_range &exc) {
            ROS_ERROR("Detector %s was not registered!", detector_name.c_str());
            return;
        }

        dt = getTime() - time;
        time += dt;

        // prediction
        cvm->update(dt);
        mtrk.template predict<CVModel3D>(*cvm);
//        mtrk.process(*(det.ctm), det.alg);
        mtrk.process(*(det.ctm), det.alg, sequence_size, sequence_time);

        PoseWithCovariance::_covariance_type constant_covariance;
        observations_with_covariance.header.stamp = observations.header.stamp;
        observations_with_covariance.header.frame_id = target_frame;

        for (auto observation : observations.objects) {

            auto inverse_observation = observation;
            PoseStamped object_in_target_frame;
            PoseWithCovariance observation_with_covariance;

            try {
                object_in_target_frame = transformPose(observation.pose, observations.header, target_frame);

                // reset the orientation of the object in target coordinates
                // (objects do not have an intrinsic orientations)
                object_in_target_frame.pose.orientation.w = 1;
                object_in_target_frame.pose.orientation.x = 0;
                object_in_target_frame.pose.orientation.y = 0;
                object_in_target_frame.pose.orientation.z = 0;

                // transform the pose back to the camera frame with the correct orientation
                // TODO: probably would be better to do this in localization node
                // TODO: and consider that objects have intrinsic orientation
                PoseStamped object_in_observer_frame = transformPose(object_in_target_frame,
                                                                     observations.header.frame_id);

                // obtain the relative position of the observer in the object's frame by inverting the transform from
                // the observer to the object.
                Point observer_in_object_frame;
                tf::Pose object_to_observer_transform;
                tf::Stamped<tf::Pose> observer_to_object_transform;

                tf::poseStampedMsgToTF(object_in_observer_frame, observer_to_object_transform);
                object_to_observer_transform = observer_to_object_transform.inverse();
                observer_in_object_frame.x = object_to_observer_transform.getOrigin().x();
                observer_in_object_frame.y = object_to_observer_transform.getOrigin().y();
                observer_in_object_frame.z = object_to_observer_transform.getOrigin().z();

                inverse_observation.pose.position = observer_in_object_frame;

                observation_with_covariance.pose = object_in_target_frame.pose;
                // TODO transform covariance
                observation_with_covariance.covariance = getObservationCovariance(detector_name);
                observations_with_covariance.poses.push_back(observation_with_covariance);

            } catch (tf::TransformException ex) {
                ROS_WARN("Failed transform: %s", ex.what());
                continue;
            }

            inverse_observations_history[next_observation_id] = inverse_observation;

            FM::Vec p(3);
            p[0] = object_in_target_frame.pose.position.x;
            p[1] = object_in_target_frame.pose.position.y;
            p[2] = object_in_target_frame.pose.position.z;

            // TODO create addObservationWithCovariance
            mtrk.addObservation(p, observations.header.stamp.toSec(), next_observation_id++, observation.class_name);
//            mtrk.addObservation(p, observations.header.stamp.toSec(), next_observation_id++, "");

        }

    }

    vector<TrackedObject3D> track(double *track_time = NULL) {
        boost::mutex::scoped_lock lock(mutex);
        dt = getTime() - time;
        time += dt;
        if (track_time) *track_time = time;

        for (typename map<string, detector_model>::const_iterator it = detectors.begin(); it != detectors.end(); ++it) {

            // prediction
            cvm->update(dt);
            mtrk.template predict<CVModel3D>(*cvm);

            // process observations, update tracks and obtain the assignment of each observation to the tracks
            map<long, int> assignments;
            mtrk.process(assignments, *(it->second.ctm), it->second.alg, sequence_size, sequence_time);

            for (auto a : assignments) {
                map<string, int> &class_samples = mtrk[a.second].filter->class_samples;
                string c = inverse_observations_history[a.first].class_name;
                auto &inverse_observation = inverse_observations_history[a.first];

                // decide whether to sample the detection, based on the position with respect to the camera
                if (euclidean_distance(mtrk[a.second].filter->last_sampled_detection,
                                       inverse_observation.pose.position) > detection_sampling_radius) {
                    mtrk[a.second].filter->last_sampled_detection = inverse_observation.pose.position;
                    if (class_samples.find(c) != class_samples.end())
                        class_samples[c]++;
                    else
                        class_samples[c] = 1;
                }
            }
        }

        vector<TrackedObject3D> tracked_objects;
        bool tracks_without_observations = false;

        for (size_t i = 0; i < mtrk.size(); i++) {
            auto &filter = mtrk[i].filter;
            map<string, int> &s = filter->class_samples;
            map<string, double> &l = filter->class_likelihood;
            map<string, float> &p = filter->class_probability;

            auto class_names_plus_epsilon = class_names;
            class_names_plus_epsilon.push_back("epsilon");

            size_t observations_count = 0;
            for (auto &c : class_names_plus_epsilon) observations_count += s[c];
            if(observations_count == 0){
                tracks_without_observations = true;
                continue;
            }
            
            for (size_t ci = 0; ci < class_names.size() + 1; ci++) {
                auto &c = class_names_plus_epsilon[ci];
                l[c] = c == "epsilon" ? epsilon_prior : 1 - epsilon_prior;
                for (size_t zi = 0; zi < class_names.size(); zi++) {
                    auto &z = class_names[zi];
                    l[c] *= pow(B[ci][zi], s[z]); // l(c) = Π p(z|c)^s(z) for each class z
                }
            }

            float sum = 0.0;
            for (auto &c : class_names_plus_epsilon) sum += l[c];
            for (auto &c : class_names_plus_epsilon) p[c] = l[c] / sum;

            if(print_debug_info)
                ROS_INFO("Track %lu \nPosition: %f, %f, %f, Std Deviation: %f, %f, %f",
                     mtrk[i].id,
                     filter->x[0], filter->x[2], filter->x[4], // x, y
                     sqrt(filter->X(0, 0)), sqrt(filter->X(2, 2)), sqrt(filter->X(4, 4)) // std dev
                );

            if(print_debug_info) {
                for (auto &c : class_names_plus_epsilon)
                    if (s[c] || c == "epsilon")
                        cout << "s: " << s[c] << ", \tl: " << l[c] << ", \tp: " << p[c] << "\t" << c << endl;

//                for (auto &c : class_names_plus_epsilon)
//                     cout << "s: " << s[c] << ", \tl: "  << l[c] << ", \tp: " << p[c] << "\t" << c << endl;
//                cout << endl << endl;
            }

            TrackedObject3D tracked_object;
            double theta = atan2(filter->x[3], filter->x[1]);

            auto uuid = generateUUID(startup_time_str, mtrk[i].id);
            uuid_map[uuid] = mtrk[i].id;
            tracked_object.uuid = uuid;
            tracked_object.class_name = class_names_plus_epsilon;
            for (auto c : class_names_plus_epsilon) tracked_object.class_probability.push_back(p[c]);
            for (auto c : class_names_plus_epsilon) tracked_object.class_likelihood.push_back(l[c]);
            tracked_object.confidence = 1.0; // TODO
            tracked_object.pose.pose.position.x = filter->x[0];
            tracked_object.pose.pose.position.y = filter->x[2];
            tracked_object.pose.pose.position.z = filter->x[4];
            tracked_object.pose.pose.orientation.z = sin(theta / 2);
            tracked_object.pose.pose.orientation.w = cos(theta / 2);
            tracked_object.pose.pose.orientation.w = cos(theta / 2);
            tracked_object.velocity.x = filter->x[1];
            tracked_object.velocity.y = filter->x[3];
            tracked_object.velocity.z = filter->x[5];

            // geometry_msgs covariance matrix
            FM::SymMatrix ros_covariance(ROS_COVARIANCE_SIZE, ROS_COVARIANCE_SIZE);
            ros_covariance.clear();

            // geometry_msgs's covariance is a 6×6 matrix, with components for x, y, z, roll, pitch, yaw; Only x, y, z components are used here
            for (size_t i = 0; i < STATE_SPACE_SIZE; i += 2)
                for (size_t j = 0; j < STATE_SPACE_SIZE; j += 2) {
                    ros_covariance(i / 2, j / 2) = filter->X(i, j);
                }

            // Flatten the geometry_msgs covariance matrix and fill the message's array
            size_t k = 0;
            for (size_t i = 0; i < ROS_COVARIANCE_SIZE; i++)
                for (size_t j = 0; j < ROS_COVARIANCE_SIZE; j++, k++) {
                    tracked_object.pose.covariance[k] = ros_covariance(i, j);
                }

            tracked_objects.push_back(tracked_object);
        }
        
        if(tracks_without_observations) ROS_DEBUG("Tracks with no class observations");

        return tracked_objects;
    }

    void deleteTrack(string uuid){
        if (uuid_map.find(uuid) != uuid_map.end()){
            auto filter_id = uuid_map[uuid];
            mtrk.deleteTrack(filter_id);
            ROS_INFO_STREAM("Deleted track. uuid: " << uuid << "\t track id: " << filter_id);
        } else {
            ROS_WARN_STREAM("Requested to delete track, but track didn't exists. uuid: " << uuid);
        }
    }

private:

    boost::mutex mutex;
    vector<vector<float>> B;
    double epsilon_prior = 0;
    vector<string> &class_names;
    unordered_map<long unsigned, RecognizedObject3D> inverse_observations_history; // TODO delete old observations
    long unsigned next_observation_id = 0;
    string &target_frame;
    tf::TransformListener *listener;
    unsigned int sequence_size;
    double sequence_time;
    bool print_debug_info;

    CVModel3D *cvm{};                   // 3D CV or static model
    double dt{}, time;
    std::string startup_time_str;
    boost::uuids::uuid dns_namespace_uuid;
    unordered_map<std::string, long unsigned int> uuid_map;

    MultiTracker<FilterType, STATE_SPACE_SIZE> mtrk = MultiTracker<FilterType, STATE_SPACE_SIZE>();   // state [x, v_x, y, v_y, z, v_z]

    typedef struct {
        CartesianModel3D *ctm;          // 3D Cartesian observation model
        association_t alg;              // Data association algorithm
    } detector_model;

    map<string, detector_model> detectors;
    float detection_sampling_radius;
};

#endif //SIMPLE_TRACKING_H
