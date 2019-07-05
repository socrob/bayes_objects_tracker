//
// Created by enrico on 08/09/18.
//

#ifndef BAYES_OBJECTS_TRACKER_CLASSDATA_H
#define BAYES_OBJECTS_TRACKER_CLASSDATA_H

#include <map>
#include <geometry_msgs/Point.h>

class FilterInfo {

public:
    std::map<std::string, int> class_samples;
    std::map<std::string, double> class_likelihood;
    std::map<std::string, float> class_probability;
    geometry_msgs::Point last_sampled_detection; // TODO index by detector id, so that multiple detectors can be used

    FilterInfo() = default;
    virtual ~FilterInfo() = default;

};


#endif //BAYES_OBJECTS_TRACKER_CLASSDATA_H
