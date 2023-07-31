#ifndef ROSNEURO_DECODER_GAUSSIAN_H
#define ROSNEURO_DECODER_GAUSSIAN_H

#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>

#include "rosneuro_decoder/GenericDecoder.h"

namespace rosneuro{
    namespace decoder{

        typedef struct {

        	std::string 		   filename;

        	std::string		       subject;
        	std::uint32_t		   nclasses;
        	std::vector<uint32_t>  classlbs;
        	std::uint32_t		   nprototypes;
        	std::uint32_t      	   nfeatures;

            // for features extraction
        	std::vector<uint32_t>               idchans;
        	std::vector<std::vector<uint32_t>>  freqs;

        } gauconfig_t;
        

        class Gaussian : public GenericDecoder{
            public:
                Gaussian(void);
                ~Gaussian(void);
                bool configure(void);
                bool isSet(void);
                Eigen::VectorXf apply(const Eigen::VectorXf& in);
		        Eigen::VectorXf getFeatures(const Eigen::MatrixXf& in);
                std::string path(void);
		        std::vector<int> classes(void);

            private:
                bool check_dimension(void);

            private:
		        gauconfig_t config_;
                ros::NodeHandle p_nh_;
                Eigen::MatrixXf centers_; // [(features*nclasses) x nprototypes]
                Eigen::MatrixXf covs_; // [(features*nclasses) x nprototypes]
        };
    }
}

#endif