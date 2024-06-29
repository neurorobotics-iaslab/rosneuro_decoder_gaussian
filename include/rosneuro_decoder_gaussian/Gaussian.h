#ifndef ROSNEURO_DECODER_GAUSSIAN_H
#define ROSNEURO_DECODER_GAUSSIAN_H

#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>
#include <gtest/gtest_prod.h>
#include <unordered_map>
#include "rosneuro_decoder/GenericDecoder.h"

namespace rosneuro{
    namespace decoder{

        typedef struct {
        	std::string 		   filename;
        	std::string		       subject;
        	std::uint32_t		   n_classes;
        	std::vector<uint32_t>  class_lbs;
        	std::uint32_t		   n_prototypes;
        	std::uint32_t      	   n_features;

        	std::vector<uint32_t>               idchans;
        	std::vector<std::vector<uint32_t>>  freqs;

        } gaussian_config;
        

        class Gaussian : public GenericDecoder{
            public:
                Gaussian(void);
                ~Gaussian(void);

                bool configure(void);
                Eigen::VectorXf apply(const Eigen::VectorXf& in);
		        Eigen::VectorXf getFeatures(const Eigen::MatrixXf& in);
                std::string getPath(void);
		        std::vector<int> getClasses(void);

            private:
                std::tuple<unsigned int, unsigned int, unsigned int> getModelParameters();
                Eigen::MatrixXf computeActivities(const Eigen::VectorXf& input, unsigned int modelRows, unsigned int modelCols, unsigned int modelFeats);
                Eigen::VectorXf computeDistance(const Eigen::VectorXf& input, const Eigen::VectorXf& M, const Eigen::VectorXf& C);
                Eigen::VectorXf computeTempDet(const Eigen::VectorXf& C);
                Eigen::VectorXf computeRawProbabilities(const Eigen::MatrixXf& activities);
                void updateOutput(Eigen::VectorXf& output, const Eigen::VectorXf& rawProbs, double sumProbs, unsigned int modelRows);
                bool checkDimension(void);
                template<typename T>
                bool getParamAndCheck(const std::string& param_name, T& param_value);

                gaussian_config config_;
                ros::NodeHandle p_nh_;
                Eigen::MatrixXf centers_;
                Eigen::MatrixXf covs_;

                FRIEND_TEST(GaussianTestSuite, Constructor);
                FRIEND_TEST(GaussianTestSuite, Path);
                FRIEND_TEST(GaussianTestSuite, Classes);
                FRIEND_TEST(GaussianTestSuite, Configure);
                FRIEND_TEST(GaussianTestSuite, CheckDimensionSize);
                FRIEND_TEST(GaussianTestSuite, CheckDimensionFreq);
                FRIEND_TEST(GaussianTestSuite, GetFeatures);
                FRIEND_TEST(GaussianTestSuite, Apply);
                FRIEND_TEST(GaussianTestSuite, Integration);
        };
    }
}

#endif