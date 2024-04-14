#ifndef ROSNEURO_DECODER_GAUSSIAN_CPP
#define ROSNEURO_DECODER_GAUSSIAN_CPP

#include "rosneuro_decoder_gaussian/Gaussian.h"

namespace rosneuro{
    namespace decoder{
        Gaussian::Gaussian(void) : p_nh_("~"){
            this->setName("gaussian");
            this->is_configured_ = false;
        }

        Gaussian::~Gaussian(void){}

        std::string Gaussian::getPath(void){
            this->isSet();
            return this->config_.filename;
        }

        std::vector<int> Gaussian::getClasses(void){
            this->isSet();
            std::vector<int> classes_lbs;
            for(int i = 0; i < this->config_.class_lbs.size(); i++){
                classes_lbs.push_back((int) this->config_.class_lbs.at(i));
            }
            return classes_lbs;
        }

        bool Gaussian::isSet(void){
            if(!this->is_configured_){
                ROS_ERROR("[%s] Decoder not configured", this->getName().c_str());
                return false;
            }
            return this->is_configured_;
        }

        template<typename T>
        bool Gaussian::getParamAndCheck(const std::string& param_name, T& param_value) {
            if (!GenericDecoder::getParam(param_name, param_value)) {
                ROS_ERROR("[%s] Cannot find param '%s'", this->getName().c_str(), param_name.c_str());
                return false;
            }
            return true;
        }

        bool Gaussian::configure(void){
            if (!getParamAndCheck("filename", this->config_.filename)) return false;
            if (!getParamAndCheck("subject", this->config_.subject)) return false;
            if (!getParamAndCheck("n_classes", this->config_.n_classes)) return false;
            if (!getParamAndCheck("class_lbs", this->config_.class_lbs)) return false;
            if (!getParamAndCheck("n_prototypes", this->config_.n_prototypes)) return false;
            if (!getParamAndCheck("n_features", this->config_.n_features)) return false;
            if (!getParamAndCheck("idchans", this->config_.idchans)) return false;

            std::string freqs_str, centers_str, covs_str;
            if (!getParamAndCheck("freqs", freqs_str)) return false;
            if(!this->loadVectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vector of vector", this->getName().c_str());
                return false;
            }

            if (!getParamAndCheck("centers", centers_str)) return false;
            this->centers_ = Eigen::MatrixXf::Zero(this->config_.n_features*this->config_.n_classes, this->config_.n_prototypes);
            if(!this->loadEigen(centers_str, this->centers_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for centers", this->getName().c_str());
                return false;
            }

            if (!getParamAndCheck("covs", covs_str)) return false;
            this->covs_ = Eigen::MatrixXf::Zero(this->config_.n_features*this->config_.n_classes, this->config_.n_prototypes);
            if(!this->loadEigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->getName().c_str());
                return false;
            }

            if(!this->checkDimension()){
                ROS_ERROR("[%s] Error in the dimension", this->getName().c_str());
                return false;
            }

            this->is_configured_ = true;
            return this->is_configured_;
        }

        Eigen::VectorXf Gaussian::apply(const Eigen::VectorXf& input) {
            Eigen::VectorXf output(this->config_.n_classes);
            if (input.size() != this->config_.n_features) {
                ROS_ERROR("[%s] input incorrect dimension", getName().c_str());
                return output;
            }

            unsigned int modelRows = this->config_.n_classes;
            unsigned int modelCols = this->config_.n_prototypes;
            unsigned int modelFeats = input.size();

            Eigen::MatrixXf activities = computeActivities(input, modelRows, modelCols, modelFeats);
            const Eigen::VectorXf rawProbs = computeRawProbabilities(activities);
            updateOutput(output, rawProbs, rawProbs.sum(), modelRows);

            return output;
        }

        Eigen::MatrixXf Gaussian::computeActivities(const Eigen::VectorXf& input, unsigned int modelRows, unsigned int modelCols, unsigned int modelFeats) {
            Eigen::MatrixXf activities = Eigen::MatrixXf::Zero(modelRows, modelCols);

            for (unsigned int i = 0; i < modelRows; ++i) {
                const Eigen::MatrixXf subModelM = this->centers_.block(i * modelFeats, 0, modelFeats, modelCols);
                const Eigen::MatrixXf subModelC = this->covs_.block(i * modelFeats, 0, modelFeats, modelCols);

                for (unsigned int j = 0; j < modelCols; ++j) {
                    const Eigen::VectorXf M = subModelM.col(j);
                    const Eigen::VectorXf C = subModelC.col(j);

                    const Eigen::VectorXf distance = computeDistance(input, M, C);
                    const Eigen::VectorXf tempDet = computeTempDet(C);
                    const double determinant = tempDet.prod();
                    const double distanceSum = distance.sum();

                    activities(i, j) = exp(-(distanceSum / 2.0)) / determinant;
                }
            }

            return activities;
        }

        Eigen::VectorXf Gaussian::computeDistance(const Eigen::VectorXf& input, const Eigen::VectorXf& M, const Eigen::VectorXf& C) {
            return (input - M).array().pow(2) / C.array();
        }

        Eigen::VectorXf Gaussian::computeTempDet(const Eigen::VectorXf& C) {
            return C.array().sqrt().unaryExpr([](float val) { return val == 0 ? 1 : val; });
        }

        Eigen::VectorXf Gaussian::computeRawProbabilities(const Eigen::MatrixXf& activities) {
            return activities.rowwise().sum();
        }

        void Gaussian::updateOutput(Eigen::VectorXf& output, const Eigen::VectorXf& rawProbs, double sumProbs, unsigned int modelRows) {
            if (sumProbs == 0) {
                output.setConstant(1.0 / modelRows);
            } else {
                output = rawProbs / sumProbs;
            }
        }

		Eigen::VectorXf Gaussian::getFeatures(const Eigen::MatrixXf& in){
            Eigen::VectorXf out(this->config_.n_features);
            this->isSet();

            int c_feature = 0;
            for(int it_chan = 0; it_chan < this->config_.idchans.size(); it_chan++){
                int id_chan = this->config_.idchans.at(it_chan) - 1;
                for(const auto& freq : this->config_.freqs.at(it_chan)){
                    out(c_feature) = in(id_chan, (int) freq/2.0);
                    c_feature ++;
                }
            }
            return out.transpose();
        }

        bool Gaussian::checkDimension() {
            const auto& idchans = this->config_.idchans;
            const auto& freqs = this->config_.freqs;
            const auto& n_features = this->config_.n_features;

            if (idchans.size() != freqs.size()) {
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' and 'idchans'", this->getName().c_str());
                return false;
            }

            int totalFreqs = 0;
            for (const auto& freq : freqs) {
                totalFreqs += freq.size();
            }

            if (totalFreqs != n_features) {
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' different from 'n_features'", this->getName().c_str());
                return false;
            }

            return true;
        }

        PLUGINLIB_EXPORT_CLASS(rosneuro::decoder::Gaussian, rosneuro::decoder::GenericDecoder);
    }
}
#endif