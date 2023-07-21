#ifndef ROSNEURO_DECODER_GAUSSIAN_CPP
#define ROSNEURO_DECODER_GAUSSIAN_CPP

#include "rosneuro_decoder_gaussian/Gaussian.h"

namespace rosneuro{
    namespace decoder{
        Gaussian::Gaussian(void) : p_nh_("~"){
            this->setname("gaussian");
            this->is_configured_ = false;
        }

        Gaussian::~Gaussian(void){}

        std::string Gaussian::path(void){
            this->isSet();
            return this->config_.filename;
        }

        std::vector<int> Gaussian::classes(void){
            this->isSet();
            std::vector<int> classeslbs;
            for(int i = 0; i < this->config_.classlbs.size(); i++){
                classeslbs.push_back((int) this->config_.classlbs.at(i));
            }
            return classeslbs;
        }

        bool Gaussian::isSet(void){
            if(this->is_configured_ == false){
                ROS_ERROR("[%s] Decoder not configured", this->name().c_str());
                return false;
            }
            
            return this->is_configured_;
        }

        bool Gaussian::configure(void){
            // get the parameters
            if(!GenericDecoder::getParam(std::string("filename"), this->config_.filename)){
                ROS_ERROR("[%s] Cannot find param 'filename'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("subject"), this->config_.subject)){
                ROS_ERROR("[%s] Cannot find param 'subject'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("nclasses"), this->config_.nclasses)){
                ROS_ERROR("[%s] Cannot find param 'nclasses'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("classlbs"), this->config_.classlbs)){
                ROS_ERROR("[%s] Cannot find param 'classlbs'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("nprototypes"), this->config_.nprototypes)){
                ROS_ERROR("[%s] Cannot find param 'nprototypes'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("nfeatures"), this->config_.nfeatures)){
                ROS_ERROR("[%s] Cannot find param 'nfeatures'", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("idchans"), this->config_.idchans)){
                ROS_ERROR("[%s] Cannot find param 'idchans'", this->name().c_str());
                return false;
            }
            std::string freqs_str;
            if(!GenericDecoder::getParam(std::string("freqs"), freqs_str)){
                ROS_ERROR("[%s] Cannot find param 'freqs'", this->name().c_str());
                return false;
            }
            if(!this->load_vectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vctor of vector", this->name().c_str());
                return false;
            }
            std::string centers_str, covs_str;
            if(!GenericDecoder::getParam(std::string("centers"), centers_str)){
                ROS_ERROR("[%s] Cannot find param 'centers'", this->name().c_str());
                return false;
            }
            this->centers_ = Eigen::MatrixXf::Zero(this->config_.nfeatures*this->config_.nclasses, this->config_.nprototypes);
            if(!this->load_eigen(centers_str, this->centers_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for centers", this->name().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("covs"), covs_str)){
                ROS_ERROR("[%s] Cannot find param 'covs'", this->name().c_str());
                return false;
            }
            this->covs_ = Eigen::MatrixXf::Zero(this->config_.nfeatures*this->config_.nclasses, this->config_.nprototypes);
            if(!this->load_eigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->name().c_str());
                return false;
            }

            // fast check for the correct dimension for the features
            if(!this->check_dimension()){
                ROS_ERROR("[%s] Error in the dimension", this->name().c_str());
                return false;
            }

            this->is_configured_ = true;

            return this->is_configured_;
        }

        Eigen::VectorXf Gaussian::apply(const Eigen::VectorXf& in){

            // check if set and correct dimension input
            Eigen::VectorXf out(this->config_.nclasses);
            this->isSet();
            if(in.size() != this->config_.nfeatures){
                ROS_ERROR("[%s] input incorrect dimension", this->name().c_str());
            }

            // create variables
            unsigned int modelRows  = this->config_.nclasses;
            unsigned int modelCols  = this->config_.nprototypes;
            unsigned int modelFeats = in.size();

            Eigen::MatrixXf subModelM = Eigen::MatrixXf::Zero(modelFeats, modelCols);
            Eigen::MatrixXf subModelC = Eigen::MatrixXf::Zero(modelFeats, modelCols);

            Eigen::VectorXf M        = Eigen::VectorXf::Zero(modelFeats);
            Eigen::VectorXf C        = Eigen::VectorXf::Zero(modelFeats);
            Eigen::VectorXf distance = Eigen::VectorXf::Zero(modelFeats);
            Eigen::VectorXf tempDet  = Eigen::VectorXf::Zero(modelFeats);

            double determinant = 0.0;
            double distanceSum = 0.0;
            double sumProbs    = 0.0;

            Eigen::MatrixXf activities = Eigen::MatrixXf::Zero(modelRows, modelCols);
            Eigen::VectorXf rawProbs = Eigen::VectorXf::Zero(modelRows);

            // iterate over classes
            for(unsigned int i = 0; i < modelRows; i++){
                subModelM = this->centers_.block(i*modelFeats, 0, modelFeats, modelCols);
                subModelC = this->covs_.block(i*modelFeats, 0, modelFeats, modelCols);

                // iterate over prototypes
                for(unsigned int j = 0; j < modelCols; j++){
                    M = subModelM.col(j);
                    C = subModelC.col(j);

                    distance = (in - M).array().pow(2) / C.array();
                    tempDet  = C.array().sqrt();

                    for(unsigned int u = 0; u < tempDet.size(); u++){
                        if(tempDet(u) == 0){
                            tempDet(u) = 1;
                        }
                    }

                    determinant = tempDet.array().prod();

                    distanceSum = distance.array().sum();

                    activities(i,j) = exp(-(distanceSum/2.0)) / determinant;
                
                }
            }

            rawProbs = activities.rowwise().sum();
            
            sumProbs = rawProbs.array().sum();

            if(sumProbs == 0){
                out = rawProbs.array().setOnes() / modelRows;
            }else{
                out = rawProbs.array() / sumProbs;
            }


            return out;
        }
        
		Eigen::VectorXf Gaussian::getFeatures(const Eigen::MatrixXf& in){
            // check if set and prepare for the correct dimension
            Eigen::VectorXf out(this->config_.nfeatures);
            this->isSet();

            // iterate over channels
            int c_feature = 0;
            for(int it_chan = 0; it_chan < this->config_.idchans.size(); it_chan++){
                int idchan = this->config_.idchans.at(it_chan) - 1; // -1 bc: channels starts from 1 and not 0
                // iterate over freqs for that channel
                for(const auto& freq : this->config_.freqs.at(it_chan)){
                    // we have the freq value and not the id of that freq
                    int idfreq = (int) freq/2.0;
                    out(c_feature) = in(idchan, idfreq);
                    c_feature ++;
                }
            }

            return out.transpose();
        }


        

        // fast check for the features used by the decoder
        bool Gaussian::check_dimension(void){
            if(this->config_.idchans.size() != this->config_.freqs.size()){
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' and 'idchans'", this->name().c_str());
                return false;
            }

            int sum = 0;
            for(int i = 0; i < this->config_.freqs.size(); i++){
                std::vector<uint32_t> temp = this->config_.freqs.at(i);
                sum += temp.size();
            }
            if(sum != this->config_.nfeatures){
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' different from 'nfeatures'", this->name().c_str());
                return false;
            }

            return true;
        }

        
PLUGINLIB_EXPORT_CLASS(rosneuro::decoder::Gaussian, rosneuro::decoder::GenericDecoder);
    }
}
#endif