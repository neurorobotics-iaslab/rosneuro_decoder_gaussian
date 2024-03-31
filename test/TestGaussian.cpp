#include "rosneuro_decoder_gaussian/Gaussian.h"
#include <gtest/gtest.h>

namespace rosneuro {
namespace decoder {

class GaussianTestSuite : public ::testing::Test {
public:
    GaussianTestSuite() {}
    ~GaussianTestSuite() {}
    void SetUp(void) {
        gaussian = new Gaussian();
    }
    void TearDown(void) {
        delete gaussian;
    }
    Gaussian* gaussian;
};


void config_gaussian(std::map<std::string, XmlRpc::XmlRpcValue>& params) {
        params["name"] = "gaussian";
        params["filename"] = "file1";
        params["subject"] = "S1";
        params["nclasses"] = 2;

        XmlRpc::XmlRpcValue classlbs;
        classlbs[0] = 771;
        classlbs[1] = 773;
        params["classlbs"] = classlbs;

        params["nprototypes"] = 1;
        params["nfeatures"] = 6;

        XmlRpc::XmlRpcValue idchans;
        idchans[0] = 1;
        idchans[1] = 2;
        params["idchans"] = idchans;

        params["freqs"] = "10 12 14; 10 12 14";
        params["covs"] = "0.4529; 0.3777; 0.5301;"
                         "0.4910; 0.7136; 0.7507;"
                         "1.0767; 1.1501; 0.7055;"
                         "0.6599; 1.2610;1.1729;";

        params["centers"] = "-2.9844;-3.2610;-2.3018;"
                            "-2.5743;-2.3550;-2.5197;"
                            "-2.1708;-2.3625;-1.8599;"
                            "-2.2393;-1.5272;-1.8280;";
}


TEST_F(GaussianTestSuite, Constructor) {
    EXPECT_EQ(gaussian->is_configured_, false);
    EXPECT_EQ(gaussian->name(), "gaussian");
}

TEST_F(GaussianTestSuite, Path) {
    gaussian->is_configured_ = true;
    EXPECT_EQ(gaussian->path(), "");
}

TEST_F(GaussianTestSuite, Classes) {
    gaussian->config_.classlbs = {1, 2, 3};
    EXPECT_EQ(gaussian->classes(), std::vector<int>({1, 2, 3}));
}

TEST_F(GaussianTestSuite, Configure) {
    config_gaussian(gaussian->params_);
    ASSERT_TRUE(gaussian->configure());
}

TEST_F(GaussianTestSuite, CheckDimensionSize) {
    config_gaussian(gaussian->params_);
    XmlRpc::XmlRpcValue idchans;
    idchans[0] = 1;
    gaussian->params_["idchans"] = idchans;
    ASSERT_FALSE(gaussian->configure());
}

TEST_F(GaussianTestSuite, CheckDimensionFreq) {
    config_gaussian(gaussian->params_);
    gaussian->params_["nfeatures"] = 1;
    ASSERT_FALSE(gaussian->configure());
}

TEST_F(GaussianTestSuite, GetFeatures) {
    gaussian->config_.nfeatures = 3;
    gaussian->config_.idchans = {1};
    gaussian->config_.freqs = {{1, 2, 3}};

    Eigen::MatrixXf in(1, 3);
    in << 1.0, 2.0, 3.0;

    Eigen::MatrixXf expected(3, 1);
    expected << 1, 2, 2;

    ASSERT_EQ(gaussian->getFeatures(in), expected);
}

TEST_F(GaussianTestSuite, Apply) {
    config_gaussian(gaussian->params_);
    gaussian->configure();

    Eigen::VectorXf in(6);
    in << -2.9844, -3.2610, -2.3018, -2.5743, -2.3550, -2.5197;

    Eigen::VectorXf expected(2);
    expected << 0.95874, 0.0412597;

    ASSERT_TRUE(gaussian->apply(in).isApprox(expected, 1e-4));
}

}
}

int main(int argc, char **argv) {
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
    ros::init(argc, argv, "test_gaussian");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}