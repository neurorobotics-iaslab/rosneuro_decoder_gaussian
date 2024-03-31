cmake_minimum_required(VERSION 3.0.2)
project(rosneuro_decoder_gaussian)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(ROSNEURO_DECODER_MIN_VERSION "1.0.0")

##############################################################################
# Find dependencies
##############################################################################

find_package(catkin REQUIRED COMPONENTS 
			 roscpp 
			 std_msgs
			 pluginlib
			 rosneuro_msgs
			 rosneuro_decoder)

find_package(Eigen3 REQUIRED)
find_package(rosneuro_decoder ${ROSNEURO_DECODER_MIN_VERSION} REQUIRED)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

##############################################################################
# Define package
##############################################################################
catkin_package(
  INCLUDE_DIRS 
  	include
  LIBRARIES 
  ${PROJECT_NAME}
	
  CATKIN_DEPENDS
  	roscpp
	std_msgs
	pluginlib
	rosneuro_msgs
	rosneuro_decoder
  DEPENDS
)

##############################################################################
# TEST
##############################################################################

catkin_add_gtest(TestGaussian test/TestGaussian.cpp)
target_link_libraries(TestGaussian ${GTEST_BOTH_LIBRARIES} pthread gmock)
target_link_libraries(TestGaussian ${PROJECT_NAME} ${catkin_LIBRARIES})

##############################################################################
# Build
##############################################################################

include_directories(${catkin_INCLUDE_DIRS} 
					${rosneuro_decoder_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
					include)

add_definitions(${EIGEN3_DEFINITIONS})

add_library(${PROJECT_NAME} src/Gaussian.cpp)

target_link_libraries(${PROJECT_NAME} ${Eigen3_LIBRARIES} 
									  ${rosneuro_gaussian_LIBRARIES}
									  ${catkin_LIBRARIES})

add_dependencies(${PROJECT_NAME} 
				 ${catkin_EXPORTED_TARGETS}
)

add_executable(test_gaussian test/test_gaussian.cpp)
target_link_libraries(test_gaussian ${catkin_LIBRARIES} fftw3)

#################
##   Install   ##
#################
install(TARGETS ${PROJECT_NAME}

	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(FILES plugin_decoder_gaussian.xml
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
