/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#include <SLAMBenchAPI.h>

#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/DepthSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <timings.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

//Headers required for ORB_SLAM2
#include <opencv2/core/core.hpp>
#include <System.h>
#include <FrameDrawer.h>
#include <MapDrawer.h>
#include <Tracking.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <Converter.h>
#include <io/sensor/GroundTruthSensor.h>

//access to slam objects 
static cv::Mat pose;
static cv::Mat frameCV;
static sb_uint2 inputSize;
static int frame_no = 0;


static cv::Mat result_tracking;
static cv::Mat* imRGB = NULL, * imD = NULL;

cv::Mat* img_one;
cv::Mat* img_two;

double tframe;

enum orbslam_input_mode {mono,stereo,rgbd,automatic};

static orbslam_input_mode input_mode;
static const orbslam_input_mode default_input_mode = orbslam_input_mode::automatic;

static std::string settings_file;
static std::string vocabulary_file;


static slambench::outputs::Output *pose_output;
static slambench::outputs::Output *pointcloud_output;

static slambench::outputs::Output *frame1_output;
static slambench::outputs::Output *frame2_output;


static slambench::TimeStamp last_frame_timestamp;

static const std::string default_settings_file = "";
static const std::string default_vocabulary_file = "./benchmarks/orbslam2/src/original/Vocabulary/ORBvoc.bin";


ORB_SLAM2::System* SLAM;


// ===========================================================
// SLAMBench Sensors
// ===========================================================

static slambench::io::DepthSensor *depth_sensor;
static slambench::io::CameraSensor *rgb_sensor;

static slambench::io::CameraSensor *grey_sensor_one = nullptr;
static slambench::io::CameraSensor *grey_sensor_two = nullptr;


// ===========================================================
// Variable parameters
// ===========================================================

static int max_features;
static const int default_max_features=1000;
static int pyramid_levels;
static const int default_pyramid_levels=8;
static float scale_factor;
static const float default_scale_factor=1.2;    
static int initial_fast_threshold;
static const int default_initial_fast_threshold=20;
static int second_fast_threshold;
static const int default_second_fast_threshold=7;



static int camera_fps;
static const int default_camera_fps=40;

static float depth_threshold;
static const float default_depth_threshold=40;

bool sb_get_tracked()  {
    return (SLAM->GetTrackingState() == 2);
}


// ===========================================================
// PERSONALIZED DATATYPE FOR ORBSLAM PARAMETERS
// ===========================================================

template<> inline const std::string  TypedParameter<orbslam_input_mode>::getValue(const void * ptr) {
	switch (*((orbslam_input_mode*) ptr))  {
	case orbslam_input_mode::mono : return "mono";
	case orbslam_input_mode::stereo : return "stereo";
	case orbslam_input_mode::rgbd : return "rgbd";
	case orbslam_input_mode::automatic : return "auto";
	}
	return "error";
};

template<> inline void  TypedParameter<orbslam_input_mode>::copyValue(orbslam_input_mode *to , orbslam_input_mode const *from) {
	*to = *from;
};

template<> inline void  TypedParameter<orbslam_input_mode>::setValue(const char* optarg)  {

	if (std::string(optarg) == "auto")
	{*((orbslam_input_mode*)ptr_) = orbslam_input_mode::automatic;}
	else if (std::string(optarg) == "mono")
	{*((orbslam_input_mode*)ptr_) = orbslam_input_mode::mono;}
	else if (std::string(optarg) == "stereo")
	{*((orbslam_input_mode*)ptr_) = orbslam_input_mode::stereo;}
	else if (std::string(optarg) == "rgbd")
	{*((orbslam_input_mode*)ptr_) = orbslam_input_mode::rgbd;}
	else
	{throw std::logic_error("The argument you gave for ORBSLAM Mode is incorrect, only 'auto', 'mono', 'stereo' or 'rgbd' are valid.");}
};


// ===========================================================
// Rectification for Stereo mode
// ===========================================================
cv::Mat M1l,M2l,M1r,M2r;

// ===========================================================
// SLAMBENCH API
// ===========================================================


bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {

	slam_settings->addParameter(TypedParameter<orbslam_input_mode>("m", "mode",     "select input mode (auto,mono,stereo,rgbd)",    &input_mode, &default_input_mode));
	slam_settings->addParameter(TypedParameter<std::string>("", "settings",     "Path to the setting file",    &settings_file, &default_settings_file));
	slam_settings->addParameter(TypedParameter<std::string>("voc", "vocabulary",     "Path to the vocabulary file",    &vocabulary_file, &default_vocabulary_file));

	// algo parameters
	slam_settings->addParameter(TypedParameter<int>("mf", "max-features",     "Maximum number of features",    &max_features, &default_max_features));
	slam_settings->addParameter(TypedParameter<int>("sl", "scale-levels",     "Number of levels in image pyramid",    &pyramid_levels, &default_pyramid_levels));
	slam_settings->addParameter(TypedParameter<float>("sf", "scale-factor",     "Scale between levels in image pyramid",    &scale_factor, &default_scale_factor));
	slam_settings->addParameter(TypedParameter<int>("ift", "initial-fast-threshold",     "Initial threshold of FAST algorithm (high)",    &initial_fast_threshold,
			&default_initial_fast_threshold));
	slam_settings->addParameter(TypedParameter<int>("sft", "second-fast-threshold",     "Second threshold of FAST algorithm (low)",    &second_fast_threshold,
			&default_second_fast_threshold));

	// camera parameters
	slam_settings->addParameter(TypedParameter<int>("fps", "camera-fps",     "???",    &camera_fps,  &default_camera_fps));
	slam_settings->addParameter(TypedParameter<float>("dt", "depth-threshold",     "Depth threshold (close/far points)",    &depth_threshold,  &default_depth_threshold));

	return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)  {


	//=========================================================================
	// We collect sensors
	//=========================================================================

	slambench::io::CameraSensorFinder sensor_finder;
	rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
	depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});
	auto grey_sensors = sensor_finder.Find(slam_settings->get_sensors(), {{"camera_type", "grey"}});

	if(grey_sensors.size() ==  2) {

		grey_sensor_one = grey_sensors.at(0);
		grey_sensor_two = grey_sensors.at(1);


	}

	if ( input_mode == orbslam_input_mode::automatic ) {
	  if (rgb_sensor || grey_sensor_one) input_mode = orbslam_input_mode::mono;
	  if (rgb_sensor && depth_sensor) input_mode = orbslam_input_mode::rgbd;
	  if (grey_sensor_one && grey_sensor_two) input_mode = orbslam_input_mode::stereo;
	}
	  

	//=========================================================================
	// We parametrize ORBSLAM2 given the current mode
	//=========================================================================

	if ( input_mode == orbslam_input_mode::rgbd ) {


		//=========================================================================
		// RGBD Mode
		//=========================================================================

		if ((rgb_sensor == nullptr) or (depth_sensor == nullptr)) {
			std::cerr << "Invalid sensors found, RGB or Depth not found." << std::endl;
			std::cerr << "RGB   = " << (rgb_sensor == nullptr) << std::endl;
			std::cerr << "DEPTH = " << (depth_sensor == nullptr) << std::endl;
			return false;
		}

		if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
			std::cerr << "RGB data is in wrong format" << std::endl;
			return false;
		}
		if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
			std::cerr << "Depth data is in wrong format" << std::endl;
			return false;
		}
		if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
			std::cerr << "RGB data is in wrong format pixel" << std::endl;
			return false;
		}
		if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
			std::cerr << "Depth data is in wrong pixel format" << std::endl;
			return false;
		}


		if(rgb_sensor->Width != depth_sensor->Width || rgb_sensor->Height != depth_sensor->Height) {
			std::cerr << "Sensor size mismatch" << std::endl;
			return false;
		}


		cv::Mat depth_camera_parameters = cv::Mat::eye(3,3,CV_32F);

		if (depth_sensor) {
			depth_camera_parameters.at<float>(0,0) = ((slambench::io::DepthSensor*)depth_sensor)->Intrinsics[0]*((slambench::io::DepthSensor*)depth_sensor)->Width;
			depth_camera_parameters.at<float>(1,1) = ((slambench::io::DepthSensor*)depth_sensor)->Intrinsics[1]*((slambench::io::DepthSensor*)depth_sensor)->Height;
			depth_camera_parameters.at<float>(0,2) =  ((slambench::io::DepthSensor*)depth_sensor)->Intrinsics[2]*((slambench::io::DepthSensor*)depth_sensor)->Width;
			depth_camera_parameters.at<float>(1,2) = ((slambench::io::DepthSensor*)depth_sensor)->Intrinsics[3]*((slambench::io::DepthSensor*)depth_sensor)->Height;
		}



		cv::Mat camera_distortion(5,1,CV_32F);
		if (depth_sensor->DistortionType == slambench::io::CameraSensor::RadialTangential) {
			camera_distortion.at<float>(0) =  depth_sensor->RadialTangentialDistortion[0];
			camera_distortion.at<float>(1) =  depth_sensor->RadialTangentialDistortion[1];
			camera_distortion.at<float>(2) =  depth_sensor->RadialTangentialDistortion[2];
			camera_distortion.at<float>(3) =  depth_sensor->RadialTangentialDistortion[3];
			camera_distortion.at<float>(4) =  depth_sensor->RadialTangentialDistortion[4];
		} else {
			camera_distortion.at<float>(0) = 0.0;
			camera_distortion.at<float>(1) = 0.0;
			camera_distortion.at<float>(2) = 0.0;
			camera_distortion.at<float>(3) = 0.0;
			camera_distortion.at<float>(4) = 0.0;
		}


		imRGB = new cv::Mat ( rgb_sensor->Height ,  rgb_sensor->Width, CV_8UC3);
		imD   = new cv::Mat ( depth_sensor->Height ,  depth_sensor->Width, CV_16UC1);
		inputSize   = make_sb_uint2(rgb_sensor->Width,rgb_sensor->Height);
		float DepthMapFactor = depth_sensor->DisparityParams[0];
		SLAM = new ORB_SLAM2::System (vocabulary_file,settings_file,ORB_SLAM2::System::RGBD,false);

		if(settings_file == "") {
			SLAM->mpTracker->ConfigureCamera(depth_camera_parameters, camera_distortion, camera_fps, DepthMapFactor, 40 , depth_threshold);
			SLAM->mpTracker->ConfigureAlgorithm(max_features,pyramid_levels,scale_factor,initial_fast_threshold,second_fast_threshold);
		}
		SLAM->mpTracker->PrintConfig();

	} else if  ( input_mode == orbslam_input_mode::mono ) {


		//=========================================================================
		// MONOCULAR Mode
		//=========================================================================

		if (rgb_sensor == nullptr and grey_sensor_one ==nullptr) {
			std::cerr << "Invalid sensors found, RGB or Grey sensor not found." << std::endl;
			return false;
		}

		if(rgb_sensor != nullptr and rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
			std::cerr << "RGB data is in wrong format" << std::endl;
			return false;
		}

		if(rgb_sensor != nullptr and rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
			std::cerr << "RGB data is in wrong format pixel" << std::endl;
			return false;
		}


		cv::Mat camera_parameters = cv::Mat::eye(3,3,CV_32F);
		cv::Mat camera_distortion(5,1,CV_32F);

		if (rgb_sensor) {
			camera_parameters.at<float>(0,0) = ((slambench::io::CameraSensor*)rgb_sensor)->Intrinsics[0]*((slambench::io::CameraSensor*)rgb_sensor)->Width;
			camera_parameters.at<float>(1,1) = ((slambench::io::CameraSensor*)rgb_sensor)->Intrinsics[1]*((slambench::io::CameraSensor*)rgb_sensor)->Height;
			camera_parameters.at<float>(0,2) =  ((slambench::io::CameraSensor*)rgb_sensor)->Intrinsics[2]*((slambench::io::CameraSensor*)rgb_sensor)->Width;
			camera_parameters.at<float>(1,2) = ((slambench::io::CameraSensor*)rgb_sensor)->Intrinsics[3]*((slambench::io::CameraSensor*)rgb_sensor)->Height;

			if (rgb_sensor->DistortionType == slambench::io::CameraSensor::RadialTangential) {
				camera_distortion.at<float>(0) =  rgb_sensor->RadialTangentialDistortion[0];
				camera_distortion.at<float>(1) =  rgb_sensor->RadialTangentialDistortion[1];
				camera_distortion.at<float>(2) =  rgb_sensor->RadialTangentialDistortion[2];
				camera_distortion.at<float>(3) =  rgb_sensor->RadialTangentialDistortion[3];
				camera_distortion.at<float>(4) =  rgb_sensor->RadialTangentialDistortion[4];
			} else {
				camera_distortion.at<float>(0) = 0.0;
				camera_distortion.at<float>(1) = 0.0;
				camera_distortion.at<float>(2) = 0.0;
				camera_distortion.at<float>(3) = 0.0;
				camera_distortion.at<float>(4) = 0.0;
			}


			imRGB       = new cv::Mat ( rgb_sensor->Height ,  rgb_sensor->Width, CV_8UC3);
			inputSize   = make_sb_uint2(rgb_sensor->Width,rgb_sensor->Height);
		} else {
			assert(grey_sensor_one);

			camera_parameters.at<float>(0,0) = grey_sensor_one->Intrinsics[0]*grey_sensor_one->Width;
			camera_parameters.at<float>(1,1) = grey_sensor_one->Intrinsics[1]*grey_sensor_one->Height;
			camera_parameters.at<float>(0,2) = grey_sensor_one->Intrinsics[2]*grey_sensor_one->Width;
			camera_parameters.at<float>(1,2) = grey_sensor_one->Intrinsics[3]*grey_sensor_one->Height;

			if (grey_sensor_one->DistortionType == slambench::io::CameraSensor::RadialTangential) {
				camera_distortion.at<float>(0) =  grey_sensor_one->RadialTangentialDistortion[0];
				camera_distortion.at<float>(1) =  grey_sensor_one->RadialTangentialDistortion[1];
				camera_distortion.at<float>(2) =  grey_sensor_one->RadialTangentialDistortion[2];
				camera_distortion.at<float>(3) =  grey_sensor_one->RadialTangentialDistortion[3];
				camera_distortion.at<float>(4) =  grey_sensor_one->RadialTangentialDistortion[4];
			} else {
				camera_distortion.at<float>(0) = 0.0;
				camera_distortion.at<float>(1) = 0.0;
				camera_distortion.at<float>(2) = 0.0;
				camera_distortion.at<float>(3) = 0.0;
				camera_distortion.at<float>(4) = 0.0;
			}
//            camera_distortion.at<float>(0) = grey_sensor_one->EquidistantDistortion[0];
//            camera_distortion.at<float>(1) = grey_sensor_one->EquidistantDistortion[1];
//            camera_distortion.at<float>(2) = grey_sensor_one->EquidistantDistortion[2];
//            camera_distortion.at<float>(3) = grey_sensor_one->EquidistantDistortion[3];
//            camera_distortion.at<float>(4) = grey_sensor_one->EquidistantDistortion[4];


			img_one     = new cv::Mat ( grey_sensor_one->Height ,  grey_sensor_one->Width, CV_8UC1);
			inputSize   = make_sb_uint2(grey_sensor_one->Width,grey_sensor_one->Height);
		}

		SLAM        = new ORB_SLAM2::System (vocabulary_file,settings_file,ORB_SLAM2::System::MONOCULAR,false);

		if(settings_file == "") {
			SLAM->mpTracker->ConfigureCamera(camera_parameters, camera_distortion, camera_fps, 1, 40 , depth_threshold);
			SLAM->mpTracker->ConfigureAlgorithm(max_features,pyramid_levels,scale_factor,initial_fast_threshold,second_fast_threshold);
		}

		SLAM->mpTracker->PrintConfig();

	} else  if ( input_mode == orbslam_input_mode::stereo ) {


		//=========================================================================
		// STEREO Mode
		//=========================================================================




		if (grey_sensor_one == nullptr or grey_sensor_two == nullptr) {
			std::cerr << "Invalid sensors found, Grey Stereo not found." << std::endl;
			return false;
		}

		cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;

		// Intrisics

		K_l = cv::Mat::zeros(3, 3,CV_64F);
		K_l.at<double>(0,0) =  grey_sensor_one->Intrinsics[0]*grey_sensor_one->Width;
		K_l.at<double>(1,1) =  grey_sensor_one->Intrinsics[1]*grey_sensor_one->Height;
		K_l.at<double>(0,2) =  grey_sensor_one->Intrinsics[2]*grey_sensor_one->Width;
		K_l.at<double>(1,2) =  grey_sensor_one->Intrinsics[3]*grey_sensor_one->Height;
		K_l.at<double>(2,2) =  1.0;

		K_r = cv::Mat::zeros(3, 3,CV_64F);
		K_r.at<double>(0,0) =  grey_sensor_two->Intrinsics[0]*grey_sensor_two->Width;
		K_r.at<double>(1,1) =  grey_sensor_two->Intrinsics[1]*grey_sensor_two->Height;
		K_r.at<double>(0,2) =  grey_sensor_two->Intrinsics[2]*grey_sensor_two->Width;
		K_r.at<double>(1,2) =  grey_sensor_two->Intrinsics[3]*grey_sensor_two->Height;
		K_r.at<double>(2,2) =  1.0;

		// Distortion

//
//		D_l = cv::Mat::zeros(1, 5,CV_64F);
//		D_l.at<double>(0,0) =  grey_sensor_one->RadialTangentialDistortion[0];
//		D_l.at<double>(0,1) =  grey_sensor_one->RadialTangentialDistortion[1];
//		D_l.at<double>(0,2) =  grey_sensor_one->RadialTangentialDistortion[2];
//		D_l.at<double>(0,3) =  grey_sensor_one->RadialTangentialDistortion[3];
//		D_l.at<double>(0,4) =  grey_sensor_one->RadialTangentialDistortion[4];
//
//		D_r = cv::Mat::zeros(1, 5,CV_64F);
//		D_r.at<double>(0,0) =  grey_sensor_two->RadialTangentialDistortion[0];
//		D_r.at<double>(0,1) =  grey_sensor_two->RadialTangentialDistortion[1];
//		D_r.at<double>(0,2) =  grey_sensor_two->RadialTangentialDistortion[2];
//		D_r.at<double>(0,3) =  grey_sensor_two->RadialTangentialDistortion[3];
//		D_r.at<double>(0,4) =  grey_sensor_two->RadialTangentialDistortion[4];


		D_l = cv::Mat::zeros(1, 5,CV_64F);
		D_l.at<double>(0,0) =  grey_sensor_one->EquidistantDistortion[0];
		D_l.at<double>(0,1) =  grey_sensor_one->EquidistantDistortion[1];
		D_l.at<double>(0,2) =  grey_sensor_one->EquidistantDistortion[2];
		D_l.at<double>(0,3) =  grey_sensor_one->EquidistantDistortion[3];
		D_l.at<double>(0,4) =  grey_sensor_one->EquidistantDistortion[4];

		D_r = cv::Mat::zeros(1, 5,CV_64F);
		D_r.at<double>(0,0) =  grey_sensor_two->EquidistantDistortion[0];
		D_r.at<double>(0,1) =  grey_sensor_two->EquidistantDistortion[1];
		D_r.at<double>(0,2) =  grey_sensor_two->EquidistantDistortion[2];
		D_r.at<double>(0,3) =  grey_sensor_two->EquidistantDistortion[3];
		D_r.at<double>(0,4) =  grey_sensor_two->EquidistantDistortion[4];



		// Height and width

		int rows_l = grey_sensor_one->Height;
		int cols_l = grey_sensor_one->Width;
		int rows_r = grey_sensor_two->Height;
		int cols_r = grey_sensor_two->Width;



		std::vector<cv::Mat> vK, vD, vTBS;
		std::vector<cv::Size> vSz;

		// //////////////////////////////////////////////////////////

		// here we read T_BS, K , D and images size for each camera

		// read TBS

		cv::Mat T_BS_l(4,4, CV_64F);

		for (int r = 0; r < 4; r++)

			for (int c = 0; c < 4; c++)

				T_BS_l.at<double>(r,c) =  grey_sensor_one->Pose(r,c) ;

		cv::Mat T_BS_r(4,4, CV_64F);

		for (int r = 0; r < 4; r++)

			for (int c = 0; c < 4; c++)

				T_BS_r.at<double>(r,c) =  grey_sensor_two->Pose(r,c) ;


		// //////////////////////////////////////////////////////////

		cv::Mat R1,R2,P1,P2,Q;

		cv::Mat Tr = (T_BS_r).inv() * (T_BS_l);

		cv::Mat R, T;

		Tr.colRange(0,3).rowRange(0,3).copyTo(R);

		Tr.col(3).rowRange(0,3).copyTo(T);



		// note that order of cameras matter (left camera, ca,era 1) sould be the first one.

		cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(cols_l,rows_l), R, T, R_l, R_r, P_l, P_r, Q, CV_CALIB_ZERO_DISPARITY,0);

		double bf = std::abs (	P_r.at<double>(0,3)  - 	P_l.at<double>(0,3) ) ;

		cv::Mat K = cv::Mat::eye(3,3,CV_32F);
		K.at<float>(0,0) = P_l.at<double>(0,0);
		K.at<float>(1,1) = P_l.at<double>(1,1);
		K.at<float>(0,2) = P_l.at<double>(0,2);
		K.at<float>(1,2) = P_l.at<double>(1,2);


		cv::Mat DistCoef(4,1,CV_32F);
		DistCoef.at<float>(0) = 0.0;
		DistCoef.at<float>(1) = 0.0;
		DistCoef.at<float>(2) = 0.0;
		DistCoef.at<float>(3) = 0.0;

		img_one = new cv::Mat ( grey_sensor_one->Height ,  grey_sensor_one->Width, CV_8UC1);
		img_two = new cv::Mat ( grey_sensor_two->Height ,  grey_sensor_two->Width, CV_8UC1);
		inputSize   = make_sb_uint2(grey_sensor_one->Width,grey_sensor_one->Height);

		SLAM = new ORB_SLAM2::System (vocabulary_file,settings_file,ORB_SLAM2::System::STEREO,false);

		if(settings_file.empty()) {
			SLAM->mpTracker->ConfigureCamera(K, DistCoef, camera_fps, 1, bf , depth_threshold);
			SLAM->mpTracker->ConfigureAlgorithm(max_features,pyramid_levels,scale_factor,initial_fast_threshold,second_fast_threshold);
			SLAM->mpTracker->PrintConfig();
		}


		if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
				rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
		{
			cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
			exit(1);
		}

		cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
		cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);


	}  else if (input_mode == orbslam_input_mode::automatic) {
		std::cout << "No valid sensor found." << std::endl;
		exit(1);
	} else {

		std::cout << "Invalid input mode '" <<   input_mode << "'" << std::endl;
		exit(1);
	}

	pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
	slam_settings->GetOutputManager().RegisterOutput(pose_output);

	pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_POINTCLOUD, true);
	pointcloud_output->SetKeepOnlyMostRecent(true);
	slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);

	frame1_output = new slambench::outputs::Output("Input Frame", slambench::values::VT_FRAME);
	frame1_output->SetKeepOnlyMostRecent(true);
	slam_settings->GetOutputManager().RegisterOutput(frame1_output);

	frame2_output = new slambench::outputs::Output("Tracking frame", slambench::values::VT_FRAME);
	frame2_output->SetKeepOnlyMostRecent(true);
	slam_settings->GetOutputManager().RegisterOutput(frame2_output);


	return true;

}

// TODO: this is ugly
bool depth_ready = false, rgb_ready = false, grey_one_ready = false , grey_two_ready = false;

bool sb_update_frame (SLAMBenchLibraryHelper * , slambench::io::SLAMFrame* s) {
	assert(s != nullptr);
	if(s->FrameSensor->GetType() == slambench::io::GroundTruthSensor::kGroundTruthTrajectoryType and !sb_get_tracked()) {
        cv::Mat matrix(4,4,CV_32F);
	    memcpy(matrix.data, s->GetData(), s->GetSize());
        SLAM->mpTracker->mpMapDrawer->SetCurrentCameraPose(matrix);
		s->FreeData();			
	} if(s->FrameSensor == depth_sensor and imD) {
		memcpy(imD->data, s->GetData(), s->GetSize());
		depth_ready = true;
		s->FreeData();
	} else if(s->FrameSensor == rgb_sensor and imRGB) {
		memcpy(imRGB->data, s->GetData(), s->GetSize());
		rgb_ready = true;		
		s->FreeData();
	} else if(s->FrameSensor == grey_sensor_one and img_one) {
		memcpy(img_one->data, s->GetData(), s->GetSize());
		grey_one_ready = true;
		s->FreeData();
	} else if(s->FrameSensor == grey_sensor_two and img_two) {
		memcpy(img_two->data, s->GetData(), s->GetSize());
		grey_two_ready = true;
		s->FreeData();
	}
	last_frame_timestamp = s->Timestamp;
	return (input_mode == orbslam_input_mode::rgbd and depth_ready and rgb_ready) or
			(input_mode == orbslam_input_mode::mono and rgb_ready) or
			(input_mode == orbslam_input_mode::mono and grey_one_ready) or
			(input_mode == orbslam_input_mode::stereo and grey_one_ready and grey_two_ready);
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {


	if (input_mode == orbslam_input_mode::rgbd) {

		depth_ready = false;
		rgb_ready = false;

		SLAM->TrackRGBD(*imRGB,*imD,tframe);

	} else if (input_mode == orbslam_input_mode::mono) {



		if (rgb_ready) {SLAM->TrackMonocular(*imRGB,tframe);}
		else if (grey_one_ready) {SLAM->TrackMonocular(*img_one,tframe);}
		else {return false;};

		rgb_ready = false;
		grey_one_ready = false;


	} else if (input_mode == orbslam_input_mode::stereo) {


		grey_one_ready = false;
		grey_two_ready = false;

		cv::Mat imLeftRect, imRightRect;

		cv::remap(*img_one,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
		cv::remap(*img_two,imRightRect,M1r,M2r,cv::INTER_LINEAR);

		SLAM->TrackStereo(imLeftRect,imRightRect,tframe);

	} else {
		std::cout << "Unsupported case." << std::endl;
	}

	SLAM->mpFrameDrawer->setState(SLAM->mpTracker->mLastProcessedState);
	frameCV = SLAM->mpFrameDrawer->DrawFrame();
	pose=SLAM->mpTracker->getPose();
    if(!sb_get_tracked())
        SLAM->Relocalize();
	tframe += 1;
	return true;
}


bool sb_clean_slam_system() {
	SLAM->Shutdown();
	return true;
}


/**
 * Getters
 */


bool sb_get_pose (Eigen::Matrix4f * mat)  {
	for(int j=0; j<4;j++) {
		for(int i=0; i<4;i++) {
			(*mat)(j,i)= pose.at<float>(j,i);
		}
	}
	return true;
}


bool sb_relocalize(SLAMBenchLibraryHelper *lib)
{
    return SLAM->Relocalize();
}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output) {
	(void)lib;

	if(pose_output->IsActive()) {
		// Get the current pose as an eigen matrix
		Eigen::Matrix4f matrix;
		sb_get_pose(&matrix);
        auto sb_pose = new slambench::values::PoseValue(matrix);
//        Eigen::Quaternionf q(sb_pose->GetRotation());
//        printf("%d  %f  %f  %f  %f  %f  %f  %f\n", frame_no, sb_pose->GetTranslation().x(), sb_pose->GetTranslation().y(), sb_pose->GetTranslation().z(), q.x(), q.y(), q.z(), q.w());
//        frame_no++;
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		pose_output->AddPoint(*latest_output, new slambench::values::PoseValue(matrix));
	}

	if(pointcloud_output->IsActive()) {

		slambench::values::PointCloudValue *point_cloud = new slambench::values::PointCloudValue();
		std::vector<ORB_SLAM2::MapPoint*> vpMPs;
		vpMPs  = SLAM->mpMapDrawer->mpMap->GetReferenceMapPoints();
		for(size_t i=0; i<vpMPs.size();i++) {
			if(vpMPs[i]->isBad())
				continue;
			cv::Mat pos = vpMPs[i]->GetWorldPos();
			point_cloud->AddPoint(slambench::values::Point3DF(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2)));
		}



		// Take lock only after generating the map
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());

		pointcloud_output->AddPoint(last_frame_timestamp, point_cloud);

	}



	if(frame1_output->IsActive()) {

		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		if (imRGB) {
			frame1_output->AddPoint(last_frame_timestamp, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888,  (void*)(imRGB->data)));
		} else if (img_one) {
			frame1_output->AddPoint(last_frame_timestamp, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::G_I_8,  (void*)(img_one->data)));
		}


	}
	if(frame2_output->IsActive() ) {
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		frame2_output->AddPoint(last_frame_timestamp, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888,  (void*)(&frameCV.at<char>(0,0))));
	}



	return true;
}




