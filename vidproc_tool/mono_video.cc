#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <KeyFrame.h>
#include <System.h>

using namespace cv;

int main(int argc, char **argv)
{
  if (argc < 6) {
    std::cerr << std::endl << "Required args: ";
    std::cerr << "[VOCABULARY_FILE] [SETTINGS_FILE] [VIDEO_FILE] [START_TIME] [END_TIME]";
    std::cerr << "Optional args: (USE_VIEWER) (OUTPUT_FILE)" << std::endl;
    return 1;
  }

  const std::string vocab_file(argv[1]);
  const std::string settings_file(argv[2]);
  const std::string source_video(argv[3]);
  double start = std::max(0., atof(argv[4]));
  double end = atof(argv[5]);
  end = (end < 0)? std::numeric_limits<double>::infinity() : end;
  std::cout << "starting at " << start << ", ending at " << end << std::endl;

  bool use_viewer = false;
  if (argc > 6) {
    use_viewer = argv[6];
  }
  std::cout << "Using viewer? " << use_viewer << std::endl;

  cv::VideoCapture cap;
  cap.open(source_video);
  if (!cap.isOpened()) {
    std::cerr << std::endl << "Cannot open video stream " << source_video << std::endl;
    return 1;
  }
  const double max_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const double start_frame = fps * start;
  const double end_frame = std::min(fps * end, max_frames);
  const int num_frames = (int)(end_frame - start_frame);
  std::cout << "start frame: " << start_frame << ", current frame: " << cap.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
  std::cout << "processing " << num_frames << " frames" << std::endl;
  std::cout << "end frame: " << end_frame << ", max frames: " << max_frames << std::endl;

  std::cout << "current frame and msec " <<  cap.get(cv::CAP_PROP_POS_FRAMES) << "\t" << cap.get(cv::CAP_PROP_POS_MSEC) << std::endl;
  std::cout << "moving start to frame " << start_frame << std::endl;
  cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
  std::cout << "current frame and msec " <<  cap.get(cv::CAP_PROP_POS_FRAMES) << "\t" << cap.get(cv::CAP_PROP_POS_MSEC) << std::endl;

  double cur_ts = start;
  double prev_ts, t_track, dt;

  cv::Mat frame;
  cap >> frame;
  std::cout << "frame size: " << frame.size() << std::endl;

  // set up the SLAM system
  ORB_SLAM3::System SLAM(vocab_file, settings_file, ORB_SLAM3::System::MONOCULAR, use_viewer);

  for (int i = 0; i < num_frames; ++i) {
    if (frame.empty()) {
      std::cout << "FRAME EMPTY PREMATURELY! EXITING" << std::endl;
      break;
    }
    
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    SLAM.TrackMonocular(frame, cv::Mat(), cur_ts);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    t_track = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    prev_ts = cur_ts;

    cap >> frame;
    cur_ts = cap.get(cv::CAP_PROP_POS_MSEC) * 1e-3;

//     if (t_track < cur_ts - prev_ts) {
//       dt = cur_ts - prev_ts - t_track;
//       usleep(dt * 1e6);
//     }

  }
  std::cout << "Finished processing frames, shutting down now" << std::endl;

  SLAM.Shutdown();

  if (argc > 7) {
    const string kf_file = string(argv[7]) + "_kf.txt";
    const string f_file = string(argv[7]) + "_f.txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
  } else {
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
  }

}
