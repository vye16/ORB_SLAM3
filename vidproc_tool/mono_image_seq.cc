#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <KeyFrame.h>
#include <System.h>

using namespace cv;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace chrono = std::chrono;

std::vector<fs::path> listdir(const fs::path& p)
{
  std::vector<fs::path> out;
  if (!fs::exists(p)) return out;
  if (!fs::is_directory(p)) return out;

  std::copy(fs::directory_iterator(p), fs::directory_iterator(), std::back_inserter(out));
  std::sort(out.begin(), out.end());
  return out;
}

int main(int argc, char **argv)
{
  fs::path cur_dir = fs::canonical(argv[0]).parent_path();
  fs::path root_dir = cur_dir.parent_path();

  fs::path default_settings = cur_dir / fs::path("480p.yaml");
  fs::path default_vocab = root_dir / fs::path("Vocabulary/ORBvoc.txt");
  printf("Default settings: %s\n", default_settings.c_str());
  printf("Default vocab: %s\n", default_vocab.c_str());

  fs::path img_dir, mask_dir;
  std::string settings_file, vocab_file;
  int wait_ms;
  bool use_viewer, use_masks;

  po::options_description desc("Usage:");
  desc.add_options()
       ("help,h", "produce help message")
       ("img_dir,I", po::value<std::string>(), "image directory")
       ("mask_dir,m", po::value<std::string>(), "mask_directory")
       ("settings,S",
            po::value<std::string>(&settings_file)->default_value(default_settings.string()),
            "camera settings yaml file")
       ("vocab,V",
            po::value<std::string>(&vocab_file)->default_value(default_vocab.string()),
            "ORB vocabulary")
       ("viewer,v",
            po::value<bool>(&use_viewer)->default_value(false),
            "use viewer")
       ("wait",
            po::value<int>(&wait_ms)->default_value(100),
            "wait time in milliseconds")
       ("output,o",
            po::value<std::string>(),
            "output file")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;;
    return 0;
  }

  if (!vm.count("img_dir"))
  {
    printf("Must provide an image directory\n");
    return 1;
  }
  img_dir = fs::path(vm["img_dir"].as<std::string>());
  printf("Using image directory: %s\n", img_dir.c_str());

  use_masks = false;
  if (vm.count("mask_dir"))
  {
    mask_dir = fs::path(vm["mask_dir"].as<std::string>());
    use_masks = true;
    printf("Using mask directory: %s\n", mask_dir.c_str());
  }

  printf("Using settings file: %s\n", settings_file.c_str());
  printf("Using vocab file: %s\n", vocab_file.c_str());
  printf("Using viewer: %d\n", use_viewer);

  std::vector<fs::path> img_files = listdir(img_dir);
  if (!img_files.size())
  {
    printf("%s has no files, exiting.", img_dir.c_str());
    return 0;
  }
  int n_frames = img_files.size();

  std::vector<fs::path> mask_files;
  if (use_masks)
  {
    mask_files = listdir(mask_dir);
    if (mask_files.size() != n_frames)
    {
      printf("Expecting %d masks, only found %d\n", n_frames, mask_files.size());
      return 1;
    }
  }
  printf("Processing %d frames...\n", n_frames);

  // set up the SLAM system
  ORB_SLAM3::System SLAM(vocab_file, settings_file, ORB_SLAM3::System::MONOCULAR, use_viewer);
  std::vector< std::pair<double, int> > tracking_states;

  cv::Mat img, mask;
  double cur_ts, prev_ts, t_track, dt;
  chrono::steady_clock::time_point t1, t2;
  cur_ts = 0;

  for (int i = 0; i < n_frames; ++i) {
    printf("\nProcessing frame %d\n", i);
    img = cv::imread(img_files[i].string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
      break;
    }

    if (use_masks) {
      mask = cv::imread(mask_files[i].string(), cv::IMREAD_UNCHANGED);
    }
    
    t1 = chrono::steady_clock::now();
    SLAM.TrackMonocular(img, mask, cur_ts);
    t2 = chrono::steady_clock::now();

    tracking_states.push_back(std::pair<double, int>(cur_ts, SLAM.GetTrackingState()));

    t_track = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    cur_ts += t_track + (double) wait_ms * 1e-3;
    std::this_thread::sleep_for(chrono::milliseconds(wait_ms));

  }
  printf("Finished processing frames, shutting down now\n");

  SLAM.Shutdown();

  std::string kf_file = "kf_traj.txt";
  std::string traj_file = "camera_traj.txt";
  std::string state_file = "tracking_states.txt";
  if (vm.count("output"))
  {
    std::string prefix = vm["output"].as<std::string>();
    kf_file = prefix + "_" + kf_file;
    traj_file = prefix + "_" + traj_file;
    state_file = prefix + "_" + state_file;
  }

  // save tracking states
  std::ofstream f(state_file.c_str());
  f << std::fixed;
  for (const auto& p: tracking_states)
  {
    std::cout << 1e9 * p.first << " " << p.second << std::endl;
    f << std::setprecision(6) << 1e9 * p.first << " ";
    f << std::setprecision(1) << p.second << std::endl;
  }
  f.close();

  // save camera trajectory
  SLAM.SaveTrajectoryEuRoC(traj_file);
  printf("Saved camera poses to %s", traj_file.c_str());

  // save keyframe trajectory
  SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
  printf("Saved keyframe poses to %s", kf_file.c_str());

  return 0;
}
