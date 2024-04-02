#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include "sophus/se3.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

/*∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗
 ∗ This program demonstrates the dense depth estimation of a monocular camera under a
known trajectory
∗ use epipolar search + NCC matching method (normalized cross corelation)
∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗*/

// ------------------------------------------------------------------
// parameters
const int border = 20;      //image border
const int width = 640;      //image width
const int height = 480;     //image height
const double fx = 481.2f;   //camera instrinsics
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int nccWindowSize = 3;  //half window size of NCC
const int nccArea = (2 * nccWindowSize + 1) * (2 * nccWindowSize + 1); //area of NCC
const double minCov = 0.1;     //convergence criteria : min covariance
const double maxCov = 10;      //divergence criteria : max covariance
// ------------------------------------------------------------------

//Read data from the REMODE data set
bool readDataset(const std::string& path, std::vector<std::string>& colorImageFiles, std::vector<Sophus::SE3d>& poses, cv::Mat& refDepth) {
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin)
        return false;

    while (!fin.eof()) {
        //Data format: image file name tx, ty, tz, qx, qy, qz, qw
        std::string image;
        fin >> image;
        double data[7];
        for (double& d : data) {
            fin >> d;
        }

        colorImageFiles.emplace_back(path + "/images/" + image);
        poses.emplace_back(Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2])));

        if (!fin.good())
            break;
    }
    fin.close();

    //load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    refDepth = cv::Mat(height, width, CV_64F);
    if (!fin)
        return false;
    for (int y = 0; y < height;y++) {
        for (int x = 0; x < width;x++) {
            double depth = 0;
            fin >> depth;
            refDepth.ptr<double>(y)[x] = depth / 100.0;
        }
    }

    return true;
}

inline Eigen::Vector3d px2cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cx) / fy,
        1
    );
}

inline Eigen::Vector2d cam2px(const Eigen::Vector3d pCam) {
    return Eigen::Vector2d(
        pCam(0, 0) * fx / pCam(2, 0) + cx,
        pCam(1, 0) * fy / pCam(2, 0) + cy
    );
}

inline bool inside(const Eigen::Vector2d pt) {
    return pt(0, 0) >= border && pt(1, 0) >= border && pt(0, 0) + border < width && pt(1, 0) + border <= height;
}

inline double getBilinearInterpolationValue(const cv::Mat& img, const Eigen::Vector2d& pt) {
    uchar* d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    double out = ((1 - xx) * (1 - yy) * double(d[0]) +
        xx * (1 - yy) * double(d[1]) +
        (1 - xx) * yy * double(d[img.step]) +
        xx * yy * double(d[img.step + 1])) / 255.0;
    return out;
}

void showEpipolarLine(const cv::Mat& ref, const cv::Mat& current, const Eigen::Vector2d& ptRef, const Eigen::Vector2d& pxMinCurr, const Eigen::Vector2d& pxMaxCurr);
void showEpipolarMatches(const cv::Mat& ref, const cv::Mat& current, const Eigen::Vector2d& ptRef, const Eigen::Vector2d& ptCurr);



double NCC(const cv::Mat& ref, const cv::Mat& current, const Eigen::Vector2d& ptRef, const Eigen::Vector2d& ptCurr) {
    double meanRef = 0, meanCurr = 0;
    std::vector<double> valuesRef, valuesCurr;
    for (int x = -nccWindowSize; x <= nccWindowSize; x++) {
        for (int y = -nccWindowSize; y <= nccWindowSize; y++) {
            double valueRef = double(ref.ptr<uchar>(int(y + ptRef(1, 0)))[int(x + ptRef(0, 0))]) / 255.0;
            meanRef += valueRef;

            double valueCurr = getBilinearInterpolationValue(current, ptCurr + Eigen::Vector2d(x, y));
            meanCurr += valueCurr;

            valuesRef.emplace_back(valueRef);
            valuesCurr.emplace_back(valueCurr);
        }
    }

    meanRef /= nccArea;
    meanCurr /= nccArea;

    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < valuesRef.size();i++) {
        double n = (valuesRef[i] - meanRef) * (valuesCurr[i] - meanCurr);
        numerator += n;
        denominator1 += (valuesRef[i] - meanRef) * (valuesRef[i] - meanRef);
        denominator1 += (valuesCurr[i] - meanCurr) * (valuesCurr[i] - meanCurr);
    }
    return numerator / sqrt(denominator1 * denominator2 * 1e-8);
}

bool epipolarSearch(const cv::Mat& ref, const cv::Mat& current, const Sophus::SE3d& TCR, const Eigen::Vector2d& ptRef,
    const double& depthMu, const double& depthCov, Eigen::Vector2d& ptCurr, Eigen::Vector2d& epipolarDirection) {
    Eigen::Vector3d fRef = px2cam(ptRef);
    fRef.normalize();
    Eigen::Vector3d pRef = fRef * depthMu; //vector in reference frame

    Eigen::Vector2d pxMeanCurr = cam2px(TCR * pRef); // points projected by mean depth
    double dMin = depthMu - 3 * depthCov;
    double dMax = depthMu + 3 * depthCov;
    if (dMin < 0.1)
        dMin = 0.1;
    Eigen::Vector2d pxMinCurr = cam2px(TCR * (fRef * dMin)); // points projected by minimum depth 
    Eigen::Vector2d pxMaxCurr = cam2px(TCR * (fRef * dMax)); // points projected by maximum depth 

    Eigen::Vector2d epipolarLine = pxMaxCurr - pxMinCurr;
    epipolarDirection = epipolarLine;
    epipolarDirection.normalize();
    double halfLength = 0.5 * epipolarLine.norm();
    if (halfLength > 100)
        halfLength = 100;

    // showEpipolarLine(ref, current, ptRef, pxMinCurr, pxMaxCurr);

    // Search on the epipolar line, taking the depth mean point as the center, taking half the length on the left and right sides
    double bestNCC = -1.0;
    Eigen::Vector2d bestPxCurr;

    for (double l = -halfLength; l <= halfLength;l += 0.7) {
        Eigen::Vector2d pxCurr = pxMeanCurr + l * epipolarDirection;
        if (!inside(pxCurr))
            continue;
        double ncc = NCC(ref, current, ptRef, ptCurr);
        if (ncc > bestNCC) {
            bestNCC = ncc;
            bestPxCurr = pxCurr;
        }
    }

    if (bestNCC < 0.85f)
        return false;
    ptCurr = bestPxCurr;
    return true;
}

bool updateDepthFilter(const Eigen::Vector2d& ptRef, const Eigen::Vector2d& ptCurr, const Sophus::SE3d& TCR, Eigen::Vector2d& epipolarDirection, cv::Mat& depth, cv::Mat& depthCov) {
    Sophus::SE3d TRC = TCR.inverse();
    Eigen::Vector3d fRef = px2cam(ptRef);
    fRef.normalize();
    Eigen::Vector3d fCurr = px2cam(ptCurr);
    fCurr.normalize();

    // dRef * fRef = dCur * ( R_RC * fCur ) + t_RC
    // f2 = R_RC * fCur
    //convert eqns to matrix form
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]

    Eigen::Vector3d t = TRC.translation();
    Eigen::Vector3d f2 = TRC.so3() * fCurr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(fRef), t.dot(f2));
    Eigen::Matrix2d A;

    A(0, 0) = fRef.dot(fRef);
    A(0, 1) = -fRef.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    Eigen::Vector2d ans = A.inverse() * b;
    Eigen::Vector3d xm = ans[0] * fRef;
    Eigen::Vector3d xn = t + ans[1] * f2;
    Eigen::Vector3d pEsti = (xm + xn) / 2.0;
    double depthEstimate = pEsti.norm();

    // Calculate uncertainty (error in one pixel)
    Eigen::Vector3d p = fRef * depthEstimate;
    Eigen::Vector3d a = p - t;
    double tNorm = t.norm();
    double aNorm = a.norm();
    double alpha = acos(fRef.dot(t) / tNorm);
    double beta = acos(-a.dot(t) / (aNorm * tNorm));
    Eigen::Vector3d fCurrPrime = px2cam(ptCurr + epipolarDirection);
    fCurrPrime.normalize();
    double betaPrime = acos(fCurrPrime.dot(-t) / tNorm);
    double gamma = M_PI - alpha - betaPrime;
    double pPrime = tNorm * sin(betaPrime) / sin(gamma);
    double dCov = pPrime - depthEstimate;
    double dCov2 = dCov * dCov;

    double mu = depth.ptr<double>(int(ptRef(1, 0)))[int(ptRef(0, 0))];
    double sigma2 = depthCov.ptr<double>(int(ptRef(1, 0)))[int(ptRef(0, 0))];

    double muFuse = (dCov2 * mu + sigma2 * depthEstimate) / (sigma2 + dCov2);
    double sigmaFuse2 = (sigma2 * dCov2) / (sigma2 + dCov2);

    depth.ptr<double>(int(ptRef(1, 0)))[int(ptRef(0, 0))] = muFuse;
    depthCov.ptr<double>(int(ptRef(1, 0)))[int(ptRef(0, 0))] = sigmaFuse2;

    return true;
}

//update the depth map
void update(const cv::Mat& ref, const cv::Mat& current, const Sophus::SE3d& TCR, cv::Mat& depth, cv::Mat& depthCov) {
    for (int x = border; x < width - border; x++) {
        for (int y = border; y < height - border;y++) {
            if (depthCov.ptr<double>(y)[x] < minCov || depthCov.ptr<double>(y)[x] > maxCov)
                continue;

            Eigen::Vector2d ptCurr;
            Eigen::Vector2d epipolarDirection;
            bool ret = epipolarSearch(ref, current, TCR, Eigen::Vector2d(x, y), depth.ptr<double>(y)[x],
                sqrt(depthCov.ptr<double>(y)[x]), ptCurr, epipolarDirection);

            if (ret == false)   //no match
                continue;

            // showEpipolarMatches(ref, current, Eigen::Vector2d(x, y), ptCurr);

            // updateDepthFilter(Eigen::Vector2d(x, y), ptCurr, TCR, epipolarDirection, depth, depthCov);
        }
    }
}



void evaluateDepth(const cv::Mat& depthGT, const cv::Mat& depthEst) {
    double avgError = 0;
    double avgErrorSquared = 0;
    int count = 0;
    for (int y = border; y < depthGT.rows - border; y++) {
        for (int x = border; x < depthGT.cols - border; x++) {
            double error = depthGT.ptr<double>(y)[x] - depthEst.ptr<double>(y)[x];
            avgError += error;
            avgErrorSquared += error * error;
            count++;
        }
    }
    avgError /= count;
    avgErrorSquared /= count;

    std::cout << "Average squared error = " << avgErrorSquared << ", average error: " << avgError << std::endl;

}

void showEpipolarLine(const cv::Mat& ref, const cv::Mat& current, const Eigen::Vector2d& pxRef, const Eigen::Vector2d& pxMinCurr, const Eigen::Vector2d& pxMaxCurr) {
    cv::Mat refShow, currShow;
    cv::cvtColor(ref, refShow, cv::COLOR_GRAY2BGR);
    cv::cvtColor(current, currShow, cv::COLOR_GRAY2BGR);

    cv::circle(refShow, cv::Point2f(pxRef(0, 0), pxRef(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(currShow, cv::Point2f(pxMinCurr(0, 0), pxMinCurr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(currShow, cv::Point2f(pxMaxCurr(0, 0), pxMaxCurr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(currShow, cv::Point2f(pxMinCurr(0, 0), pxMinCurr(1, 0)), cv::Point2f(pxMaxCurr(0, 0), pxMaxCurr(1, 0)),
        cv::Scalar(0, 255, 0), 1);

    cv::imshow("ref", refShow);
    cv::imshow("current", currShow);
    cv::waitKey(1);
}

void showEpipolarMatches(const cv::Mat& ref, const cv::Mat& current, const Eigen::Vector2d& pxRef, const Eigen::Vector2d& pxCurr) {
    cv::Mat refShow, currShow;
    cv::cvtColor(ref, refShow, cv::COLOR_GRAY2BGR);
    cv::cvtColor(current, currShow, cv::COLOR_GRAY2BGR);

    cv::circle(refShow, cv::Point2f(pxRef(0, 0), pxRef(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(currShow, cv::Point2f(pxCurr(0, 0), pxCurr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", refShow);
    cv::imshow("current", currShow);
    cv::waitKey(1);
}

void plotDepth(const cv::Mat& depthGT, const cv::Mat& depthEst) {
    cv::imshow("depthGT", depthGT * 0.4);
    cv::imshow("depthEst", depthEst * 0.4);
    cv::imshow("depthError", depthGT - depthEst);
    cv::waitKey(1);
}

int main() {
    std::vector<std::string> colorImageFiles;
    std::vector<Sophus::SE3d> poses_TWC;
    cv::Mat refDepth;
    std::string path = "../data/";
    bool ret = readDataset(path, colorImageFiles, poses_TWC, refDepth);
    if (ret == false) {
        std::cout << "reading data failed, check path" << std::endl;
        return -1;
    }
    std::cout << "total files : " << colorImageFiles.size() << std::endl;

    cv::Mat ref = cv::imread(colorImageFiles[0], 0); //grayscale
    Sophus::SE3d poseRef_TWC = poses_TWC[0];
    double initDepth = 3.0;     //initial depth
    double initCov = 3.0;    //initial covariance
    cv::Mat depth(height, width, CV_64F, initDepth);    //depth image
    cv::Mat depthCov(height, width, CV_64F, initCov);      //depth covariance image

    for (int index = 1; index < colorImageFiles.size(); index++) {
        std::cout << "*** loop " << index << " *** " << std::endl;
        cv::Mat current = cv::imread(colorImageFiles[index], 0);
        if (current.data == nullptr)
            continue;
        Sophus::SE3d poseCurrent_TWC = poses_TWC[index];
        Sophus::SE3d poseTCR = poseCurrent_TWC.inverse() * poseRef_TWC; // TCW * TWR = TCR
        update(ref, current, poseTCR, depth, depthCov);
        evaluateDepth(refDepth, depth);
        plotDepth(refDepth, depth);
        cv::imshow("image", current);
        cv::waitKey(1);
    }

    std::cout << "estimation converged, saving depth" << std::endl;
    cv::imwrite("../output/depth.png", depth);

    std::cout << "Done!" << std::endl;
    return 0;
}