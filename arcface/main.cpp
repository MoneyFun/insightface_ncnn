#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "arcface.h"
#include "mtcnn.h"

#include "dlib/opencv.h"
#include "dlib/model_utils.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
using namespace cv;
using namespace std;

#define FACE_COUNT 5

cv::Mat ncnn2cv(ncnn::Mat img)
{
    unsigned char pix[img.h * img.w * 3];
    img.to_pixels(pix, ncnn::Mat::PIXEL_BGR);
    cv::Mat cv_img(img.h, img.w, CV_8UC3);
    for (int i = 0; i < cv_img.rows; i++)
    {
        for (int j = 0; j < cv_img.cols; j++)
        {
            cv_img.at<cv::Vec3b>(i,j)[0] = pix[3 * (i * cv_img.cols + j)];
            cv_img.at<cv::Vec3b>(i,j)[1] = pix[3 * (i * cv_img.cols + j) + 1];
            cv_img.at<cv::Vec3b>(i,j)[2] = pix[3 * (i * cv_img.cols + j) + 2];
        }
    }
    return cv_img;
}

float calcY(int left, int center, int right){
    float left_width = center - left;
    float right_width = right - center;

    return ((left_width - right_width) * 10/(right - left));
}

#define ANGLE_DIMENSION 3

int main(int argc, char* argv[])
{
    Mat img[FACE_COUNT];
    vector<FaceInfo> results[FACE_COUNT];
    ncnn::Mat ncnn_img[FACE_COUNT];
    ncnn::Mat det[FACE_COUNT];
    vector<float> feature[FACE_COUNT];
    Arcface arc("../models");
    //3 X,Y,Z
    float angle[ANGLE_DIMENSION] = {0};
    float mouth_open_level = 0;
    float yAngle = 0;
    float eye[2] = {0};

    MtcnnDetector detector("../models");

    img[0] = imread("../image/zhuqinghua.jpg");
    img[1] = imread("../image/wangyuanfei.jpg");
    img[2] = imread("../image/jiuge.jpg");
    img[3] = imread("../image/fbb1.jpeg");
    img[4] = imread("../image/gyy1.jpeg");
    for (int i = 0; i < FACE_COUNT; i ++){
        ncnn_img[i] = ncnn::Mat::from_pixels(img[i].data, ncnn::Mat::PIXEL_BGR, img[i].cols, img[i].rows);
        results[i] = detector.Detect(ncnn_img[i]);
        det[i] = preprocess(ncnn_img[i], results[i][0]);
        feature[i] = arc.getFeature(det[i]);
    }

    dlib::shape_predictor pose_model;
    med::load_shape_predictor_model(pose_model, "shape_predictor_68_face_landmarks_small.dat");
    dlib::full_object_detection detect_result;
    dlib::point p;
    std::ostringstream outtext;

    VideoCapture capture(1);
    while (1)
    {
        Mat frame;
        capture >> frame;
        int width;
        int height;

        ncnn::Mat ncnn_img3 = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
        vector<FaceInfo> results3 = detector.Detect(ncnn_img3);
        if(results3.size() > 0){
            ncnn::Mat det3 = preprocess(ncnn_img3, results3[0]);
            vector<float> feature3 = arc.getFeature(det3);
            for (int i = 0; i < FACE_COUNT; i ++){
                std::cout << i <<  "Similarity: " << calcSimilar(feature[i], feature3) << std::endl;
            }

            rectangle(frame, Point(results3[0].x[0], results3[0].y[0]),
                Point(results3[0].x[1], results3[0].y[1]), cv::Scalar(0, 255, 255), 2, 4);
            width = results3[0].x[1] - results3[0].x[0];
            height = results3[0].y[1] - results3[0].y[0];

            dlib::rectangle face(results3[0].x[0], results3[0].y[0], results3[0].x[1], results3[0].y[1]);

            // Mat faceFrame = frame(Rect(results3[0].x[0], results3[0].y[0], width, height));

            dlib::cv_image<dlib::bgr_pixel> cimg(frame);

            detect_result = pose_model(cimg, face);
#if 0
            for (int j = 36; j < 42; j++) {
                p = detect_result.part(j);
                cv::circle(frame, cv::Point(p.x(), p.y()), 3, cv::Scalar(255,0,0), -1);
                cout << j << ":" << p.x() << endl;
            }

            for (int j = 42; j < 48; j++) {
                p = detect_result.part(j);
                cv::circle(frame, cv::Point(p.x(), p.y()), 3, cv::Scalar(255,0,0), -1);
                cout << j << ":" << p.x() << endl;
            }
#endif

            mouth_open_level = mouth_open_level * 0.9 + abs(0.1 * (detect_result.part(57).y() - detect_result.part(51).y())/
                (detect_result.part(54).x() - detect_result.part(48).x()));

            outtext << "mouth: " << std::setprecision(2) << mouth_open_level;
            cv::putText(frame, outtext.str(), cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");

            float currentY = calcY(detect_result.part(0).x(), detect_result.part(27).x(), detect_result.part(16).x());
            // cout << "currentY: " << currentY << endl;
            yAngle = yAngle * 0.9 + 0.1 * currentY;

            outtext << std::setprecision(2) << yAngle;
            cv::putText(frame, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");

            eye[0] = eye[0] * 0.9 + abs(0.1 * (detect_result.part(39).x() - detect_result.part(36).x()) /
                (detect_result.part(41).y() - detect_result.part(37).y()));

            outtext << "eye0: " << std::setprecision(2) << eye[0];
            cv::putText(frame, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");

            eye[1] = eye[1] * 0.9 + abs(0.1 * (detect_result.part(45).x() - detect_result.part(42).x()) /
                (detect_result.part(43).y() - detect_result.part(47).y()));

            outtext << "eye1: " << std::setprecision(2) << eye[1];
            cv::putText(frame, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");
        }
        imshow("camera", frame);

        char key = waitKey(1);
        printf("%d\n", (int)time(NULL));
        switch(key){
            case 'c':
            case 'C':
            imwrite("capture.jpg", frame);
            break;
            case 'q':
            return 0;
        }
    }
    return 0;
}
