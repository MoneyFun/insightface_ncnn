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

int main(int argc, char* argv[])
{
    Mat img[FACE_COUNT];
    vector<FaceInfo> results[FACE_COUNT];
    ncnn::Mat ncnn_img[FACE_COUNT];
    ncnn::Mat det[FACE_COUNT];
    vector<float> feature[FACE_COUNT];
    Arcface arc("../models");

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

    //for (auto it = results1.begin(); it != results1.end(); it++)
    //{
    //    rectangle(img1, cv::Point(it->x[0], it->y[0]), cv::Point(it->x[1], it->y[1]), cv::Scalar(0, 255, 0), 2);
    //    circle(img1, cv::Point(it->landmark[0], it->landmark[1]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img1, cv::Point(it->landmark[2], it->landmark[3]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img1, cv::Point(it->landmark[4], it->landmark[5]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img1, cv::Point(it->landmark[6], it->landmark[7]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img1, cv::Point(it->landmark[8], it->landmark[9]), 2, cv::Scalar(0, 255, 0), 2);
    //}

    //for (auto it = results2.begin(); it != results2.end(); it++)
    //{
    //    rectangle(img2, cv::Point(it->x[0], it->y[0]), cv::Point(it->x[1], it->y[1]), cv::Scalar(0, 255, 0), 2);
    //    circle(img2, cv::Point(it->landmark[0], it->landmark[1]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img2, cv::Point(it->landmark[2], it->landmark[3]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img2, cv::Point(it->landmark[4], it->landmark[5]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img2, cv::Point(it->landmark[6], it->landmark[7]), 2, cv::Scalar(0, 255, 0), 2);
    //    circle(img2, cv::Point(it->landmark[8], it->landmark[9]), 2, cv::Scalar(0, 255, 0), 2);
    //}

    //waitKey(1000);
    dlib::frontal_face_detector dlib_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    med::load_shape_predictor_model(pose_model, "shape_predictor_68_face_landmarks_small.dat");
    dlib::full_object_detection detect_result;
    dlib::point p;

    VideoCapture capture(-1);
    while (1)
    {
        Mat frame;
        capture >> frame;
        int width;
        int height;

        ncnn::Mat ncnn_img3 = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
        vector<FaceInfo> results3 = detector.Detect(ncnn_img3);
        if(results3.size() > 0){
            // printf("x[0]:%d\n", results3[0].x[0]);
            // printf("x[1]:%d\n", results3[0].x[1]);
            // printf("y[0]:%d\n", results3[0].y[0]);
            // printf("y[1]:%d\n", results3[0].y[1]);
            // for(int i = 0;i < 10;i ++)
            //    cout << results3[0].landmark[i] << endl;
            ncnn::Mat det3 = preprocess(ncnn_img3, results3[0]);
            vector<float> feature3 = arc.getFeature(det3);
            for (int i = 0; i < FACE_COUNT; i ++){
                std::cout << i <<  "Similarity: " << calcSimilar(feature[i], feature3) << std::endl;
            }

            rectangle(frame, Point(results3[0].x[0], results3[0].y[0]),
                Point(results3[0].x[1], results3[0].y[1]), cv::Scalar(0, 255, 255), 2, 4);
            width = results3[0].x[1] - results3[0].x[0];
            height = results3[0].y[1] - results3[0].y[0];

            Mat faceFrame = frame(Rect(results3[0].x[0], results3[0].y[0], width, height));

            dlib::cv_image<dlib::bgr_pixel> cimg(faceFrame);

            std::vector<dlib::rectangle> faces = dlib_detector(cimg);
            // Find the pose of each face.
            // std::vector<full_object_detection> shapes;

            for (unsigned long i = 0; i < faces.size(); ++i)
                //pose_model(cimg, faces[i]);
                detect_result = pose_model(cimg, faces[i]);
            #if 0
            for (char j = 0; j < detect_result.num_parts(); j++) {
                p = detect_result.part(j);
                cv::circle(faceFrame, cv::Point(p.x(), p.y()), 3, cv::Scalar(255,0,0), -1);
            }
            #endif
            imshow("faceFrame", faceFrame);
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
