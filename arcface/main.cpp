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

double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

Mat calcAngle(dlib::full_object_detection shape) {
    std::vector<cv::Point2d> image_pts;

    //fill in cam intrinsics and distortion coefficients
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    std::vector<cv::Point3d> object_pts;
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);


    //fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
    image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
    image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
    image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
    image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
    image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
    image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
    image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
    image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
    image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
    image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
    image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
    image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
    image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
    image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

    //calc pose
    cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

    //calc euler angle
    cv::Rodrigues(rotation_vec, rotation_mat);
    cv::hconcat(rotation_mat, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

    return euler_angle;
}

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
            for (int j = 0; j < detect_result.num_parts(); j++) {
                p = detect_result.part(j);
                cv::circle(frame, cv::Point(p.x(), p.y()), 3, cv::Scalar(255,0,0), -1);
                cout << j << ":" << p.x() << endl;
            }
#endif
#if 1
            Mat euler_angle = calcAngle(detect_result);
            cout << euler_angle.at<double>(0) << endl;

            outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);
            cv::putText(frame, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");
            outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);
            cv::putText(frame, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");
            outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);
            cv::putText(frame, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");
#endif
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
