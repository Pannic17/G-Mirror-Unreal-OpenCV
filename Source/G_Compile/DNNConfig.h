#pragma once

#include "OpenCVLibrary.h"

using namespace cv;
using namespace dnn;
using namespace std;

TArray<float> VACUNT;
bool DoEnhanceImage = false;

bool UseTCP = false;
bool UseYolov5 = true;
bool UseYolov3 = false;
bool UseSSDRes = false;
vector<Mat> Yolov5Outs;

bool DoResizeImage = false;
bool DoKeepRatio = true;
int Yolov5Width = 640;
int Yolov5Height = 640;
int Yolov5StrideNum = 3;
int NewWidth = 0;
int NewHeight = 0;
int NewTop = 0;
int NewLeft = 0;

int Yolov3Width = 608;
int Yolov3Height = 608;
vector<Mat> Yolov3Outs;

int SSDResWidth = 300;
int SSDResHeight = 300;
float SSDResConfidence = 0.5;
TArray<float> SSDResFaceX;
TArray<float> SSDResFaceY;
TArray<float> SSDResFaceSize;