#pragma once

#include "OpenCVLibrary.h"

using namespace cv;
using namespace dnn;
using namespace std;

bool DUseTCP = false;
bool UseYolov5 = true;
bool UseYolov3 = false;
bool UseSSDRes = false;

int Yolov3Width = 608;
int Yolov3Height = 608;
vector<Mat> Yolov3Outs;


int SSDResWidth = 300;
int SSDResHeight = 300;
float SSDResConfidence = 0.5;