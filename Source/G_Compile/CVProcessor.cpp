// Fill out your copyright notice in the Description page of Project Settings.


#include "CVProcessor.h"
#include "DNNConfig.h"

// Sets default values
ACVProcessor::ACVProcessor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	FString NetworkPath = FPaths::GameSourceDir() + "Network/";

	/* Yolov5 Model */
	FString Yolov5ModelPath = NetworkPath + "yolov5s.onnx";
	FString Yolov5ClassPath = NetworkPath + "class.names";
	this->Yolov5Net = readNet(TCHAR_TO_UTF8(*Yolov5ModelPath));
    if (Yolov5Net.empty())
    {
	    UE_LOG(LogTemp, Warning, TEXT("Yolov5Net Did Not Load!!!"));
    }
    else
    {
	    UE_LOG(LogTemp, Warning, TEXT("Yolov5Net Loaded!!!"));
    }

	/* Yolov3 Model */
	FString Yolov3WeightPath = NetworkPath + "yolov3.weights";
	FString Yolov3CfgPath = NetworkPath + "yolov3.cfg";
	FString Yolov3ClassPath = NetworkPath + "coco.names";
	this->Yolov3Net = readNetFromDarknet(TCHAR_TO_UTF8(*Yolov3CfgPath), TCHAR_TO_UTF8(*Yolov3WeightPath));
	this->Yolov3Net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_OPENCV);
	this->Yolov3Net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);

	/* ResNet SSD Model */
	FString ResSSDModelPath = NetworkPath + "res10_300x300_ssd_iter_140000_fp16.caffemodel";
	FString ResSSDProtoPath = NetworkPath + "deploy.prototxt";
	UE_LOG(LogTemp, Warning, TEXT("file: %s"), *ResSSDModelPath);
	this->SSDResNet = readNetFromCaffe(TCHAR_TO_UTF8(*ResSSDProtoPath), TCHAR_TO_UTF8(*ResSSDModelPath));
	this->SSDResNet.setPreferableBackend(DNN_BACKEND_OPENCV);
	this->SSDResNet.setPreferableTarget(DNN_TARGET_CPU);
}

// Called when the game starts or when spawned
void ACVProcessor::BeginPlay()
{
	Super::BeginPlay();
	if (!UseTCP)
	{
		InitCameraAndThreadRunnable(0);
	}
}

// Called every frame
void ACVProcessor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (Yolov5Count)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Heads: %d"), Yolov5Count);
		ShowYolov5Result(Yolov5Count);
	}
	if (UseYolov3)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Bodies: %d"), Yolov3Count);
		ShowYolov3Result(Yolov3Count);
	}
	if (UseSSDRes)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Faces: %d"), SSDResCount);
		// TODO: Show ResNet SSD Result
	}
}

void ACVProcessor::ReadFrame()
{
	if (Camera.isOpened())
	{
		Mat frame;
		Camera.read(frame);
		if (frame.empty())
		{
			UE_LOG(LogTemp, Warning, TEXT("Frame is Empty !!!"));
			return;
		}
		if (frame.channels() == 4)
		{
			cvtColor(frame, frame, COLOR_BGRA2BGR);
		}
		
		// TODO: EnhanceImage
		if (DoEnhanceImage)
		{
			
		}
		
		AsyncTask(ENamedThreads::GameThread, [=]()
		{
			UTexture2D* OutTexture = ConvertMat2Texture2D(frame);
			// Show Native Capture Image
			ShowNativeImage(OutTexture);
		});

		if (UseYolov5)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use Yolov5 Model"));
			Yolov5Count = 0;
			// TODO: Detect with YoLov5
		}
		if (UseYolov3)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use Yolov3 Model"));
			Yolov3Count = 0;
			DetectYolov3Body(frame);
		}
		if (UseSSDRes)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use ResNet SSD Model"));
			SSDResCount = 0;
			DetectSSDResFace(frame);
		}
		
	}
}

void ACVProcessor::DetectYolov5Head(Mat& Frame)
{
	// TODO: Detect With Yolov5 Model
	if (Frame.empty()) return;
	Yolov5Count = 0;
	int Width = Frame.cols;
	int Height = Frame.rows;
	
	if (DoResizeImage)
	{
		Mat Resized = ResizeImage(Frame, &NewWidth, &NewHeight, &NewTop, &NewLeft);
		
		Mat Yolov5Bolb = blobFromImage(Resized, 1 / 255.0, Size(Yolov5Width, Yolov5Height), Scalar(0, 0, 0), true, false);

		Yolov5Net.setInput(Yolov5Bolb);
		Yolov5Net.forward(Yolov5Outs, Yolov5Net.getUnconnectedOutLayersNames());

		const int NumProposal = Yolov5Outs[0].size[1];
		int nout = Yolov5Outs[0].size[2];
		if (Yolov5Outs[0].dims > 2)
		{
			Yolov5Outs[0] = Yolov5Outs[0].reshape(0, NumProposal);
		}
		float RatioWidth = static_cast<float>(Width) / NewWidth;
		float RatioHeight = static_cast<float>(Height) / NewHeight;
		// TODO Change Variable Name
		int xMin = 0, yMin = 0, xMax = 0, yMax = 0, Index = 0; 
		int n = 0, q = 0, i = 0, j = 0, row_ind = 0;
		float* pdata = (float*)Yolov5Outs[0].data;

		
		if (!Yolov5Bolb.empty()) Yolov5Bolb.resize(0);
	}
	else
	{
		Mat Yolov5Bolb = blobFromImage(Frame, 1 / 255.0, Size(Yolov5Width, Yolov5Height), Scalar(0, 0, 0), true, false);
		Yolov5Net.setInput(Yolov5Bolb);
		Yolov5Net.forward(Yolov5Outs, Yolov5Net.getUnconnectedOutLayersNames());
	}
	
}

// Detect With Yolov3 Model
void ACVProcessor::DetectYolov3Body(Mat& Frame)
{
	if (Frame.empty()) return;
	Yolov3Count = 0;
	int Width = Frame.cols;
	int Height = Frame.rows;
	Mat Yolov3Bolb = blobFromImage(Frame, 1 / 255.0, Size(Yolov3Width, Yolov3Height), Scalar(0, 0, 0),false, false);
	Yolov3Net.setInput(Yolov3Bolb);
	// TODO: Get Yolov3 Output
	// Yolov3Net.forward(m_outs, getOutputsNames(Yolov3Net));
	// postprocess(Frame, m_outs);
	if(!Yolov3Bolb.empty()) Yolov3Bolb.resize(0);
}

// Detect With ResNet SSD Model
void ACVProcessor::DetectSSDResFace(Mat& Frame)
{
	if (Frame.empty()) return;
	TArray<UTexture2D*> SSDResOuts;//类似std::Vector动态数组
	SSDResCount = 0;
	SSDResFaceX = VACUNT;
	SSDResFaceY = VACUNT;
	SSDResFaceSize = VACUNT;
	int Width = Frame.cols;
	int Height = Frame.rows;
	Mat SSDResBlob = blobFromImage(Frame, 0.5, Size(SSDResWidth, SSDResHeight), Scalar(0, 0, 0), false, false);
	SSDResNet.setInput(SSDResBlob, "data");
	Mat FaceDetection = SSDResNet.forward("detection_out");
	Mat Detections(FaceDetection.size[2], FaceDetection.size[3], CV_32F, FaceDetection.ptr<float>());
	

	UE_LOG(LogTemp, Warning, TEXT("####Detected: %d"), Detections.rows);

	for (int i = 0; i < Detections.rows; i++)
	{
		const float Confidence = Detections.at<float>(i, 2);
		if(Confidence > SSDResConfidence)
		{
			++SSDResCount;

			float xTL = Detections.at<float>(i, 3);
			float yTL = Detections.at<float>(i, 4);
			float xBR = Detections.at<float>(i, 5);
			float yBR = Detections.at<float>(i, 6);

			float w = xBR - xTL;
			float h = yBR - yTL;

			float centerX = (xTL + w / 2) * 1920;
			float centerY = (yTL + h / 2) * 1080;
			SSDResFaceX.Add(centerX);
			SSDResFaceY.Add(centerY);

			float size = sqrt(w * w + h * h);
			SSDResFaceSize.Add(size);

			UE_LOG(LogTemp, Warning, TEXT("####Face At X:%f, Y:%f, S:%f"), centerX, centerY, size);
		}
	}
}

// Convert captured image from OpenCV Mat to Texture2D for Unreal
UTexture2D* ACVProcessor::ConvertMat2Texture2D(const Mat& InMat)
{
	int32 Width = InMat.cols;
	int32 Height = InMat.rows;
	int32 Channels = InMat.channels();
	Mat OutMat;
	cvtColor(InMat, OutMat, CV_RGB2RGBA);
	UTexture2D* OutTexture = UTexture2D::CreateTransient(Width, Height);
	OutTexture->SRGB = 0;
	//NewTexture >CompressionSettings = TextureCompressionSettings::TC_Displacementmap;
	const int DataSize = Width * Height * 4;
	void* TextureData = OutTexture->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	FMemory::Memmove(TextureData, OutMat.data, DataSize);
	OutTexture->PlatformData->Mips[0].BulkData.Unlock();
	OutTexture->UpdateResource();
	return OutTexture;
}

// Initialize Camera and Thread Runnable
void ACVProcessor::InitCameraAndThreadRunnable(uint32 index)
{
	Async<>(EAsyncExecution::Thread, [=]()
	{
		if (Camera.open(index))
		{
			UE_LOG(LogTemp, Warning, TEXT("Open Camera Sucessful !!!"));
			Camera.set(CV_CAP_PROP_FRAME_WIDTH,1920);
			Camera.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
			Camera.set(CV_CAP_PROP_FPS, 30);
			ReadThread = FReadImageRunnable::InitReadRunnable(this);
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("Open Camera Failed !!!"));
		}
		FPlatformProcess::Sleep(0.01);
	});
}

Mat ACVProcessor::ResizeImage(Mat InMat, int *Width, int *Height, int *Top, int *Left)
{
	const int InWidth = InMat.cols;
	const int InHeight = InMat.rows;
	*Width = Yolov5Width;
	*Height = Yolov5Height;
	Mat OutMat;
	if (DoKeepRatio && InHeight != InWidth) {
		const float InScale = static_cast<float>(InHeight) / InWidth;
		if (InScale > 1) {
			*Width = static_cast<int>(Yolov5Width / InScale);
			*Height = Yolov5Height;
			resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
			*Left = static_cast<int>((Yolov5Width - *Width) * 0.5);
			copyMakeBorder(OutMat, OutMat, 0, 0, *Left, Yolov5Width - *Width - *Left, BORDER_CONSTANT, 114);
		}
		else {
			*Width = Yolov5Width;
			*Height = static_cast<int>(Yolov5Height * InScale);
			resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
			*Top = static_cast<int>((Yolov5Height - *Height) * 0.5);
			copyMakeBorder(OutMat, OutMat, *Top, Yolov5Height - *Height - *Top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
	}
	return OutMat;
}

void ACVProcessor::PostProcessing(const Mat& Frame, vector<Mat>& Outs)
{
	for (size_t i = 0; i < Outs.size(); i++)
	{
	}
		// Mat Out = Outs[i];
		// for (int j = 0; j < Out.rows; j++)
		// {
		// 	const int ClassId = Out.at<float>(j, 1);
		// 	const float Confidence = Out.at<float>(j, 2);
		// 	if (Confidence > Yolov5Confidence)
		// 	{
		// 		const int Left = static_cast<int>(Out.at<float>(j, 3) * Frame.cols);
		// 		const int Top = static_cast<int>(Out.at<float>(j, 4) * Frame.rows);
		// 		const int Right = static_cast<int>(Out.at<float>(j, 5) * Frame.cols);
		// 		const int Bottom = static_cast<int>(Out.at<float>(j, 6) * Frame.rows);
		// 		const int Width = Right - Left + 1;
		// 		const int Height = Bottom - Top + 1;
		// 		UE_LOG(LogTemp, Warning, TEXT("####Detected: %d"), ClassId);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Confidence: %f"), Confidence);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Left: %d"), Left);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Top: %d"), Top);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Right: %d"), Right);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Bottom: %d"), Bottom);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Width: %d"), Width);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Height: %d"), Height);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Width: %d"), Frame.cols);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Height: %d"), Frame.rows);
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Channels: %d"), Frame.channels());
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Type: %d"), Frame.type());
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Depth: %d"), Frame.depth());
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Size: %d"), Frame.size());
		// 		UE_LOG(LogTemp, Warning, TEXT("####Frame Step: %d"), Frame.step);
		// 		UE_LOG(LogTemp, Warning, TEXT("	"));
}

/*Thread Instance*/
FReadImageRunnable*  FReadImageRunnable::ReadInstance = nullptr;

