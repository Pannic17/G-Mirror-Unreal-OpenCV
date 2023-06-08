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
	if (!DUseTCP)
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
			// TODO: Detect with YoLov3
		}
		if (UseSSDRes)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use ResNet SSD Model"));
			SSDResCount = 0;
			// TODO: Detect with ResNet SSD
		}
		
	}
}

void ACVProcessor::DetectYolov3Body(Mat& inMat)
{
	
	if (inMat.empty()) return;
	Yolov3Count = 0;
	int Width = inMat.cols;
	int Height = inMat.rows;
	cv::Mat m_blob = cv::dnn::blobFromImage(inMat, 1 / 255.0, cv::Size(Yolov3Width, Yolov3Height), cv::Scalar(0, 0, 0),false, false);
	Yolov3Net.setInput(m_blob);
	vector<Mat> m_outs;
	// TODO: Get Yolov3 Output
	// Yolov3Net.forward(m_outs, getOutputsNames(Yolov3Net));
	// postprocess(inMat, m_outs);
	if(!m_outs.empty()) m_outs.resize(0);
}

// Convert captured image from OpenCV Mat to Texture2D for Unreal
UTexture2D* ACVProcessor::ConvertMat2Texture2D(const Mat& InMat)
{
	int32 Width = InMat.cols;
	int32 Height = InMat.rows;
	int32 Channels = InMat.channels();
	cv::Mat OutMat;
	cv::cvtColor(InMat, OutMat, CV_RGB2RGBA);
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

/*Thread Instance*/
FReadImageRunnable*  FReadImageRunnable::ReadInstance = nullptr;

