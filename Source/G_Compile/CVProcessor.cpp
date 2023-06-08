// Fill out your copyright notice in the Description page of Project Settings.


#include "CVProcessor.h"

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

void ACVProcessor::ReadFrame()
{
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

	if (UseYolov5)
	{
		UE_LOG(LogTemp, Warning, TEXT("Use Yolov5 Model"));
		// TODO: Show Yolov5 Result
	}
	if (UseYolov3)
	{
		UE_LOG(LogTemp, Warning, TEXT("Use Yolov3 Model"));
		// TODO: Show Yolov3 Result
	}
	if (UseSSDRes)
	{
		UE_LOG(LogTemp, Warning, TEXT("Use ResNet SSD Model"));
		// TODO: Show ResNet SSD Result
	}
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

