// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"


#include "OpenCVLibrary.h"
#include "Runtime/Core/Public/HAL/RunnableThread.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

#include "CVProcessor.generated.h"

class FRunnable;
class FReadImageRunnable;
using namespace cv;
using namespace dnn;
using namespace std;

struct NetConfig
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

struct DetectionResult
{
	int count;
	vector<float> confidences;
	vector<cv::Rect> boxes;
	vector<int> classIds;
};

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };


UCLASS()
class G_COMPILE_API ACVProcessor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACVProcessor();

	/* Define Networks */
	Net Yolov5Net;
	Net Yolov3Net;
	Net SSDResNet;

	/* Camera */
	VideoCapture Camera;
	FReadImageRunnable* ReadThread;

	/* Actor Default */
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	/* Core */
	void ReadFrame();

	/* Result Var - UPROPERTY */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int Yolov5Count = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int Yolov3Count = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int SSDResCount = 0;

	/* Show Event - UFUNCTION */
	UFUNCTION(BlueprintImplementableEvent)
	void ShowImage(UTexture2D* outRGB,int Width,int Height);
	UFUNCTION(BlueprintImplementableEvent)
	void ShowCutImage( UTexture2D* outRGB);
	UFUNCTION(BlueprintImplementableEvent)
	void ShowNativeImage(UTexture2D* outRGB);

	// TODO: Yolov5 Result
	UFUNCTION(BlueprintImplementableEvent)
	void ShowYolov5Result(int Count);
	// Yolov3
	UFUNCTION(BlueprintImplementableEvent)
	void ShowYolov3Result(int Count);
	// ResNet SSD
	UFUNCTION(BlueprintImplementableEvent)
	void ShowSSDResResult(int Count, const TArray<float>& FaceX, const TArray<float>& FaceY, const TArray<float>& FaceSize);
	
	/* Detections */
	void DetectYolov5Head(Mat& Frame);
	void DetectYolov3Body(Mat& Frame);
	void DetectSSDResFace(Mat& Frame);

private:
	// Define private variables and helper functions
	/* Configs */
	// bool DUseTCP = false;
	// bool UseYolov5 = true;
	// bool UseYolov3 = false;
	// bool UseSSDRes = false;

	static UTexture2D* ConvertMat2Texture2D(const Mat& InMat);
	void InitCameraAndThreadRunnable(uint32 index);

	void CutImage(const Mat inMat, FVector2D inPos);
	void CutImageRect(const Mat inMat, cv::Rect inRect);
};




class G_COMPILE_API FReadImageRunnable :public FRunnable
{
public:
	static FReadImageRunnable* InitReadRunnable(ACVProcessor* inActor)
	{
		if (!ReadInstance&&FPlatformProcess::SupportsMultithreading())
		{
			ReadInstance=new FReadImageRunnable(inActor);
		}
		return ReadInstance;
	}

public:

	virtual bool Init() override
	{
		StopThreadCounter.Increment();
		return true;
	}

	virtual uint32 Run() override
	{
		if (!ReadActor)
		{
			UE_LOG(LogTemp, Warning, TEXT("AHikVisionActor Actor is not spawn"));
			return 1;
		}

		
		while (StopThreadCounter.GetValue())
		{
			ReadActor->ReadFrame();
		}

		/*
			double StartTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
			while (StopThreadCounter.GetValue())
			{
				double EndTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
				if (EndTime - StartTime > 1000)
				{
					ReadActor->ReadFrame();
					StartTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
				}
				else
				{
					continue;
				}
			}
			*/
		return 0;
	}


	virtual void Exit() override
	{
		
	}

	virtual void Stop() override
	{
		if(ReadInstance)
		{
			ReadInstance->EnsureThread();
			delete ReadInstance;
			ReadInstance = nullptr;
		}

	}
	void EnsureThread()
	{
		StopThreadCounter.Decrement();
		if (ReadImageThread) {
			ReadImageThread->WaitForCompletion();
		}
	}
protected:
	FReadImageRunnable(ACVProcessor* inReadActor) 
	{
		
		ReadActor = inReadActor;
		ReadImageThread = FRunnableThread::Create(this, TEXT("ReadImageRunnable"));
	}


	~FReadImageRunnable() {

	};


private:
	FRunnableThread* ReadImageThread;
	ACVProcessor* ReadActor;
	static FReadImageRunnable* ReadInstance;
	FThreadSafeCounter StopThreadCounter;
};
