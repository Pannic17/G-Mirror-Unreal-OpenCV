// Copyright Epic Games, Inc. All Rights Reserved.
using System.IO;
using UnrealBuildTool;

public class G_Compile : ModuleRules
{
	
	string OPENCV_VERSION = "451";
	private string ThirdPartyPath
	{
		get
		{
			return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/"));
		}
	}
	
	public bool LoadOpenCV(ReadOnlyTargetRules Target)
	{
		// only set up for Win64
		bool isLibrarySupported = false;

		// Create OpenCV Path 
		string OpenCVPath = Path.Combine(ThirdPartyPath, "OpenCV");

		// Get Library Path 
		string LibPath = "";
		bool isdebug = Target.Configuration == UnrealTargetConfiguration.Debug;
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			LibPath = Path.Combine(OpenCVPath, "Libraries", "Win64");
			isLibrarySupported = true;
		}
		else
		{
			string Err = string.Format("{0} dedicated server is made to depend on {1}. We want to avoid this, please correct module dependencies.", Target.Platform.ToString(), this.ToString());
			System.Console.WriteLine(Err);
		}

		if (isLibrarySupported)
		{
			//Add Include path 
			PublicIncludePaths.AddRange(new string[] { Path.Combine(OpenCVPath, "include") });
			//Add Static Libraries
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "opencv_world" + OPENCV_VERSION + ".lib"));
			//Add Dynamic Libraries
			PublicDelayLoadDLLs.Add("opencv_world" + OPENCV_VERSION + ".dll");
			PublicDelayLoadDLLs.Add("opencv_videoio_ffmpeg" + OPENCV_VERSION + "_64.dll");
		}
		PublicDefinitions.Add(string.Format("WITH_OPENCV_BINDING={0}", isLibrarySupported ? 1 : 0));
		return isLibrarySupported;
	}
	
	public G_Compile(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });
		
		PrivateDependencyModuleNames.AddRange(new string[] {  });
		
		PublicDependencyModuleNames.AddRange(new string[] { "RHI", "RenderCore", "Media", "MediaAssets" });

		LoadOpenCV(Target);
	}
}
