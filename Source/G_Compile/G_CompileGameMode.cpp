// Copyright Epic Games, Inc. All Rights Reserved.

#include "G_CompileGameMode.h"
#include "G_CompileCharacter.h"
#include "UObject/ConstructorHelpers.h"

AG_CompileGameMode::AG_CompileGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
