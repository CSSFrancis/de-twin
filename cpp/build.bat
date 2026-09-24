@echo off
rem Build cpp\build\test_consumer.exe (the ExternalFrameSource check) with MSVC.
setlocal
set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat
)
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /Fobuild\ /Fe:build\test_consumer.exe test_consumer.cpp ExternalFrameSource.cpp || exit /b 1
echo BUILD OK
