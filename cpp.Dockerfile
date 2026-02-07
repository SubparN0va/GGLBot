# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["cmd", "/S", "/C"]

# ============================================================
# Install VS 2022 Build Tools + CMake + Windows SDK
# ============================================================
RUN powershell -NoProfile -Command `
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " `
    "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -OutFile 'vs_buildtools.exe'"

RUN (start /w vs_buildtools.exe --quiet --wait --norestart --nocache `
        --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" `
        --add Microsoft.VisualStudio.Workload.VCTools `
        --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        --add Microsoft.VisualStudio.Component.VC.CMake.Project `
        --add Microsoft.VisualStudio.Component.Windows10SDK.19041 `
     || IF "%ERRORLEVEL%"=="3010" EXIT 0) `
    && del /q vs_buildtools.exe

# ============================================================
# Install Git using Chocolatey
# ============================================================
SHELL ["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]
RUN $ErrorActionPreference = 'Stop'; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; `
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco feature enable -n allowGlobalConfirmation; `
    choco install git --no-progress; 

SHELL ["cmd", "/S", "/C"]

ENV PATH="C:\Program Files\Git\cmd;C:\Program Files\Git\bin;C:\ProgramData\chocolatey\bin;%PATH%"

RUN C:\Windows\System32\where.exe git && git --version

SHELL ["C:\\Windows\\System32\\cmd.exe", "/S", "/C"]

# ============================================================
# Copy project into container
# ============================================================
WORKDIR C:\src
COPY . C:\src

# Sanity: libtorch must exist at repo root
RUN if not exist C:\src\libtorch\NUL (echo ERROR: Expected C:\src\libtorch at repo root & exit /b 1)

# ============================================================
# Configure + Build (VS generator)
# ============================================================
RUN call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 && `
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DLIBTORCH_ROOT=C:\src\libtorch `
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=C:\src\_BIN `
      -DGGLBOT_COPY_TO_RLBOT=OFF && `
    cmake --build build --config Release -- /m

# ============================================================
# Package everything for bob: ship only the minimum DLLs beside the exe
# ============================================================
RUN mkdir _BOB_OUT && mkdir _BOB_OUT\x86_64-pc-windows-msvc

# Copy the built GGLBot.exe from the actual output directory (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE)
RUN if not exist "C:\src\_BIN\GGLBot.exe" (echo ERROR: Expected C:\src\_BIN\GGLBot.exe but it is missing. && dir /s /b C:\src\_BIN\*.exe 2>NUL && dir /s /b C:\src\build\*.exe 2>NUL && exit /b 1)

RUN copy /Y "C:\src\_BIN\GGLBot.exe" "C:\src\_BOB_OUT\x86_64-pc-windows-msvc\GGLBot.exe"

# Copy logo.png to parent folder (if it exists)
RUN if exist "C:\src\rlbot\logo.png" ( `
    copy /Y "C:\src\rlbot\logo.png" "C:\src\_BOB_OUT\logo.png" `
) else ( `
    echo No logo.png found `
)

# Copy any *.lt files from rlbot (recursively) beside the exe
RUN if exist "C:\src\rlbot" ( `
    for /r "C:\src\rlbot" %F in (*.lt) do ( `
        copy /Y "%F" "C:\src\_BOB_OUT\x86_64-pc-windows-msvc\" `
    ) `
) else ( `
    echo No rlbot folder found at C:\src\rlbot `
)

# Copy any project-produced DLLs that sit next to the exe in _BIN (if any)
RUN if exist "C:\src\_BIN\*.dll" (copy /Y "C:\src\_BIN\*.dll" "C:\src\_BOB_OUT\x86_64-pc-windows-msvc\") else (echo No project DLLs in _BIN)

# Copy libtorch runtime DLLs beside the exe
RUN if exist "C:\src\libtorch\lib\*.dll" (copy /Y "C:\src\libtorch\lib\*.dll" "C:\src\_BOB_OUT\x86_64-pc-windows-msvc\") else (echo No DLLs in libtorch\lib) && `
    if exist "C:\src\libtorch\bin\*.dll" (copy /Y "C:\src\libtorch\bin\*.dll" "C:\src\_BOB_OUT\x86_64-pc-windows-msvc\") else (echo No DLLs in libtorch\bin)

# List output contents for fun
RUN dir /b "C:\src\_BOB_OUT\x86_64-pc-windows-msvc"

# Emit tar to stdout for bob
ENTRYPOINT ["C:\\Windows\\System32\\tar.exe","-C","C:\\src\\_BOB_OUT","-cf","-","x86_64-pc-windows-msvc"]
