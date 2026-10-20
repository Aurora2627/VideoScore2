param(
    [string]$EnvironmentPath = $(if ($env:VIDEOSCORE2_ENV) { $env:VIDEOSCORE2_ENV } else { 'D:/career-projects/envs/videoscore2' })
)
$ErrorActionPreference = 'Stop'
$activate = Join-Path $EnvironmentPath 'Scripts/Activate.ps1'
if (-not (Test-Path $activate)) {
    throw "VideoScore2 environment not found at $EnvironmentPath. Set VIDEOSCORE2_ENV or pass -EnvironmentPath."
}
. $activate -Prompt 'videoscore2-cuda'
$env:PYTHONUTF8 = '1'
$env:VIDEOSCORE2_ENV = $EnvironmentPath
python -c "import os,sys; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(os.environ['VIDEOSCORE2_ENV']).resolve(); print('Environment: videoscore2-cuda'); print('Python:',sys.executable)"
if ($LASTEXITCODE -ne 0) { throw 'Unexpected Python interpreter after activation.' }
