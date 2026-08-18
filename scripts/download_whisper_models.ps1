param(
    [ValidateSet("tiny.en", "base.en", "small.en")]
    [string[]]$Models = @("base.en", "small.en")
)

$ErrorActionPreference = "Stop"

Push-Location "D:\Assignment\Assignment"
try {
    foreach ($model in $Models) {
        Write-Host "Downloading $model ..."
        python -c "from faster_whisper import WhisperModel; WhisperModel('$model', device='cpu', compute_type='int8', download_root='models'); print('$model ready')"
    }
}
finally {
    Pop-Location
}
