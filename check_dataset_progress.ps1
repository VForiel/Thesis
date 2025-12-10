# Script to check dataset generation progress

$targetFile = "generated/dataset/phase_maps/dataset_ns10000_nst20_gamma9.npz"

Write-Host "Monitoring dataset generation..." -ForegroundColor Cyan
Write-Host "Target file: $targetFile`n"

while ($true) {
    if (Test-Path $targetFile) {
        $fileInfo = Get-Item $targetFile
        $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
        
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Dataset file exists!" -ForegroundColor Green
        Write-Host "  Size: $sizeMB MB"
        Write-Host "  Created: $($fileInfo.CreationTime)"
        Write-Host "  Last modified: $($fileInfo.LastWriteTime)"
        
        # Load and check dimensions
        try {
            $output = python -c "import numpy as np; data = np.load('$targetFile'); print(f'Samples: {data[''X''].shape[0]}')"
            Write-Host "  $output" -ForegroundColor Yellow
            
            if ($output -match "10000") {
                Write-Host "`nDataset generation COMPLETE!" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "  File exists but not yet finalized..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Waiting for dataset file..." -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 300  # Check every 5 minutes
}

Write-Host "`nDataset ready for training!" -ForegroundColor Cyan
