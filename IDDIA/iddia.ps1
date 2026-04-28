$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..").Path
python -m IDDIA @args
