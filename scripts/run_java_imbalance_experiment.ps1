$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = if ($args.Count -ge 1) { $args[0] } else { "datasets\simulation" }
$output = if ($args.Count -ge 2) { $args[1] } else { "results\FNNWV_imbalance_ratio_experiment.tsv" }
$times = if ($args.Count -ge 3) { $args[2] } else { "10" }
$classesDir = Join-Path $root "build\classes"
$libDir = Join-Path $root "vendor\Ceka-v1.0.1\lib"
$classpath = "$classesDir;$libDir\*;$libDir"

Push-Location $root
try {
    java -cp $classpath ceka.FNNWV.RunImbalanceRatioExperiment $dataDir $output $times
}
finally {
    Pop-Location
}
