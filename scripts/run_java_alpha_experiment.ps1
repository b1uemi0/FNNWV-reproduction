$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = if ($args.Count -ge 1) { $args[0] } else { "datasets\simulation" }
$runsOutput = if ($args.Count -ge 2) { $args[1] } else { "results\FNNWV_alpha_experiment_runs.tsv" }
$summaryOutput = if ($args.Count -ge 3) { $args[2] } else { "results\FNNWV_alpha_experiment_summary.tsv" }
$times = if ($args.Count -ge 4) { $args[3] } else { "10" }
$workers = if ($args.Count -ge 5) { $args[4] } else { "5" }
$distribution = if ($args.Count -ge 6) { $args[5] } else { "uniform" }
$baseSeed = if ($args.Count -ge 7) { $args[6] } else { "20260508" }
$alphas = if ($args.Count -ge 8) { $args[7] } else { "0.01,0.03,0.05,0.1,0.15,0.2,0.3" }
$quantiles = if ($args.Count -ge 9) { $args[8] } else { "0.1,0.2,0.3" }
$classesDir = Join-Path $root "build\classes"
$libDir = Join-Path $root "vendor\Ceka-v1.0.1\lib"
$classpath = "$classesDir;$libDir\*;$libDir"

Push-Location $root
try {
    java -cp $classpath ceka.FNNWV.RunAlphaExperiment $dataDir $runsOutput $summaryOutput $times $workers $distribution $baseSeed $alphas $quantiles
}
finally {
    Pop-Location
}
