$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = if ($args.Count -ge 1) { $args[0] } else { "datasets\simulation" }
$output = if ($args.Count -ge 2) { $args[1] } else { "results\FNNWV_simulation_mv_fnnwv.tsv" }
$times = if ($args.Count -ge 3) { $args[2] } else { "10" }
$workers = if ($args.Count -ge 4) { $args[3] } else { "5" }
$distribution = if ($args.Count -ge 5) { $args[4] } else { "uniform" }
$classesDir = Join-Path $root "build\classes"
$libDir = Join-Path $root "vendor\Ceka-v1.0.1\lib"
$classpath = "$classesDir;$libDir\*;$libDir"

Push-Location $root
try {
    java -cp $classpath ceka.FNNWV.RunSimulationAll $dataDir $output $times $workers $distribution
}
finally {
    Pop-Location
}
