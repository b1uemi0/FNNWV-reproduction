$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataset = if ($args.Count -ge 1) { $args[0] } else { "datasets\simulation\iris0.arff" }
$classesDir = Join-Path $root "build\classes"
$libDir = Join-Path $root "vendor\Ceka-v1.0.1\lib"
$classpath = "$classesDir;$libDir\*;$libDir"

Push-Location $root
try {
    java -cp $classpath ceka.FNNWV.RunNeighborSelfCheck $dataset
}
finally {
    Pop-Location
}
