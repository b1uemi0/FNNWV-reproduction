$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = if ($args.Count -ge 1) { $args[0] } else { "datasets\real-world" }
$output = if ($args.Count -ge 2) { $args[1] } else { "results\FNNWV_realworld_ner_mv_fnnwv.tsv" }
$classesDir = Join-Path $root "build\classes"
$libDir = Join-Path $root "vendor\Ceka-v1.0.1\lib"
$classpath = "$classesDir;$libDir\*;$libDir"

Push-Location $root
try {
    java -cp $classpath ceka.FNNWV.RunRealWorldNer $dataDir $output
}
finally {
    Pop-Location
}
