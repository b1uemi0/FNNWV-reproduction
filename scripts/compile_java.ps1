$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$classesDir = Join-Path $root "build\classes"
$sourceList = Join-Path $root "build\ceka_sources.txt"
$cekaSrc = Join-Path $root "vendor\Ceka-v1.0.1\src"
$cekaLib = Join-Path $root "vendor\Ceka-v1.0.1\lib\*"

New-Item -ItemType Directory -Force (Join-Path $root "build") | Out-Null
if (Test-Path $classesDir) {
    Remove-Item -Recurse -Force $classesDir
}
New-Item -ItemType Directory -Force $classesDir | Out-Null

Get-ChildItem -Recurse $cekaSrc -Filter *.java |
    ForEach-Object { '"' + ($_.FullName -replace "\\", "/") + '"' } |
    Set-Content $sourceList

javac --release 8 -encoding windows-1252 -cp $cekaLib -d $classesDir "@$sourceList"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile CEKA sources"
}

$localSources = Get-ChildItem (Join-Path $root "code") -Filter *.java |
    Where-Object {
        $_.Name.StartsWith("FNNWV") -or
        $_.Name.StartsWith("Run") -or
        $_.Name -eq "IWMV.java" -or
        $_.Name -eq "LAWMV.java" -or
        $_.Name -eq "MNLDP.java" -or
        $_.Name -eq "NeighborUtils.java" -or
        $_.Name -eq "MyQP.java"
    } |
    ForEach-Object { $_.FullName }
javac --release 8 -encoding GBK -cp "$classesDir;$cekaLib" -d $classesDir $localSources
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile FNNWV runner sources"
}

@"
log4j.rootLogger=WARN, stdout
log4j.appender.stdout=org.apache.log4j.ConsoleAppender
log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
log4j.appender.stdout.layout.ConversionPattern=%-5p %c - %m%n
"@ | Set-Content (Join-Path $classesDir "log4j.properties")

Write-Host "Compiled Java classes to $classesDir"
