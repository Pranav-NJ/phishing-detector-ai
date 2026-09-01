# MongoDB Installation Script for Windows
# Run this script as Administrator in PowerShell

Write-Host "🚀 Starting MongoDB Installation..." -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Please run this script as Administrator!" -ForegroundColor Red
    exit 1
}

# Download MongoDB
Write-Host "📥 Downloading MongoDB Community Server..." -ForegroundColor Yellow
$downloadUrl = "https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-7.0.5-signed.msi"
$installerPath = "$env:TEMP\mongodb-installer.msi"

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "✅ MongoDB downloaded successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download MongoDB. Please download manually from: https://www.mongodb.com/try/download/community" -ForegroundColor Red
    exit 1
}

# Install MongoDB silently
Write-Host "🔧 Installing MongoDB..." -ForegroundColor Yellow
try {
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $installerPath, "/quiet", "/norestart", "ADDLOCAL=Server,Client,Router,MonitoringTools,ImportExportTools" -Wait
    Write-Host "✅ MongoDB installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install MongoDB. Please run the installer manually." -ForegroundColor Red
    exit 1
}

# Add MongoDB to PATH
$mongoPath = "C:\Program Files\MongoDB\Server\7.0\bin"
if ($env:PATH -notlike "*$mongoPath*") {
    [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$mongoPath", "User")
    Write-Host "✅ MongoDB added to PATH" -ForegroundColor Green
}

# Start MongoDB service
Write-Host "🔄 Starting MongoDB service..." -ForegroundColor Yellow
try {
    Start-Service -Name "MongoDB" -ErrorAction SilentlyContinue
    Write-Host "✅ MongoDB service started!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ MongoDB service not found, attempting to install..." -ForegroundColor Yellow
    try {
        & "C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --install
        Start-Service -Name "MongoDB"
        Write-Host "✅ MongoDB service installed and started!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to start MongoDB service" -ForegroundColor Red
    }
}

# Test connection
Write-Host "🔍 Testing MongoDB connection..." -ForegroundColor Yellow
$testResult = Test-NetConnection -ComputerName "localhost" -Port 27017
if ($testResult.TcpTestSucceeded) {
    Write-Host "✅ MongoDB is running on port 27017!" -ForegroundColor Green
} else {
    Write-Host "❌ MongoDB connection failed" -ForegroundColor Red
}

# Create database and user
Write-Host "👤 Creating database user..." -ForegroundColor Yellow
try {
    $mongoScript = @"
use phishing_detector;
db.createUser({
    user: "phishing_user",
    pwd: "phishing123",
    roles: ["readWrite"]
});
exit;
"@
    
    $mongoScript | Out-File -FilePath "$env:TEMP\setup_mongo.js" -Encoding ASCII
    & "C:\Program Files\MongoDB\Server\7.0\bin\mongo.exe" "$env:TEMP\setup_mongo.js"
    Write-Host "✅ Database user created successfully!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Failed to create database user. You may need to do this manually." -ForegroundColor Yellow
}

# Cleanup
Remove-Item $installerPath -ErrorAction SilentlyContinue
Remove-Item "$env:TEMP\setup_mongo.js" -ErrorAction SilentlyContinue

Write-Host "🎉 MongoDB installation completed!" -ForegroundColor Green
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart your server: cd server && npm start" -ForegroundColor White
Write-Host "2. Test the system at: http://localhost:3000" -ForegroundColor White
Write-Host "3. Check for MongoDB connection success message" -ForegroundColor White
