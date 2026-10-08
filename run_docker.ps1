# Script de redéploiement automatique local FinTrust AI
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " FinTrust AI - Nettoyage et Relance Docker    " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Arrêter et supprimer le conteneur existant
Write-Host "[1/3] Suppression de l'ancien conteneur 'fintrust-api'..." -ForegroundColor Yellow
docker rm -f fintrust-api 2>$null

# 2. Reconstruire ou mettre à jour l'image
Write-Host "[2/3] Construction de la nouvelle image avec le frontend..." -ForegroundColor Yellow
docker build -t zaidharboul25/fintrust-api:latest .

# 3. Lancer le nouveau conteneur
Write-Host "[3/3] Démarrage du nouveau conteneur sur le port 8000..." -ForegroundColor Green
docker run -d -p 8000:8000 --name fintrust-api zaidharboul25/fintrust-api:latest

Start-Sleep -Seconds 3
Write-Host "`n✅ Succès ! L'interface est accessible sur : http://localhost:8000" -ForegroundColor Cyan
Start-Process "http://localhost:8000"
