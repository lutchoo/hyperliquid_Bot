#!/bin/bash

# Ajouter directement le script Python à exécuter
PYTHON_SCRIPT="python3 hyperliqui_RsiBot.py"

# Vérifier si la ligne existe déjà dans 1hcron.sh
if grep -Fxq "$PYTHON_SCRIPT" hyperBot/1hcron.sh; then
    echo "Le script $PYTHON_SCRIPT existe déjà dans 1hcron.sh"
else
    # Ajouter le script Python au fichier 1hcron.sh
    echo "$PYTHON_SCRIPT" >> hyperBot/1hcron.sh
    echo "Le script $PYTHON_SCRIPT a été ajouté à 1hcron.sh"
fi

# Mise à jour du serveur
echo "Mise à jour du serveur..."
sudo apt-get update

# Installation de pip
echo "Installation de pip..."
sudo apt install -y pip

# Création du fichier de log s'il n'existe pas
touch cronlog.log

# Configuration de l'environnement virtuel Python
cd hyperBot
sudo apt-get install -y python3-venv
python3 -m venv .hyperliquidEnv
source .hyperliquidEnv/bin/activate
pip install -r requirements.txt
git update-index --assume-unchanged secret.py
cd ..

# Ajouter la tâche cron si elle n'existe pas déjà
if ! crontab -l | grep -q 'bash ./hyperliquid/1hcron.sh'; then
    (crontab -l 2>/dev/null; echo "0 * * * * /bin/bash ./hyperBot/1hcron.sh >> cronlog.log 2>&1") | crontab -
    echo "Tâche cron ajoutée avec succès."
else
    echo "La tâche cron existe déjà."
fi