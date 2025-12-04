@echo off
echo ========================================
echo Entrainement du modele VITS
echo ========================================

echo Activation de l'environnement mon_env...
call conda activate mon_env

echo Demarrage de l'entrainement...
python train.py -c configs/ljs_base.json -m ljs_base

echo ========================================
echo Entrainement termine!
echo ========================================
pause
