@echo off
echo ========================================
echo Configuration VITS avec Anaconda Python 3.9
echo ========================================

echo Activation de l'environnement mon_env...
call conda activate mon_env

echo Installation des dependances...
pip install Cython==0.29.21
pip install librosa==0.8.0
pip install matplotlib==3.3.1
pip install numpy==1.18.5
pip install phonemizer==2.2.1
pip install scipy==1.5.2
pip install tensorboard==2.3.0
pip install torch==1.6.0
pip install torchvision==0.7.0
pip install Unidecode==1.1.1
pip install pysoundfile==0.9.0.post1
pip install jamo==0.4.1

echo Installation de espeak-ng...
echo IMPORTANT: Installez manuellement espeak-ng depuis https://github.com/espeak-ng/espeak-ng/releases

echo Compilation de Monotonic Alignment Search...
cd monotonic_align
python setup.py build_ext --inplace
cd ..

echo ========================================
echo Configuration terminee!
echo ========================================
