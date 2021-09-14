python -m pip --disable-pip-version-check install "optuna>=2.9.1"

if [ "$OSTYPE" = "linux-gnu" ] 
then
    python -m pip --disable-pip-version-check install "carla-client-unofficial>=0.9.11"
else
    python -m pip --disable-pip-version-check install "carla>=0.9.12"
fi