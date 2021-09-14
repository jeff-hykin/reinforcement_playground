python -m pip --disable-pip-version-check install "optuna>=2.9.1"

if [ "$OSTYPE" = "linux-gnu" ] 
then
    python -m pip --disable-pip-version-check install "carla-client-unofficial>=0.9.11"
else
    # there is no carla for mac :/ I'd have to automate building from source
fi