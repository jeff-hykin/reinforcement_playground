# globbing
setopt extended_glob &>/dev/null
shopt -s globstar &>/dev/null
shopt -s dotglob &>/dev/null

for item in "$PROJECTR_FOLDER/modules"*
do
    # quick/weak check that its a module
    if [[ -f "$item/setup.py" ]]
    then
        python3 -m pip --disable-pip-version-check install -e "$item"
    fi
done