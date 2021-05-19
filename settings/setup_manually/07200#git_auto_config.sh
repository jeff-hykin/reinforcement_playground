# 
# link git config
# 
# if the project config exists
if [[ -f "$PROJECTR_FOLDER/settings/git/config" ]]
then
    # force the .git one to match
    rm -f "$PROJECTR_FOLDER/.git/config"
    # hardlink them together
    ln "$PROJECTR_FOLDER/settings/git/config" "$PROJECTR_FOLDER/.git/config"
# otherwise link the existing one to the project
elif [[ -f "$PROJECTR_FOLDER/.git/config" ]]
then
    mkdir -p "$PROJECTR_FOLDER/settings/git"
    ln "$PROJECTR_FOLDER/.git/config" "$PROJECTR_FOLDER/settings/git/config"
fi

# if there's no pull setting, then add it to the project
git config pull.rebase &>/dev/null || git config pull.ff &>/dev/null || git config --add pull.rebase false &>/dev/null

# 
# enable a hidden gitignore
# 
mkdir -p "$PROJECTR_FOLDER/.git/info/"
# check if file exists
if [[ -f "$PROJECTR_FOLDER/settings/git/exclude.ignore" ]]
then
    rm -f "$PROJECTR_FOLDER/.git/info/exclude"
    ln "$PROJECTR_FOLDER/settings/git/exclude.ignore" "$PROJECTR_FOLDER/.git/info/exclude"
fi