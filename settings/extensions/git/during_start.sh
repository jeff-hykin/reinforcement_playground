# run the git subrepo setup
if [ -d "$FORNIX_FOLDER/settings/extensions/git/git_subrepo/" ]
then
    export GIT_SUBREPO_ROOT="$FORNIX_FOLDER/settings/extensions/git/git_subrepo/"
    export PATH="$GIT_SUBREPO_ROOT/lib:$PATH"
    export MANPATH="$GIT_SUBREPO_ROOT/man:$MANPATH"
    source "$GIT_SUBREPO_ROOT/share/enable-completion.sh"
fi