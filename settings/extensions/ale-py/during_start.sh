cat "$FORNIX_FOLDER/requirements.txt" | grep '\./settings/extensions/ale-py/internals/ale-py' &>/dev/null || echo '
./settings/extensions/ale-py/internals/ale-py' >> "$FORNIX_FOLDER/requirements.txt"


mkdir -p "$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore"

if [ "$(uname)" = "Darwin" ] 
then
    echo '#!/usr/bin/env bash
'"'$(which ar)'"' "$@"' > "$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore/CMAKE_CXX_COMPILER_AR-NOTFOUND"; chmod u=rwx,g=rwx,o=rwx "$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore/CMAKE_CXX_COMPILER_AR-NOTFOUND"
    echo '#!/usr/bin/env bash
'"'$(which ranlib)'"' "$@"' > "$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore/CMAKE_CXX_COMPILER_RANLIB-NOTFOUND"; chmod u=rwx,g=rwx,o=rwx "$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore/CMAKE_CXX_COMPILER_RANLIB-NOTFOUND"
fi

export PATH="$PATH:$FORNIX_FOLDER/settings/extensions/ale-py/commands.ignore/"