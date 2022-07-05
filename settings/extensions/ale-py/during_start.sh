cat "$FORNIX_FOLDER/.gitignore" | grep 'commands/\*-NOTFOUND' &>/dev/null || echo '
commands/*-NOTFOUND' >> "$FORNIX_FOLDER/.gitignore"

cat "$FORNIX_FOLDER/requirements.txt" | grep '\./settings/extensions/ale-py/internals/ale-py' &>/dev/null || echo '
./settings/extensions/ale-py/internals/ale-py' >> "$FORNIX_FOLDER/requirements.txt"

if [ "$(uname)" = "Darwin" ] 
then
    echo '#!/usr/bin/env bash
'"'$(which ar)'"' "$@"' > "$FORNIX_FOLDER/commands/CMAKE_CXX_COMPILER_AR-NOTFOUND"; chmod u=rwx,g=rwx,o=rwx "$FORNIX_FOLDER/commands/CMAKE_CXX_COMPILER_AR-NOTFOUND"
    echo '#!/usr/bin/env bash
'"'$(which ranlib)'"' "$@"' > "$FORNIX_FOLDER/commands/CMAKE_CXX_COMPILER_RANLIB-NOTFOUND"; chmod u=rwx,g=rwx,o=rwx "$FORNIX_FOLDER/commands/CMAKE_CXX_COMPILER_RANLIB-NOTFOUND"
fi

