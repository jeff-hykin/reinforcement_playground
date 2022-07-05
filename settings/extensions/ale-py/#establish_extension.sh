#!/usr/bin/env bash

# 
# this is just a helper (common to most all extensions)
# 
relatively_link="$FORNIX_FOLDER/settings/extensions/#standard/commands/tools/file_system/relative_link"

# 
# connect during_start
# 
"$relatively_link" "$__THIS_FORNIX_EXTENSION_FOLDERPATH__/during_start.sh"                "$FORNIX_FOLDER/settings/during_start/019_001_setup_ale_py.sh"

# 
# connect during_clean
# 
"$relatively_link" "$__THIS_FORNIX_EXTENSION_FOLDERPATH__/during_clean.sh"                "$FORNIX_FOLDER/settings/during_clean/801_ale_py.sh"