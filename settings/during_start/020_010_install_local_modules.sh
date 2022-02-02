{
    requirements_path="$FORNIX_FOLDER/requirements.txt"
    requirements="$(cat "$requirements_path")"
    relative_path="modules"
    # this loop is so stupidly complicated because of many inherent-to-shell reasons, for example: https://stackoverflow.com/questions/13726764/while-loop-subshell-dilemma-in-bash
    for_each_item_in="$FORNIX_FOLDER/$relative_path"; [ -z "$__NESTED_WHILE_COUNTER" ] && __NESTED_WHILE_COUNTER=0;__NESTED_WHILE_COUNTER="$((__NESTED_WHILE_COUNTER + 1))"; trap 'rm -rf "$__temp_var__temp_folder"' EXIT; __temp_var__temp_folder="$(mktemp -d)"; mkfifo "$__temp_var__temp_folder/pipe_for_while_$__NESTED_WHILE_COUNTER"; (find "$for_each_item_in" -maxdepth 1 ! -path "$for_each_item_in" -print0 2>/dev/null | sort -z > "$__temp_var__temp_folder/pipe_for_while_$__NESTED_WHILE_COUNTER" &); while read -d $'\0' each
    do
        # quick/weak check that its a module
        if [[ -f "$each/setup.py" ]]
        then
            entry="./$relative_path/$(basename "$each")"
            # make sure its not already in requirements.txt
            echo "$requirements" | grep "$entry" || {
                echo "
$entry" >> "$requirements_path"
            } &>/dev/null
        fi
    done < "$__temp_var__temp_folder/pipe_for_while_$__NESTED_WHILE_COUNTER";__NESTED_WHILE_COUNTER="$((__NESTED_WHILE_COUNTER - 1))"
}