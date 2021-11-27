#!/usr/bin/env bash

name="donkeycar"
link_to="$FORNIX_FOLDER/main/environments/donkey/servers"

temp_folder="$FORNIX_FOLDER/settings/.cache"
mkdir -p "$temp_folder"

unzip_output_path="$temp_folder/download.do_not_sync"
download_path="$unzip_output_path.zip"
# remove possibly corrupted things in the way
rm -f "$download_path"
rm -rf "$download_path"

# 
# download
# 
if [ "$OSTYPE" = "linux-gnu" ] 
then
    # linux version
    curl -L https://github.com/tawnkramer/gym-donkeycar/releases/download/v21.07.24/DonkeySimLinux.zip > "$download_path" \
        && unzip "$download_path" -d "$unzip_output_path"
else
    # mac version
    curl -L https://github.com/tawnkramer/gym-donkeycar/releases/download/v21.07.24/DonkeySimMac.zip > "$download_path" \
        && unzip "$download_path" -d "$unzip_output_path"
fi

if [ -d "$unzip_output_path" ]
then
    final_location="$FORNIX_FOLDER/resources/semi_permanent.do_not_sync"
    mkdir -p "$final_location"
    rm -rf "$final_location/$name" 2>/dev/null
    rm -f "$final_location/$name" 2>/dev/null
    mv "$unzip_output_path" "$final_location/$name"
    
    mkdir -p "$(dirname "$link_to")"
    rm -f "$link_to"
    rm -rf "$link_to"
    # link it to whereever it needs to be
    ln -s "$final_location/$name" "$link_to"
    # make the binaries executable
    chmod 777 "$final_location/$name/DonkeySimLinux/donkey_sim.x86_64" 2>/dev/null
    chmod 777 "$final_location/$name/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim" 2>/dev/null
fi
