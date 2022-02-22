this_folder="$FORNIX_FOLDER/main/world_builders/carla"

mkdir -p "$this_folder/workspace.ignore/"

carla_download_path="$this_folder/workspace.ignore/carla_download.tar"
curl https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz --output "$carla_download_path"
{
    cd "$(dirname "$carla_download_path")"
    tar -xvf "$carla_download_path"
}