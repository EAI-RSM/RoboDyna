#!/usr/bin/env bash
set -e

cd assets
python _download.py

# background_texture
unzip -o background_texture.zip
rm -rf background_texture.zip

# embodiments
unzip -o embodiments.zip
rm -rf embodiments.zip

# objects
unzip -o objects.zip
rm -rf objects.zip

cd ..
echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py