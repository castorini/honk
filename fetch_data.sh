#!/bin/sh
echo "Downloading data..."
wget -nc https://github.com/castorini/honk-data/archive/master.zip -O data.zip
wget -nc https://github.com/castorini/honk-models/archive/master.zip -O models.zip

echo "Extracting honk data..."
yes no | unzip data.zip
yes no | unzip models.zip
mv honk-data-master/training_data .
mv -T honk-models-master model

echo "Cleaning up..."
rm -rf honk-models-master
rm -rf honk-data-master 
