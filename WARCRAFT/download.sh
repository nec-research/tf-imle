#!/usr/bin/env bash

mkdir -p data/
cd data/

rm -rf warcraft_shortest_path
wget -c http://data.neuralnoise.com/warcraft_maps.tar.gz
tar xvfz warcraft_maps.tar.gz
mv warcraft_shortest_path_oneskin warcraft_shortest_path
rm -f warcraft_maps.tar.gz

rm -rf lavacrossing_shortest_path
wget -c http://data.neuralnoise.com/lavacrossing_shortest_path.tar.gz
tar xvfz lavacrossing_shortest_path.tar.gz
rm -f lavacrossing_shortest_path.tar.gz

rm -rf simplecrossing_shortest_path
wget -c http://data.neuralnoise.com/simplecrossing_shortest_path.tar.gz
tar xvfz simplecrossing_shortest_path.tar.gz
rm -f simplecrossing_shortest_path.tar.gz
