#!/bin/sh
sed 's/png/jpg/g' imagelist.txt | sed "s/img/img_jpg/g" > imagelist_jpg.txt
paste -d '\n' imagelist.txt imagelist_jpg.txt | parallel --eta --max-args=2 convert {1} -resize 640x360 {2}
