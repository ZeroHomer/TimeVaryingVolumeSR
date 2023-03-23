#! /bin/sh
for i in *.zip
do
k=$i
s=${k%.zip*}
echo $s
unzip $i -d $s
done