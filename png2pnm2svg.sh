for png in *.png
do
        convert "$png" "$png".pnm
done

echo "### (25%)"

rm *.png

echo "##### (50%)"

for pnm in *.pnm
do
        potrace "$pnm" -s -o "$pnm".svg
done

echo "######## (75%)"

rm *.pnm

echo "############(100%)"
