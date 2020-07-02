res=0
for i in {1..100}; do

	restmp=$(./pdcp_benchmark | tail -1)
	res=$(($res+$restmp))
	echo $i
done
res=$(($res/100))
echo "mean time:"
echo $res
