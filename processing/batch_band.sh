
for reference in ../bands_sorted/*.bed; do
    for query in ../query_beds/*.bed; do
        queryname=$(basename $query)
        reference_name=$(basename $reference)
        echo "${queryname%.*}"_band_"${reference_name%.*}".csv;
        python ../intersect.py $reference $query 100 "${queryname%.*}"_band_"${reference_name%.*}".csv;
    done;
done