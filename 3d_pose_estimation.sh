for f in images_from_video/*; do

	filename=$(basename -- "$f")
  no_ext="${filename%.*}"
  
  python demo.py --img_path $f \
                     --json_path jsons_from_video/$no_ext.json  
  
done

echo "Done"
