VIDEO_FILENAME=$1
OUTPUT_DIR=$(pwd)/images_from_video/
FPS=$2

filename_basename=$(basename -- "$VIDEO_FILENAME")
filename_noext="${filename_basename%.*}"
	echo "converting this video to imges"
	mkdir -p $OUTPUT_DIR
	ffmpeg -i $VIDEO_FILENAME -r $FPS "$OUTPUT_DIR/$filename_noext%03d.jpg"
	# -qscale:3
	# -threads 8
	echo "finish converting with fps $FPS"
	# echo "generating json joints"
	# python ../RMPPE/2d_pose_estimation.py $OUTPUT_DIR
	# echo "json done"
