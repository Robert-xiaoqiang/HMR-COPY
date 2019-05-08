VIDEO_FILENAME=$1
INPUT_DIR=$(pwd)/meshes_from_video
FPS=$2
filename_basename=$(basename -- "$VIDEO_FILENAME")
filename_noext="${filename_basename%.*}"
echo "converting these images to video"
mkdir -p $INPUT_DIR
ffmpeg -i "$INPUT_DIR/$filename_noext%03d.jpg" -r $FPS $VIDEO_FILENAME
echo "finish converting with fps $FPS"
