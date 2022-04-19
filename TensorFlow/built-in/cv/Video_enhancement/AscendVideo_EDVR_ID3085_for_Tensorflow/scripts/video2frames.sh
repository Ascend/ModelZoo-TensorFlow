#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 [-h] [-i VIDEO_PATH] [-o OUTPUT_PATH] [-f FILTERS] [-l LOG_DIR] [-q]"
   echo -e "\t-i: Input source video file."
   echo -e "\t-o: Output frames directory."
   echo -e "\t-f: Filters for deinterlacing the video when it's interlaced. Default 'bobweaver'."
   echo -e "\t-r: Frame rate if interlaced. Default '1' "
   echo -e "\t-l: Log directory."
   echo -q "\t-q: Quiet mode. Will not print any information."
   exit 1 # Exit script after printing help
}

while getopts "i:o:f:l:r:hq" opt
do
   case "$opt" in
      i ) video_path="$OPTARG" ;;
      o ) frames_dir="$OPTARG" ;;
      f ) filter="$OPTARG" ;;
      l ) log_dir="$OPTARG" ;;
      r ) rate="$OPTARG" ;;
      q ) quiet=true ;;
      h ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ x"$video_path" = x"" -o ! -f "$video_path" ]; then
  echo "[ERROR] Invalid video path: $video_path"
  helpFunction
fi

if [ x"$quiet" = x"" ]; then
  quiet=false
fi

if [ x"$rate" = x"" ]; then
  rate='1'
fi

timestamp=$(date '+%Y%m%dR%H%M%S')
if [ x"$log_dir" = x"" ]; then
  log_dir=/tmp/.mindvideo
fi

if [ ! -d "$log_dir" ]; then
  mkdir -p $log_dir
fi
report_log=${log_dir}/${timestamp}_probe.log
extract_log=${log_dir}/${timestamp}_video.log

# =======================================================================================
# check which type of videos:
#    progressive, pseudo-interlaced (will be treated as progressive), truly interlaced
# =======================================================================================
test_nframes=400
export FFREPORT=file=$extract_log
if [ -e $report_log ]; then
    rm $report_log
fi

ffmpeg -report -i ${video_path} -vframes $test_nframes -vf idet -f null - 2> $report_log
wait < <(jobs -p)

nframes=( $(cat $report_log | grep 'Multi frame detection: ' | grep -woP '(\d+)') )
# nframes: [tff, bff, progressive, undetermined]

n_frame_interlaced=$(awk -vp=${nframes[0]} -vq=${nframes[1]} 'BEGIN{printf "%d" ,p + q}')
# echo $n_frame_interlaced

if [ $n_frame_interlaced -gt ${nframes[2]} ]; then
    type=interlaced
else
    type=progressive
fi

# if it's progressive, use none filter regardless of the previous settings
if [ $type = "progressive" ]; then
  filter_name=none
elif [ x"$filter" = x"" ]; then
  # else if it's interlaced, used bobweaver as the default deintelacing filter
  filter_name=bobweaver
else
  filter_name=$filter
fi

if [ $quiet = "false" ]; then
  echo "[INFO] Video type: $type; Filter: ${filter_name}"
fi

# ======================================================================================
# extract frames from video with the given deinterlacing filter.
# record the fps first.
# ======================================================================================
fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate $video_path)
fps=$(echo "print(f'{$fps:.3f}')" | python3)
frames_fps=$fps
#n_total_frames=$(ffprobe -v error -select_streams v:0 -count_packets -of default=noprint_wrappers=1:nokey=1 -show_entries stream=nb_read_packets $video_path)

# determine the filter
if [ "$filter_name" = "bobweaver" ]; then
    if [ $type = "interlaced" ] && [ $rate = '2' ] ; then
        # deinterlace with 2x fps
        filter_cmd="-vf bwdif=1:0:0"
        frames_fps=$(echo "print(f'{2*$frames_fps:.3f}')" | python3)
    else
        filter_cmd="-vf bwdif=0:0:0"
    fi
elif [ "$filter_name" = "yadif" ]; then
    if [ $type = "interlaced" ] && [ $rate = '2' ] ; then
        # deinterlace with 2x fps
        filter_cmd="-vf yadif=1:0:0"
        frames_fps=$(echo "print(f'{2*$frames_fps:.3f}')" | python3)
    else
        filter_cmd="-vf yadif=0:0:0"
    fi
elif [ "$filter_name" = "QTGMC" ]; then
    if [ $type = "interlaced" ] && [ $rate = '2' ] ; then
        filter_cmd="50fps.vpy"
        frames_fps=$(echo "print(f'{2*$frames_fps:.3f}')" | python3)
    else
        filter_cmd="25fps.vpy"
    fi
elif [ "$filter_name" = "none" ]; then
    filter_cmd=""
fi

frames_dir=$frames_dir/${frames_fps}FPS_frames

if [ ! -d "$frames_dir" ]; then
  mkdir -p $frames_dir
fi

# ==============================================================================
# check whether is HDR
# ==============================================================================
COLORS=$(ffprobe -show_streams -v error "${video_path}" |egrep "^color_transfer|^color_space=|^color_primaries=" |head -3)
for C in $COLORS; do
  if [[ "$C" = "color_space="* ]]; then
    COLORSPACE=${C##*=}
  elif [[ "$C" = "color_transfer="* ]]; then
    COLORTRANSFER=${C##*=}
  elif [[ "$C" = "color_primaries="* ]]; then
    COLORPRIMARIES=${C##*=}
  fi
done

if [ "${COLORSPACE}" = "bt2020nc" ] && [ "${COLORTRANSFER}" = "smpte2084" ] && [ "${COLORPRIMARIES}" = "bt2020" ]; then
  ext='exr'
elif [ "${COLORSPACE}" = "bt2020nc" ] && [ "${COLORTRANSFER}" = "arib-std-b67" ] && [ "${COLORPRIMARIES}" = "bt2020" ]; then
  ext='exr'
else
  ext='png'
fi

if [ $quiet = "false" ]; then
  echo "[INFO] Extracting frames from ${video_path}. This may take a while."
  echo "[INFO] Cmd: ffmpeg -i ${video_path} $filter_cmd $frames_dir/%08d.${ext}"
fi

if [ "$filter_name" = "QTGMC" ]; then
    # This is only valid when in x86
    vspipe --y4m $filter_cmd -a "video_path=${video_path}" - | ffmpeg -i pipe: $frames_dir/%08d.${ext}
else
    ffmpeg -i ${video_path} $filter_cmd $frames_dir/%08d.${ext}
fi
wait < <(jobs -p)

echo "$type, ${fps}, ${frames_fps}"