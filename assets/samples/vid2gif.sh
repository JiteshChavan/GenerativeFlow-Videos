
set -euo pipefail

IN_DIR="${1:-.}"
OUT_DIR="${2:-../gifs}"
FPS_ARG="${3:-15}"   # "auto" = preserve fps or  12/15/20

mkdir -p "$OUT_DIR"

find "$IN_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.MP4" \) -print0 \
| while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  name="${base%.*}"
  out_gif="$OUT_DIR/${name}.gif"

  if [[ "$FPS_ARG" == "auto" ]]; then
    src_fps="$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "$f" \
      | awk -F'/' 'NF==2 { if ($2!=0) printf "%.0f", $1/$2; else print 15 } NF!=2 {print 15}')"
    FPS="$src_fps"
  else
    FPS="$FPS_ARG"
  fi

  echo "MP4 -> GIF: '$base' (fps=$FPS, keep-res) -> '$(basename "$out_gif")'"


  ffmpeg -y -nostdin -v warning -i "$f" \
    -filter_complex "[0:v]fps=${FPS},format=rgba,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
    "$out_gif"
done

echo "Done. GIFs saved to: $OUT_DIR"
