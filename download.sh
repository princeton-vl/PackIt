wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
wgetgdrive 1x6GbwLDSBcwmxSvhL1SY7oIW5x118sDF tmp/env.zip
wgetgdrive 1kTOcgB0Mz082v0eEkUNDMjVT2Bj1_E9s tmp/env_viz.zip
wgetgdrive 11aT7Lw1asQXYhBZUPSEt3Q7NxWhHUXZ1 tmp/env_viz_mac.zip
wgetgdrive 132G9nqMpOkTYfMaUihLpyocOp1ezx2On tmp/model.zip
wgetgdrive 1DD7bdRJSPbEIXe8qXzrD3lEz0nzfbGbU tmp/data.zip
wgetgdrive 1MLZ8g0w-TcFtwSMxwv8CwPpvTjV5zeJ9 tmp/results.zip

mkdir unity/envs
unzip tmp/env.zip -d unity/envs
unzip tmp/env_viz.zip -d unity/envs
unzip tmp/env_viz_mac.zip -d unity/envs

unzip tmp/data.zip
mv StreamingAssets data
ln -s ../../../data unity/envs/packit_Data/StreamingAssets
ln -s ../../../data unity/envs/packit_viz_Data/StreamingAssets
ln -s ../../../../../../data unity/envs/packit_viz_mac.app/Contents/Resources/Data/StreamingAssets

mkdir log
unzip tmp/model.zip -d log
mkdir log/unity

unzip tmp/results.zip

rm -r tmp
