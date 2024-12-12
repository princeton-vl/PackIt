wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
wgetgdrive 1778eDit0-6MpznaC5rGve0o1EQIyqhst tmp/env.zip
wgetgdrive 1pvcYEJrD8Jru617VuDzKYdpZJf5sTlmz tmp/env_viz.zip
wgetgdrive 1ZJCebs_wA3-Ry7TiB4vMfHTHMZv3QyLq tmp/env_viz_mac.zip
wgetgdrive 1Bo6hG05FHM7TC7BYKA1VAGHWiaxGD1JI tmp/model.zip
wgetgdrive 1T1P_GLgvPp_mF-tXGe_Yq_0FdARLF1R1 tmp/data.zip
wgetgdrive 1XY1TsJcwze5312F3XKjt8pwbFeoDLSYC tmp/results.zip

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
