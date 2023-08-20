wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
wgetgdrive 11rTpF0uiBUHGqbeDa96noIXCM3dYWkOz tmp/env.zip
wgetgdrive 1cR6svpZGYaQS8DnIHQ2hFknbFXpGaoBj tmp/env_viz.zip
wgetgdrive 1usvzw2t-WijbXr9p4CnwrYIcnKf3AEdM tmp/env_viz_mac.zip
wgetgdrive 1WdaSNlpsGdcwtGO5sKICSbgrZnYzT2mE tmp/model.zip
wgetgdrive 1VpwiAx0i1sanmqHLf1TIe9re5LBVaPQW tmp/data.zip
wgetgdrive 1ntgGFZRKdC8BUAwUXMttnhyzL1dcKJ2J tmp/results.zip

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
