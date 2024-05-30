
scene_name=$1           # ['3RScan', 'ARKitScenes', 'HM3D', 'MultiScan', 'ScanNet']
echo "Preprocessing scene $scene_name"
export SCAN_NAME=$scene_name
export CUDA_VISIBLE_DEVICES=$2

cd models/openscene

/home/admin/micromamba/envs/ll3da/bin/python run/preprocess.py \
 --config config/scannet/ours_openseg_pretrained.yaml \
 feature_type distill \
 save_folder results \