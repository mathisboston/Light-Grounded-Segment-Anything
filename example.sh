folder="path/to/dataset/images"
python scripts/grounded_sam_demo.py \
 --input_image $folder --output_dir $folder/masks \
 --text_prompt "person." \
 --sam_version "vit_h"  --sam_checkpoint "checkpoints/sam_hq_vit_h.pth" \
 --box_threshold 0.3 --overview --use_sam_hq