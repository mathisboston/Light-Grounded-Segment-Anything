folder="/data/mb273924/_DATASETS/sacamain/images_8"
python scripts/grounded_sam_demo.py \
 --input_image $folder --output_dir $folder/../masks_8 \
 --text_prompt "bag." \
 --sam_version "vit_b"  --sam_checkpoint "checkpoints/sam_hq_vit_b.pth" \
 --box_threshold 0.3 --overview --use_sam_hq