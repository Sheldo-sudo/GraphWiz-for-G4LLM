CUDA_VISIBLE_DEVICES=0,1 python3 run_graphwiz.py \
    --model_path ../checkpoints/GraphWiz-LLaMA2-7B \
    --data_dir   ../dataset/GraphInstruct-Test \
    --save_dir   ../results/GraphWiz-LLaMA2-7B \
    --demos_dir  ../dataset/demos \
    --batch_size 8 \
    --max_tokens 1024