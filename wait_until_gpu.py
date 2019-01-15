import os
import time
import subprocess

from gpustat import GPUStatCollection

if __name__ == "__main__":
    while True:
        gpu_stats = GPUStatCollection.new_query()
        available_gpus = []
        for gpu in gpu_stats.gpus:
            if gpu.memory_free > 9000:
                available_gpus.append(gpu.index)

        if available_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['{}'.format(_id) for _id in available_gpus])
            command = [
                'python',
                'train.py',
                '--model_variant="shufflenet_v2"',
                '--tf_initial_checkpoint=./checkpoints/shufflenet_v2_imagenet/model.ckpt-1661328',
                '--training_number_of_steps=200000',
                '--base_learning_rate=0.007',
                '--fine_tune_batch_norm=True',
                '--initialize_last_layer=False',
                '--output_stride=16',
                '--resize_factor=16',
                '--train_crop_size=224',
                '--train_crop_size=224',
                '--min_resize_value=224',
                '--max_resize_value=224',
                '--train_batch_size=30',
                '--dataset="coco"',
                '--dataset_dir=./dataset/records',
                '--train_logdir=./logs_pretrain_coco1',
                ]
            try:
                subprocess.check_call(command)
                break
            except:
                continue

            # run the script
        # visible in this process + all children
        time.sleep(10)
