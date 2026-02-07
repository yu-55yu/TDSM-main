import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from tqdm import tqdm
import pandas as pd
import numpy as np

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

import random
import traceback
import sys

from utils import Train_Report
# [cite_start]从 model/dit.py 导入模型和部件索引定义 [cite: 75-78]
from model.dit import DiT, PART_INDICES 


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_data_loader = data_loader['test']

        # Accelerator 配置
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process:
            if args.work_dir is not None:
                os.makedirs(args.work_dir, exist_ok=True)

        # 权重数据类型
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # 模型初始化
        self.load_checkpoint()
        self.text_prompt() 
        
        # 显存优化：noise 形状为 [N, 25, C]，移除多余维度防止广播 OOM
        self.noise = torch.randn(self.args.num_noise, 25, self.args.in_channels).to(self.accelerator.device)
        self.dit = DiT(in_channels=self.args.in_channels, hidden_size=self.args.hidden_size, depth=self.args.depth, num_heads=self.args.num_heads)

        self.unseen_num = self.args.unseen_label
        self.unseen_labels = np.load(self.args.unseen_label_path)

        if 'ntu60' in self.args.unseen_label_path:
            self.total_classes = 60
            self.seen_labels = np.where(~np.isin(np.arange(60), self.unseen_labels))[0]
        else:  # ntu120
            self.total_classes = 120
            self.seen_labels = np.where(~np.isin(np.arange(120), self.unseen_labels))[0]

        self.seen_num = len(self.seen_labels)
        self.seen_label_mapping = {label: idx for idx, label in enumerate(self.seen_labels)}
        self.unseen_label_mapping = {label: idx for idx, label in enumerate(self.unseen_labels)}

        # 优化器与调度器
        params_to_opt = self.dit.parameters()
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.lr_scheduler = get_scheduler(self.args.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=self.args.num_warmup, num_training_steps=self.args.num_iter)

        self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader, self.test_data_loader = self.accelerator.prepare(
            self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader, self.test_data_loader)

    def load_checkpoint(self):
        noise_scheduler_config = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler").config
        noise_scheduler_config['num_train_timesteps'] = self.args.num_steps
        noise_scheduler_config['prediction_type'] = self.args.prediction_type
        self.noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)

    def text_prompt(self):
        """ 方案二：从 JSON 加载全局描述、部件描述及权重 [cite: 58-68] """
        csv_path = './data/class_lists/ntu60.csv' if self.args.unseen_label_path.find('ntu60') != -1 else './data/class_lists/ntu120.csv'
        df = pd.read_csv(csv_path)
        base_prompts = df['label'].values.tolist()
        
        # 加载部件 JSON 数据
        json_path = self.args.parts_json_path if hasattr(self.args, 'parts_json_path') else './data/class_lists/ntu_parts.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            self.parts_data = json.load(f)

        def get_clip_embed(text):
            inputs = self.tokenizer(text, padding="max_length", max_length=35, truncation=True, return_tensors="pt").to(self.accelerator.device)
            embeds = self.text_encoder(inputs.input_ids)
            return torch.cat((embeds['last_hidden_state'], embeds['pooler_output'].unsqueeze(1)), dim=1).detach().clone()

        self.text_embed = [] 
        self.part_weights = { 'hands': [], 'legs': [], 'torso': [] }

        for i in range(len(base_prompts)):
            action_key = str(i + 1)
            base_emb = get_clip_embed(base_prompts[i])
            global_desc = self.parts_data[action_key]['global']
            global_emb = get_clip_embed(global_desc)
            
            self.text_embed.append(torch.cat((base_emb, global_emb), dim=-1).to(self.accelerator.device))
            
            for p in ['hands', 'legs', 'torso']:
                w = self.parts_data[action_key]['parts'][p]['weight']
                self.part_weights[p].append(w)

        self.text_embed = torch.cat(self.text_embed, dim=0)
        for p in ['hands', 'legs', 'torso']:
            self.part_weights[p] = torch.tensor(self.part_weights[p]).to(self.accelerator.device)

    def save_best_model(self):
        """ 修复 AttributeError：保存最优训练状态 """
        save_path = os.path.join(self.args.work_dir, 'best')
        self.accelerator.save_state(save_path)

    def train(self, train_log, global_step):
        self.dit.train()
        self.dit.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        for idx, (features, labels) in tqdm(enumerate(self.train_data_loader)):
            batch_size = features.shape[0]
            with self.accelerator.accumulate(self.dit):
                with torch.no_grad():
                    # 修复广播 OOM：保持特征为 [B, 25, 256]
                    features = features.to(self.accelerator.device, dtype=self.weight_dtype)
                    text_embed = self.text_embed[labels.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                    
                    noise = torch.randn(features.shape, device=features.device).repeat(2, 1, 1)
                    timesteps = torch.randint(0, self.args.num_steps, (batch_size,), device=features.device).long().repeat(2)

                    labels_r = torch.Tensor(random.choices(self.seen_labels, k=batch_size)).to(self.accelerator.device)
                    text_embed_r = self.text_embed[labels_r.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                    mask = (labels != labels_r).to(self.accelerator.device, dtype=self.weight_dtype)

                    features = features.repeat(2, 1, 1)
                    text_embed = torch.cat((text_embed, text_embed_r), dim=0)
                    noisy_latents = self.noise_scheduler.add_noise(features, noise, timesteps)
                
                model_pred = self.dit(noisy_latents, timesteps, text_embed[:, -1, :], text_embed[:, :-1, :])

                if "sample" == self.args.prediction_type:
                    target = features
                elif "epsilon" == self.args.prediction_type:
                    target = noise
                elif "v_prediction" == self.args.prediction_type:
                    target = self.noise_scheduler.get_velocity(features, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.args.prediction_type}")

                # [cite_start]--- 方案二：部件感知 Loss 计算 [cite: 80-109] ---
                loss_diff = F.mse_loss(model_pred[:batch_size], target[:batch_size]) * self.args.d_weight

                pos_g = F.mse_loss(model_pred[:batch_size], target[:batch_size], reduction='none').mean(dim=(1, 2))
                neg_g = F.mse_loss(model_pred[batch_size:], target[batch_size:], reduction='none').mean(dim=(1, 2))
                loss_triplet_global = (torch.clamp(pos_g - neg_g + self.args.margin, min=0.0) * mask).mean()

                loss_triplet_local = 0
                for part, indices in PART_INDICES.items():
                    weights_p = self.part_weights[part][labels.long()]
                    pred_p = model_pred[:, indices, :]
                    target_p = target[:, indices, :]
                    
                    pos_p = F.mse_loss(pred_p[:batch_size], target_p[:batch_size], reduction='none').mean(dim=(1, 2))
                    neg_p = F.mse_loss(pred_p[batch_size:], target_p[batch_size:], reduction='none').mean(dim=(1, 2))
                    
                    loss_p = (torch.clamp(pos_p - neg_p + self.args.margin, min=0.0) * mask * weights_p).mean()
                    loss_triplet_local += loss_p

                g_w = self.args.global_weight if hasattr(self.args, 'global_weight') else 0.5
                l_w = self.args.local_weight if hasattr(self.args, 'local_weight') else 0.5
                loss = loss_diff + (loss_triplet_global * g_w + loss_triplet_local * l_w) * self.args.t_weight

                reduced_loss = self.accelerator.gather(loss).mean()
                reduced_loss_diff = self.accelerator.gather(loss_diff).mean()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(batch_size, reduced_loss.item(), reduced_loss_diff.item(), (loss_triplet_global + loss_triplet_local).item() / 2)

            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str(lr, period_time)
                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                save_path = os.path.join(self.args.work_dir, f'checkpoint-{global_step}')
                self.accelerator.save_state(save_path)

        return global_step

    def test(self):
        """ 测试阶段集成部件加权推理 """
        self.dit.eval()
        self.dit.requires_grad_(False)
        cnt = 0
        num = 0

        with torch.no_grad():
            for idx, (features, labels) in tqdm(enumerate(self.test_data_loader)):
                batch_size = features.shape[0]
                label_pred = torch.zeros((batch_size, self.unseen_num)).to(self.accelerator.device)
                mapped_labels = [self.unseen_label_mapping[label.item()] for label in labels]
                mapped_labels = torch.Tensor(mapped_labels).to(self.accelerator.device).long()

                t = torch.ones((batch_size)).to(self.accelerator.device) * self.args.idx_inference_step
                features = features.to(self.accelerator.device, dtype=self.weight_dtype)

                for j in range(self.args.num_noise):
                    noise = self.noise[j].repeat(batch_size, 1, 1)
                    noisy_latents = self.noise_scheduler.add_noise(features, noise, t.long())
                    
                    for idx_u, i in enumerate(self.unseen_labels):
                        text_embed = self.text_embed[int(i)].unsqueeze(0).repeat(batch_size, 1, 1).to(self.accelerator.device, dtype=self.weight_dtype)
                        noise_pred = self.dit(noisy_latents, t, text_embed[:, -1, :], text_embed[:, :-1, :])
                        
                        if "sample" == self.args.prediction_type:
                            target = features
                        elif "epsilon" == self.args.prediction_type:
                            target = noise
                        else:
                            target = self.noise_scheduler.get_velocity(features, noise, t.long())

                        # [cite_start]综合全局与部件评分 [cite: 104-106]
                        dist_global = F.mse_loss(noise_pred, target, reduction='none').mean(dim=(1, 2))
                        dist_parts = 0
                        for part, indices in PART_INDICES.items():
                            w = self.part_weights[part][int(i)]
                            mse_p = F.mse_loss(noise_pred[:, indices, :], target[:, indices, :], reduction='none').mean(dim=(1, 2))
                            dist_parts += mse_p * w
                        
                        label_pred[:, idx_u] += (dist_global * 0.5 + dist_parts * 0.5)

                cnt += torch.sum(torch.argmin(label_pred, -1) == mapped_labels)
                num += batch_size

        zsl_accuracy = float(cnt) / num
        return zsl_accuracy