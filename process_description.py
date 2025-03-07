

#
#
#
#
# #
# # import torch
# # import torch.nn as nn
# # import pickle
# # import numpy as np
# #
# # # 假设映射字典文件的路径和超曲率嵌入文件的路径
# # description_embedding_file = "/root/autodl-fs/autodl-fs/bert/data/description.pkl"
# # hyperbolic_embedding_file = "/root/autodl-fs/autodl-fs/bert/data/hyperbolic_vectors_all.pkl"
# #
# # # 1. 加载映射字典文件
# # with open(description_embedding_file, 'rb') as f:
# #     description_embedding_dict = pickle.load(f)
# #
# # # 2. 加载超曲率嵌入文件
# # with open(hyperbolic_embedding_file, 'rb') as f:
# #     chapter_vectors, block_vectors, category_vectors, code_vectors = pickle.load(f)
# #
# # # 将超曲率嵌入转化为字典
# # hyperbolic_embedding_dict = {
# #     'Chapter': chapter_vectors,
# #     'Block': block_vectors,
# #     'Category': category_vectors,
# #     'Code': code_vectors
# # }
# #
# #
# # # 3. 融合描述嵌入和超曲率嵌入，并保持维度 768
# # class EmbeddingFusion:
# #     def __init__(self, desc_dim=768, hyper_dim=50, output_dim=768):
# #         # 初始化线性层，输入维度是 desc_dim + hyper_dim，输出维度是 output_dim
# #         self.linear_layer = nn.Linear(desc_dim + hyper_dim, output_dim)
# #
# #     def fuse_embeddings(self, desc_embedding, hyper_embedding):
# #         # 拼接描述嵌入和超曲率嵌入
# #         concatenated_embedding = torch.cat([desc_embedding, hyper_embedding], dim=-1)
# #         # 线性变换
# #         fused_embedding = self.linear_layer(concatenated_embedding)
# #         return fused_embedding
# #
# #     def process_and_save(self, description_embedding_dict, hyperbolic_embedding_dict, output_file):
# #         # 初始化一个保存结果的字典
# #         fused_embedding_dict = {}
# #         count = 0  # 初始化计数器
# #
# #         for level, entities in description_embedding_dict.items():
# #             if level in hyperbolic_embedding_dict:
# #                 level_hyper_embeds = hyperbolic_embedding_dict[level]
# #                 fused_embedding_dict[level] = {}
# #
# #                 for entity_id, desc_embedding in entities.items():
# #                     # 检查描述嵌入是否存在
# #                     if isinstance(desc_embedding, dict) and 'description_embedding' in desc_embedding:
# #                         desc_tensor = torch.tensor(desc_embedding['description_embedding']).float()
# #                         # 获取该实体的超曲率嵌入
# #                         hyper_tensor = torch.tensor(level_hyper_embeds[int(entity_id[1:])]).float()
# #
# #                         # 融合嵌入
# #                         fused_tensor = self.fuse_embeddings(desc_tensor, hyper_tensor)
# #                         fused_embedding_dict[level][entity_id] = fused_tensor.detach().numpy()
# #
# #                         # 打印前10个并停止
# #                         if count < 10:
# #                             print(f"Level: {level}, Entity: {entity_id}, Fused Embedding: {fused_tensor}")
# #                             count += 1
# #
# #                     else:
# #                         print(f"Warning: 'description_embedding' not found for {entity_id} in {level}")
# #
# #             # 检查是否已打印了10个并停止外层循环
# #             if count >= 10:
# #                 break
# #
# #         # 保存融合后的嵌入字典
# #         with open(output_file, 'wb') as f:
# #             pickle.dump(fused_embedding_dict, f)
# #         print(f"Fused embeddings saved to {output_file}")
# #
# #
# # # 4. 初始化 EmbeddingFusion 实例
# # fuser = EmbeddingFusion()
# #
# # # 5. 处理并保存融合后的嵌入
# # output_fused_embedding_file = "/root/autodl-fs/autodl-fs/bert/data/fused_embeddings.pkl"  # 替换为保存路径
# # fuser.process_and_save(description_embedding_dict, hyperbolic_embedding_dict, output_fused_embedding_file)
# # import pickle
# # import torch
# # import torch.nn as nn
# #
# # # 1. 文件路径（替换为实际路径）
# # description_embeddings_file = "/root/autodl-fs/autodl-fs/bert/data/description_embeddings.pkl"  # 描述嵌入文件
# # hyperbolic_embeddings_file = "/root/autodl-fs/autodl-fs/bert/data/hyperbolic_embeddings.pkl"  # 超曲率嵌入文件
# # fused_embeddings_file = "/root/autodl-fs/autodl-fs/bert/data/fused_embeddings.pkl"  # 保存融合后的文件
# #
# # # 2. 加载描述嵌入
# # with open(description_embeddings_file, 'rb') as f:
# #     description_embeddings = pickle.load(f)
# #
# # # 加载超曲率嵌入
# # with open(hyperbolic_embeddings_file, 'rb') as f:
# #     hyperbolic_embeddings = pickle.load(f)
# #
# # # 3. 初始化线性层融合
# # # 假设描述嵌入和超曲率嵌入的维度
# # desc_dim = 768
# # hyper_dim = 50
# # output_dim = 768  # 最终融合后的维度
# #
# # # 定义线性层用于融合
# # fusion_layer = nn.Linear(desc_dim + hyper_dim, output_dim)
# #
# #
# # def fuse_embeddings(desc_embedding, hyper_embedding):
# #     # 将嵌入转换为 Tensor
# #     desc_tensor = torch.tensor(desc_embedding).float()
# #     hyper_tensor = torch.tensor(hyper_embedding).float()
# #
# #     # 拼接描述嵌入和超曲率嵌入
# #     concatenated_embedding = torch.cat([desc_tensor, hyper_tensor], dim=-1)
# #
# #     # 线性融合
# #     fused_embedding = fusion_layer(concatenated_embedding).tolist()  # 转换为列表
# #     return fused_embedding
# #
# #
# # # 4. 融合嵌入
# # fused_embeddings = {}
# #
# # for level in description_embeddings.keys():
# #     fused_embeddings[level] = {}
# #     for entity_id in description_embeddings[level]:
# #         if entity_id in hyperbolic_embeddings[level]:
# #             # 获取对应的描述和超曲率嵌入
# #             desc_embedding = description_embeddings[level][entity_id]
# #             hyper_embedding = hyperbolic_embeddings[level][entity_id]
# #
# #             # 融合嵌入
# #             fused_embedding = fuse_embeddings(desc_embedding, hyper_embedding)
# #
# #             # 保存融合后的嵌入
# #             fused_embeddings[level][entity_id] = fused_embedding
# #         else:
# #             print(f"Warning: {entity_id} not found in hyperbolic embeddings for level {level}")
# #
# # # 5. 保存融合后的嵌入
# # with open(fused_embeddings_file, 'wb') as f:
# #     pickle.dump(fused_embeddings, f)
# #
# # print(f"Fused embeddings saved to {fused_embeddings_file}")
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import re

# 1. 使用 BioBERT 提取嵌入
def load_biobert(model_path="/root/autodl-fs/autodl-fs/bert/clinical_finetuned/best_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

def get_biobert_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] token 的最后一层隐藏状态作为句子嵌入
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
    return cls_embedding.squeeze(0).cpu().numpy()

# 2. 处理层次结构树描述，生成嵌入
def generate_embeddings(df, level_columns, tokenizer, model, device):
    embeddings_dict = {level: {} for level in level_columns.keys()}

    # 去重后生成唯一描述特征
    unique_rows = df.drop_duplicates(subset=[col_info[1] for col_info in level_columns.values()])

    for _, row in tqdm(unique_rows.iterrows(), total=len(unique_rows)):
        for level, col_info in level_columns.items():
            description_col, id_col = col_info
            description = str(row[description_col])
            # 提取 BioBERT 嵌入
            embedding = get_biobert_embedding(description, tokenizer, model, device)
            embeddings_dict[level][row[id_col]] = embedding.tolist()

    return embeddings_dict

# 3. 保存嵌入到文件，确保 ICD_CODE 保留两位小数
def save_embeddings(embeddings_dict, output_file, level_encoders):
    formatted_embeddings = {}
    for level, embeddings in embeddings_dict.items():
        encoder = level_encoders[level]

        # 遍历每个 ID，并确保 ID 存在于 encoder.classes_ 中
        formatted_embeddings[level] = {}
        for original_id in embeddings.keys():
            if original_id in encoder.classes_:
                index = encoder.transform([original_id])[0]
                formatted_id = encoder.inverse_transform([index])[0]

                # 确保 ICD_CODE 保留两位小数
                if level == 'Code':
                    formatted_id = ensure_two_decimal_places(formatted_id)

                formatted_embeddings[level][formatted_id] = embeddings[original_id]
            else:
                print(f"Warning: ID {original_id} not found in encoder for level {level}")
                formatted_embeddings[level][original_id] = embeddings[original_id]

        # 打印每层级的形状
        print(f"Level '{level}' embedding shape: ({len(formatted_embeddings[level])}, {len(next(iter(formatted_embeddings[level].values())))})")

    # 保存 embeddings
    with open(output_file, 'wb') as f:
        pickle.dump(formatted_embeddings, f)
    print(f"Embeddings saved to {output_file}")

# 处理 ICD_CODE 的函数，确保所有 ID 都保留两位小数并清除空格
def ensure_two_decimal_places(icd_code):
    icd_code = str(icd_code).strip()
    match = re.match(r'([A-Za-z]*)(\d*\.?\d*)', icd_code)
    if match:
        prefix = match.group(1)
        number_part = match.group(2)
        if '.' in number_part:
            formatted_number = f"{float(number_part):.2f}"
        else:
            formatted_number = f"{number_part}.00"
        return f"{prefix}{formatted_number}"
    else:
        return icd_code

# 4. 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_biobert()
    model.to(device)

    description_file = "/root/autodl-fs/autodl-fs/bert/data/zy3.csv"
    df = pd.read_csv(description_file, dtype={'ICD_CODE': str})
    nan_rows = df[df[['chapter_id', 'block_id', 'categories_id', 'ICD_CODE']].isnull().any(axis=1)]
    print(nan_rows)

    df['ICD_CODE'] = df['ICD_CODE'].apply(ensure_two_decimal_places)

    level_columns = {
        'Chapter': ('chapter_disprition', 'chapter_id'),
        'Block': ('block_disprition', 'block_id'),
        'Category': ('categories_disprition', 'categories_id'),
        'Code': ('Description', 'ICD_CODE')
    }

    chapter_encoder = LabelEncoder()
    block_encoder = LabelEncoder()
    categories_encoder = LabelEncoder()
    code_encoder = LabelEncoder()

    chapter_encoder.fit(df['chapter_id'].unique())
    block_encoder.fit(df['block_id'].unique())
    categories_encoder.fit(df['categories_id'].unique())
    code_encoder.fit(df['ICD_CODE'].unique())

    level_encoders = {
        'Chapter': chapter_encoder,
        'Block': block_encoder,
        'Category': categories_encoder,
        'Code': code_encoder
    }

    # 使用 BioBERT 模型生成嵌入
    embeddings_dict = generate_embeddings(df, level_columns, tokenizer, model, device)

    output_file = "/root/autodl-fs/autodl-fs/biobert_embeddings.pkl"
    save_embeddings(embeddings_dict, output_file, level_encoders)

