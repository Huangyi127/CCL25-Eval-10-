import json
import torch
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split



# --- 1. 配置参数 ---
class Config:
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 或 "bert-base-chinese"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 1.5e-5
    NUM_EPOCHS = 2
    WARMUP_STEPS = 0.1  # 学习率预热比例
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    # 注意：这里的 Group 标签顺序和名称要与实际数据保持一致，且 'non-hate' 通常放在最后
    GROUP_LABELS = ["Sexism", "Racism", "Region", "LGBTQ", "Others", "non-hate"]
    SPAN_LABELS = ["O", "B-TARGET", "I-TARGET", "B-ARGUMENT", "I-ARGUMENT"]

 # 损失权重 (可以根据任务重要性调整)
    SPAN_LOSS_WEIGHT = 1.2
    GROUP_LOSS_WEIGHT = 0.7 # 适当提高分类权重，如果分类重要
    HATEFUL_LOSS_WEIGHT = 1.0 # 如果判断是否仇恨是核心，可以提高
    # 增加 Early Stopping 配置
    EARLY_STOPPING_PATIENCE = 5 # 增加耐心值
    GRADIENT_ACCUMULATION_STEPS = 8 # 启用梯度累积，模拟 BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 16 的有效批量大小
# --- 2. 数据加载与预处理 ---

def parse_output_string(output_str):
    """
    解析 output 字符串，提取 Target, Argument, Group, Hateful 信息。
    示例: "没爹的黑孩 | 到处扔 | Racism | hate [END]"
    或 "Target1|Argument1|Group1|Hateful1 [SEP] Target2|Argument2|Group2|Hateful2 [END]"
    返回: list of (span_type, span_text, group_label, hateful_label)
    """
    if output_str is None or output_str.strip() == "":
        return []

    results = []
    output_str = output_str.replace(" [END]", "").strip()
    quad_strs = output_str.split(" [SEP] ")

    for q_str in quad_strs:
        parts = [p.strip() for p in q_str.split('|')]
        if len(parts) >= 4:
            target_text = parts[0] if parts[0] else None
            argument_text = parts[1] if parts[1] else None
            group_label = parts[2]
            hateful_label = parts[3]

            if target_text:
                results.append(("TARGET", target_text, group_label, hateful_label))
            if argument_text:
                results.append(("ARGUMENT", argument_text, group_label, hateful_label))
            # 如果 Target 和 Argument 都为空，但有分类信息，也添加
            if not target_text and not argument_text:
                results.append(("NO_SPAN", None, group_label, hateful_label))

    return results


def get_span_labels(text, spans_info, tokenizer, max_length, span_type="TARGET"):
    """
    为给定文本和span信息生成B-I标签序列。
    text: 原始文本字符串
    spans_info: 经过 parse_output_string 得到的列表，形如 (span_type, span_text, group_label, hateful_label)
    tokenizer: 分词器
    max_length: 最大序列长度
    span_type: 期望提取的 span 类型，如 "TARGET" 或 "ARGUMENT"
    """
    tokenized_text = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length,
                               return_offsets_mapping=True)

    actual_token_length = len(tokenized_text['input_ids'])
    offset_mapping = tokenized_text['offset_mapping']

    labels = [Config.SPAN_LABELS.index("O")] * max_length

    for s_type, s_text, _, _ in spans_info:
        if s_type != span_type or s_text is None:
            continue

        for match in re.finditer(re.escape(s_text), text):
            char_start, char_end = match.span()

            for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
                if i >= actual_token_length:
                    break

                # 跳过特殊 token ([CLS], [SEP], padding) 的 (0,0) 偏移，但保留第一个 [CLS]
                if token_char_start == 0 and token_char_end == 0 and i != 0:
                    continue

                # 更标准的 BIO 标签分配逻辑
                if max(token_char_start, char_start) < min(token_char_end, char_end):  # Token and span overlap
                    if token_char_start == char_start:  # Token starts with span
                        labels[i] = Config.SPAN_LABELS.index(f"B-{span_type}")
                    elif token_char_start > char_start:  # Token starts after span's beginning (inside span)
                        labels[i] = Config.SPAN_LABELS.index(f"I-{span_type}")

    return labels


class HateSpeechDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_data = data_list

        self.data = []
        for item in self.raw_data:
            content = item['content']
            output_str = item.get('output', None)

            if output_str is not None:
                parsed_info = parse_output_string(output_str)

                target_labels = get_span_labels(content, parsed_info, tokenizer, max_length, "TARGET")
                argument_labels = get_span_labels(content, parsed_info, tokenizer, max_length, "ARGUMENT")

                group_label_id = Config.GROUP_LABELS.index("non-hate")
                hateful_label_id = 0

                if parsed_info:
                    # 优先使用实际抽取到的span的分类信息，如果没有则使用NO_SPAN的分类信息
                    found_classification_for_spans = False
                    for s_type, s_text, g_label, h_label in parsed_info:
                        if s_type != "NO_SPAN":  # 如果找到了具体的 span，则以此分类为准
                            if g_label in Config.GROUP_LABELS:
                                group_label_id = Config.GROUP_LABELS.index(g_label)
                            hateful_label_id = 1 if h_label == "hate" else 0
                            found_classification_for_spans = True
                            break
                    # 如果所有解析的 quads 都是 NO_SPAN 类型，则使用第一个 NO_SPAN 的分类信息
                    if not found_classification_for_spans and parsed_info and parsed_info[0][0] == "NO_SPAN":
                        if parsed_info[0][2] in Config.GROUP_LABELS:
                            group_label_id = Config.GROUP_LABELS.index(parsed_info[0][2])
                        hateful_label_id = 1 if parsed_info[0][3] == "hate" else 0
            else:  # 如果 output_str 不存在 (测试集情况)
                target_labels = [Config.SPAN_LABELS.index("O")] * self.max_length
                argument_labels = [Config.SPAN_LABELS.index("O")] * self.max_length
                group_label_id = Config.GROUP_LABELS.index("non-hate")  # 占位符，实际预测时会改变
                hateful_label_id = 0  # 占位符，实际预测时会改变

            encoding = self.tokenizer(
                content,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_offsets_mapping=True
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            token_type_ids = encoding.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.squeeze(0)

            offset_mapping = encoding['offset_mapping'].squeeze(0)

            combined_labels = [Config.SPAN_LABELS.index("O")] * self.max_length
            for i in range(self.max_length):
                if target_labels[i] != Config.SPAN_LABELS.index("O"):
                    combined_labels[i] = target_labels[i]
                elif argument_labels[i] != Config.SPAN_LABELS.index("O"):
                    combined_labels[i] = argument_labels[i]

            self.data.append({
                'id': item['id'],
                'content': content,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(combined_labels, dtype=torch.long),
                'group_labels': torch.tensor(group_label_id, dtype=torch.long),
                'hateful_labels': torch.tensor(hateful_label_id, dtype=torch.long),
                'offset_mapping': offset_mapping
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- 3. 模型定义 ---
class HateSpeechDetector(torch.nn.Module):
    def __init__(self, num_group_labels, num_span_labels):
        super(HateSpeechDetector, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.1)

        self.group_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_group_labels)
        self.hateful_classifier = torch.nn.Linear(self.bert.config.hidden_size, 2)  # 0: non-hate, 1: hate

        self.span_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_span_labels)

        self.crf = CRF(num_span_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, group_labels=None,
                hateful_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(outputs.pooler_output)

        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        emissions = self.span_classifier(sequence_output)

        total_loss = 0

        if group_labels is not None:
            group_loss = CrossEntropyLoss()(group_logits, group_labels)
            total_loss += group_loss

        if hateful_labels is not None:
            hateful_loss = CrossEntropyLoss()(hateful_logits, hateful_labels)
            total_loss += hateful_loss

        if labels is not None:
            span_loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            total_loss += span_loss

        return emissions, group_logits, hateful_logits, total_loss


# --- 4. 训练函数 ---
def train_model(model, dataloader, val_dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        token_type_ids = batch['token_type_ids'].to(Config.DEVICE) if batch['token_type_ids'] is not None else None
        labels = batch['labels'].to(Config.DEVICE)
        group_labels = batch['group_labels'].to(Config.DEVICE)
        hateful_labels = batch['hateful_labels'].to(Config.DEVICE)

        optimizer.zero_grad()

        emissions, group_logits, hateful_logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            group_labels=group_labels,
            hateful_labels=hateful_labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")


# --- 5. 评估函数 ---

def extract_spans_from_bio(text, bio_labels_indices, offset_mapping, id_to_label_map, span_type_prefix="TARGET"):
    """
    根据BIO标签和offset_mapping从原始文本中提取实体片段。
    text: 原始文本字符串
    bio_labels_indices: 模型预测的token标签索引列表 (e.g., [0, 1, 2, 0, ...])
    offset_mapping: tokenizer返回的offset_mapping ([(start_char, end_char), ...])
    id_to_label_map: 标签索引到标签字符串的映射 (e.g., {0: 'O', 1: 'B-TARGET', ...})
    span_type_prefix: 期望提取的span类型前缀，例如 "TARGET" 或 "ARGUMENT"
    """
    extracted_spans = []
    current_span_start_char = -1
    current_span_end_char = -1

    min_len = min(len(bio_labels_indices), len(offset_mapping))

    for i in range(min_len):
        label_id = bio_labels_indices[i]
        char_start, char_end = offset_mapping[i]

        if char_start == 0 and char_end == 0 and i != 0:
            if current_span_start_char != -1:
                extracted_spans.append(text[current_span_start_char:current_span_end_char])
                current_span_start_char = -1
                current_span_end_char = -1
            continue

        label_str = id_to_label_map.get(label_id, "O")

        if label_str == f"B-{span_type_prefix}":
            if current_span_start_char != -1:  # End previous span if any
                extracted_spans.append(text[current_span_start_char:current_span_end_char])
            current_span_start_char = char_start
            current_span_end_char = char_end
        elif label_str == f"I-{span_type_prefix}" and current_span_start_char != -1:
            current_span_end_char = char_end
        else:  # O or other B/I
            if current_span_start_char != -1:  # End current span if any
                extracted_spans.append(text[current_span_start_char:current_span_end_char])
            current_span_start_char = -1
            current_span_end_char = -1

    if current_span_start_char != -1:  # Add last span if it extends to the end
        extracted_spans.append(text[current_span_start_char:current_span_end_char])

    cleaned_spans = []
    seen_spans = set()
    for span in extracted_spans:
        clean_span = span.strip()
        if clean_span and clean_span not in seen_spans:
            cleaned_spans.append(clean_span)
            seen_spans.add(clean_span)

    return cleaned_spans


def calculate_pr_f1(tp, pred, true):
    """辅助函数，计算精确率、召回率和F1"""
    precision = tp / pred if pred > 0 else 0
    recall = tp / true if true > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def calculate_f1_scores(true_data_list, pred_data_list):
    """
    计算 F1 分数 (Hard F1 和 Soft F1)。
    true_data_list: list of {'id': id, 'targets': [], 'arguments': [], 'content': content}
    pred_data_list: list of {'id': id, 'targets': [], 'arguments': []}
    """
    pred_dict = {item['id']: item for item in pred_data_list}

    total_true_targets = 0
    total_pred_targets = 0
    total_true_pos_targets_hard = 0
    total_true_pos_targets_soft = 0

    total_true_arguments = 0
    total_pred_arguments = 0
    total_true_pos_arguments_hard = 0
    total_true_pos_arguments_soft = 0

    for true_item in true_data_list:
        doc_id = true_item['id']
        true_targets = set(true_item['targets'])
        true_arguments = set(true_item['arguments'])

        pred_item = pred_dict.get(doc_id, {'targets': [], 'arguments': []})
        pred_targets = set(pred_item['targets'])
        pred_arguments = set(pred_item['arguments'])

        total_true_targets += len(true_targets)
        total_pred_targets += len(pred_targets)
        total_true_arguments += len(true_arguments)
        total_pred_arguments += len(pred_arguments)

        # 硬匹配计算
        total_true_pos_targets_hard += len(true_targets.intersection(pred_targets))
        total_true_pos_arguments_hard += len(true_arguments.intersection(pred_arguments))

        # 软匹配计算
        for gt_target in true_targets:  # Iterate over true_targets directly
            for pred_target in pred_targets:  # Iterate over pred_targets directly
                # 软匹配条件：真实span包含预测span，或预测span包含真实span，或两者有词语交集
                if (gt_target in pred_target or pred_target in gt_target or
                        set(gt_target.split()).intersection(set(pred_target.split()))):
                    total_true_pos_targets_soft += 1
                    break  # Count this true_target as matched once

        for gt_arg in true_arguments:  # Iterate over true_arguments directly
            for pred_arg in pred_arguments:  # Iterate over pred_arguments directly
                if (gt_arg in pred_arg or pred_arg in gt_arg or
                        set(gt_arg.split()).intersection(set(pred_arg.split()))):
                    total_true_pos_arguments_soft += 1
                    break  # Count this true_argument as matched once

    # 分别计算Target和Argument的硬匹配和软匹配F1
    f1_targets_hard = calculate_pr_f1(total_true_pos_targets_hard, total_pred_targets, total_true_targets)
    f1_arguments_hard = calculate_pr_f1(total_true_pos_arguments_hard, total_pred_arguments, total_true_arguments)

    f1_targets_soft = calculate_pr_f1(total_true_pos_targets_soft, total_pred_targets, total_true_targets)
    f1_arguments_soft = calculate_pr_f1(total_true_pos_arguments_soft, total_pred_arguments, total_true_arguments)

    # 按照比赛要求，计算硬匹配和软匹配的平均F1
    hard_f1 = (f1_targets_hard + f1_arguments_hard) / 2
    soft_f1 = (f1_targets_soft + f1_arguments_soft) / 2

    # 最终的评价指标是硬匹配F1和软匹配F1的平均值（单一维度）
    avg_f1 = (hard_f1 + soft_f1) / 2
    return hard_f1, soft_f1, avg_f1


def eval_model(model, dataloader, dataset):
    model.eval()
    all_predictions = []
    # 准备真实标签数据
    true_data_for_f1 = []
    for item in dataset.raw_data:
        targets = []
        arguments = []
        parsed_info = parse_output_string(item.get('output', ""))
        for s_type, s_text, _, _ in parsed_info:
            if s_type == "TARGET" and s_text:
                targets.append(s_text)
            elif s_type == "ARGUMENT" and s_text:
                arguments.append(s_text)
        true_data_for_f1.append(
            {'id': item['id'], 'targets': targets, 'arguments': arguments, 'content': item['content']})

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(Config.DEVICE) if batch['token_type_ids'] is not None else None
            original_contents = batch['content']
            original_ids = batch['id']

            emissions, group_logits, hateful_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            predicted_span_labels_batch = model.crf.decode(emissions, mask=attention_mask.byte())

            for i in range(input_ids.size(0)):
                content_id = original_ids[i].item()
                content_text = original_contents[i]

                sample_pred_labels = predicted_span_labels_batch[i]
                sample_offset_mapping = batch['offset_mapping'][i].cpu().numpy()

                offset_mapping_list = [(int(start_off), int(end_off)) for start_off, end_off in sample_offset_mapping]

                pred_targets = extract_spans_from_bio(
                    content_text,
                    sample_pred_labels,
                    offset_mapping_list,
                    {idx: label for idx, label in enumerate(Config.SPAN_LABELS)},
                    span_type_prefix="TARGET"
                )
                pred_arguments = extract_spans_from_bio(
                    content_text,
                    sample_pred_labels,
                    offset_mapping_list,
                    {idx: label for idx, label in enumerate(Config.SPAN_LABELS)},
                    span_type_prefix="ARGUMENT"
                )
                all_predictions.append({'id': content_id, 'targets': pred_targets, 'arguments': pred_arguments})

    hard_f1, soft_f1, avg_f1 = calculate_f1_scores(true_data_for_f1, all_predictions)
    return hard_f1, soft_f1, avg_f1


def predict(model, dataloader):
    model.eval()
    predictions = []
    # 获取 'non-hate' 对应的 group 标签索引
    non_hate_group_idx = Config.GROUP_LABELS.index("non-hate")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting", unit="batch")):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(Config.DEVICE) if batch['token_type_ids'] is not None else None
            original_contents = batch['content']
            original_ids = batch['id']

            emissions, group_logits, hateful_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            predicted_span_labels_batch = model.crf.decode(emissions, mask=attention_mask.byte())

            # 获取分类预测的原始 logits (用于后续修正)
            # predicted_group_labels_ids = torch.argmax(group_logits, dim=-1).cpu().numpy() # No need for argmax directly here
            predicted_hateful_labels_ids = torch.argmax(hateful_logits, dim=-1).cpu().numpy()

            for i in range(input_ids.size(0)):
                content_id = original_ids[i].item()
                content_text = original_contents[i]

                pred_hateful_id = predicted_hateful_labels_ids[i]
                hateful_label_str = "hate" if pred_hateful_id == 1 else "non-hate"

                group_label_str = ""
                if hateful_label_str == "hate":
                    # 如果预测为仇恨言论，则目标群体不能是 'non-hate'
                    # 从当前样本的 group_logits 中，将 'non-hate' 对应的分数设为极小值
                    current_sample_group_logits = group_logits[i].clone().detach().cpu()
                    current_sample_group_logits[non_hate_group_idx] = -float('inf')  # 排除 non-hate 选项

                    # 重新选择得分最高的仇恨类别
                    pred_group_id_corrected = torch.argmax(current_sample_group_logits).item()
                    group_label_str = Config.GROUP_LABELS[pred_group_id_corrected]

                    # 极端情况的 fallback：如果所有仇恨类别的 logits 也都非常低，导致修正后仍然指向 non-hate
                    # (理论上，如果训练数据充分，这种情况很少发生，但为了鲁棒性可以加)
                    if group_label_str == "non-hate":
                        group_label_str = "Others"  # 强制设为一个具体的仇恨类别
                else:
                    # 如果预测为非仇恨言论，则目标群体必须是 'non-hate'
                    group_label_str = "non-hate"

                # 提取 Span
                sample_pred_labels = predicted_span_labels_batch[i]
                sample_offset_mapping = batch['offset_mapping'][i].cpu().numpy()
                offset_mapping_list = [(int(start_off), int(end_off)) for start_off, end_off in sample_offset_mapping]

                pred_targets = extract_spans_from_bio(
                    content_text,
                    sample_pred_labels,
                    offset_mapping_list,
                    {idx: label for idx, label in enumerate(Config.SPAN_LABELS)},
                    span_type_prefix="TARGET"
                )
                pred_arguments = extract_spans_from_bio(
                    content_text,
                    sample_pred_labels,
                    offset_mapping_list,
                    {idx: label for idx, label in enumerate(Config.SPAN_LABELS)},
                    span_type_prefix="ARGUMENT"
                )

                output_quads_list = []
                seen_quads = set()  # 确保四元组不重复

                # 根据预测的 Target 和 Argument 生成四元组，同时处理空值情况
                if pred_targets or pred_arguments:
                    if pred_targets and pred_arguments:
                        for t in pred_targets:
                            for a in pred_arguments:
                                quad_str = f"{t.strip()} | {a.strip()} | {group_label_str} | {hateful_label_str}"
                                if quad_str not in seen_quads:
                                    output_quads_list.append(quad_str)
                                    seen_quads.add(quad_str)
                    elif pred_targets:  # 只找到 Target
                        # 如果只有 Target，且 Argument 是空的，用原始文本作为 Argument 的 fallback
                        fallback_argument = content_text.strip() if content_text.strip() else ""
                        for t in pred_targets:
                            quad_str = f"{t.strip()} | {fallback_argument} | {group_label_str} | {hateful_label_str}"
                            if quad_str not in seen_quads:
                                output_quads_list.append(quad_str)
                                seen_quads.add(quad_str)
                    elif pred_arguments:  # 只找到 Argument
                        # 如果只有 Argument，且 Target 是空的，用原始文本作为 Target 的 fallback
                        fallback_target = content_text.strip() if content_text.strip() else ""
                        for a in pred_arguments:
                            quad_str = f"{fallback_target} | {a.strip()} | {group_label_str} | {hateful_label_str}"
                            if quad_str not in seen_quads:
                                output_quads_list.append(quad_str)
                                seen_quads.add(quad_str)
                else:  # 模型既没有找到 Target 也没有找到 Argument
                    # 强制使用原始文本作为 Target 和 Argument 的内容，避免空白项
                    fallback_span = content_text.strip()
                    if fallback_span:  # 确保 fallback 内容不为空
                        # 既没有 Target 也没有 Argument 时，同时填充
                        quad_str = f"{fallback_span} | {fallback_span} | {group_label_str} | {hateful_label_str}"
                        if quad_str not in seen_quads:
                            output_quads_list.append(quad_str)
                            seen_quads.add(quad_str)
                    else:  # 如果原始文本本身也为空（极少情况，但增加鲁棒性）
                        # 最终 fallback：如果连原始文本都为空，也用空白字符串填充
                        quad_str = f" | | {group_label_str} | {hateful_label_str}"
                        if quad_str not in seen_quads:
                            output_quads_list.append(quad_str)
                            seen_quads.add(quad_str)

                # 构建最终输出字符串
                output_str = " [SEP] ".join(output_quads_list) + " [END]"

                predictions.append({'id': content_id, 'output': output_str})

    return predictions


# --- Main Execution ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("Loading training data...")
    full_train_data = []
    with open('train.json', 'r', encoding='utf-8') as f:
        full_train_data = json.load(f)

    train_data, val_data = train_test_split(full_train_data, test_size=0.2, random_state=42)

    train_dataset = HateSpeechDataset(data_list=train_data, tokenizer=tokenizer, max_length=Config.MAX_LENGTH)
    val_dataset = HateSpeechDataset(data_list=val_data, tokenizer=tokenizer, max_length=Config.MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("Loading test data...")
    test_raw_data = []
    with open('test1.json', 'r', encoding='utf-8') as f:
        test_raw_data = json.load(f)

    test_dataset = HateSpeechDataset(data_list=test_raw_data, tokenizer=tokenizer, max_length=Config.MAX_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = HateSpeechDetector(len(Config.GROUP_LABELS), len(Config.SPAN_LABELS)).to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_dataloader) * Config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_STEPS),
        num_training_steps=total_steps
    )

    print("Starting training...")
    best_avg_f1 = -1.0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        train_model(model, train_dataloader, None, optimizer, scheduler)

        print("Evaluating on validation set...")
        hard_f1, soft_f1, avg_f1 = eval_model(model, val_dataloader, val_dataset)
        print(f"Validation Metrics - Hard F1: {hard_f1:.4f}, Soft F1: {soft_f1:.4f}, Average F1: {avg_f1:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved best model with Average F1: {best_avg_f1:.4f}")

    print("Training complete.")

    try:
        model.load_state_dict(torch.load("best_model.pt"))
        print("Loaded best model for final prediction.")
    except FileNotFoundError:
        print("best_model.pt not found, using the last trained model.")

    print("Generating predictions on test set...")
    predictions = predict(model, test_dataloader)

    output_file_txt = "submission.txt"
    with open(output_file_txt, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(f"{item['id']}\t{item['output']}\n")
    print(f"Predictions saved to {output_file_txt}")