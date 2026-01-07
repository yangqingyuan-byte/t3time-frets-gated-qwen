import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class GenPromptEmbQwen3(nn.Module):
    """
    使用 Qwen3 生成 Prompt Embeddings，与原版 GenPromptEmb 接口保持一致：
    - 相同的 prompt 模板与时间序列拼接方式；
    - 仅替换底层 LLM 为 Qwen/Qwen3-4B-Base（默认 d_model=2560）。
    """

    def __init__(
        self,
        data_path: str = "FRED",
        model_name: str = "Qwen/Qwen3-4B-Base",
        device: str = "cuda:0",
        input_len: int = 96,
        d_model: int = 2560,
        layer: int = 24,
        divide: str = "train",
    ):
        super().__init__()
        self.data_path = data_path
        # 统一使用单一 device，避免多卡自动切分导致的 device mismatch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_len = input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len - 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        # Last token trend
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"

        # Date
        if self.data_path in ["FRED", "ILI"]:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ["ETTh1", "ETTh2", "ECL"]:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else:  # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)

        tokenized_prompt = self.tokenizer(in_prompt, return_tensors="pt").to(self.device)
        # Qwen tokenizer 返回 dict，需要取 input_ids
        input_ids = tokenized_prompt["input_ids"]
        return input_ids

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            outputs = self.model(tokenized_prompt)
            # 兼容 Qwen 的返回格式，选取 last_hidden_state
            prompt_embeddings = outputs.last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
        input_templates = {
            "FRED": "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
            "ILI": "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            "ETTh1": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            "ETTh2": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            "ECL": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            "ETTm1": "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            "ETTm2": "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            "Weather": "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
        }

        input_template = input_templates.get(self.data_path, input_templates["FRED"])

        tokenized_prompts = []
        max_token_count = 0
        B, _, N = in_data.shape
        for i in range(B):
            for j in range(N):
                tokenized_prompt = self._prepare_prompt(
                    input_template, in_data, in_data_mark, i, j
                )
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

        in_prompt_emb = torch.zeros(
            (B, max_token_count, self.d_model, N),
            dtype=torch.float32,
            device=self.device,
        )

        for i, tokenized_prompt, j in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
            else:
                prompt_embeddings_padded = prompt_embeddings

            in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded

        last_token_emb = in_prompt_emb[:, max_token_count - 1 : max_token_count, :, :]
        last_token_emb = last_token_emb.squeeze()
        return last_token_emb


