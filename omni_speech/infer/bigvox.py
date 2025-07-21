import torch
from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import preprocess_hcx
import whisper
import argparse
import re

BIGVOX_MODEL = "./checkpoints"

class BigVoxModel:
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.empty = True

        self.temperature = kwargs.get('temperature', 0.0)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)

    def __initilize__(self):
        if self.empty:
            self.empty = False
            self.tokenizer, self.model, _ = load_pretrained_model(self.model_name_or_path, s2s=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, messages: list) -> dict:
        audio_path = messages[0]['path']
        speech = whisper.load_audio(audio_path)
        
        if self.model.config.speech_encoder_type == "glm4voice":
            speech_length = torch.LongTensor([speech.shape[0]])
            speech = torch.from_numpy(speech)
            speech = torch.nn.functional.layer_norm(speech, speech.shape)
        else:
            raw_len = len(speech)
            speech = whisper.pad_or_trim(speech)
            padding_len = len(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
            speech_length = round(raw_len / padding_len * 3000 + 0.5)
            speech_length = torch.LongTensor([speech_length])
        
        conversation = [{"from": "human", "value": "<speech>"}]
        processed_inputs = preprocess_hcx([conversation], self.tokenizer, has_speech=True)
        input_ids = processed_inputs['input_ids']
        
        assistant_start = "<|im_start|>assistant\n"
        assistant_tokens = self.tokenizer.encode(assistant_start, add_special_tokens=False)
        
        input_ids = torch.cat([
            input_ids.squeeze(0),
            torch.tensor(assistant_tokens, device=input_ids.device)
        ]).unsqueeze(0)
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = attention_mask.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)
         
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p if self.top_p is not None else 0.0,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.15,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            full_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            if "<|im_start|>assistant" in full_output:
                assistant_start_idx = full_output.find("<|im_start|>assistant")
                if assistant_start_idx != -1:
                    after_assistant = full_output[assistant_start_idx:]
                    if "\n" in after_assistant:
                        output_text = after_assistant.split("\n", 1)[1]
                    else:
                        output_text = after_assistant.replace("<|im_start|>assistant", "").strip()
                else:
                    output_text = full_output
            else:
                output_text = full_output
            
            if '<|im_end|>' in output_text:
                output_text = output_text.split('<|im_end|>')[0].strip()
            
            if 'assistant' in output_text:
                assistant_pattern = r'\n?assistant\n?'
                output_text = re.split(assistant_pattern, output_text)[0].strip()
            
        result = {"text": output_text}
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BigVox S2T Inference')
    parser.add_argument('--query_audio', type=str, required=True, help='Path to the input audio file')
    args = parser.parse_args()

    audio_messages = [{"role": "user", "content": "<speech>", "path": args.query_audio}]
    print("Initialized BigVox for S2T")
    
    bigvox = BigVoxModel(BIGVOX_MODEL)
    bigvox.__initilize__()
    
    print(f"Running inference for {args.query_audio}...")
    response = bigvox(audio_messages)
    
    print("Inference result:")
    print(response)
