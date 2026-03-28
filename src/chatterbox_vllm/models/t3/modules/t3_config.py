class T3Config:
    def __init__(self):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = 50276
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 6563
        self.max_speech_tokens = 4096

        self.input_pos_emb = None
        self.speech_cond_prompt_len = 375

        # For T3CondEnc
        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = False
        self.emotion_adv = False
        self.n_channels = 1024
