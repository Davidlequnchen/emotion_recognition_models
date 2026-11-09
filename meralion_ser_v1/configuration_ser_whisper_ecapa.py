from transformers import PretrainedConfig

class SERConfig(PretrainedConfig):
    model_type = "ser-whisper-ecapa"
    def __init__(
        self, 
        whisper_name="openai/whisper-medium", 
        num_emotions=7, 
        pooling_type="hybrid", 
        sampling_rate=16000, 
        use_weighted_sum=False, 
        id2label=None, 
        label2id=None,
        projection_dim=256,
        num_dims=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.whisper_name = whisper_name
        self.num_emotions = int(num_emotions)
        self.pooling_type = pooling_type
        self.sampling_rate = int(sampling_rate)
        self.use_weighted_sum = bool(use_weighted_sum)
        self.id2label = id2label
        self.label2id = label2id
        self.projection_dim = int(projection_dim)
        self.num_dims = int(num_dims)