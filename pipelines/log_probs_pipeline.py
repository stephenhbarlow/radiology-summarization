from transformers import TextClassificationPipeline
import torch.nn.functional as F


class LogProbsPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        log_probs = F.log_softmax(best_class, dim=-1)      
        return log_probs
    