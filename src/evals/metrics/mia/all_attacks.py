from enum import Enum
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class AllAttacks(str, Enum):
    LOSS = 'loss'
    REFERENCE_BASED = 'ref'
    ZLIB = 'zlib'
    MIN_K = 'min_k'
    MIN_K_PLUS_PLUS = 'min_k++'
    GRADNORM = 'gradnorm'
    RECALL = 'recall'

class Attack:

    def __init__(self, model, data, collator, batch_size, **kwargs):
        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        pass

    def compute_batch_values(self, batch):
        raise NotImplementedError

    def compute_score(self, sample_stats):
        raise NotImplementedError

    def attack(self):
        all_scores = []
        all_indices = []
        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            indices = batch.pop('index').cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]
            all_scores.extend(scores)
            all_indices.extend(indices)
        scores_by_index = {str(idx): {'score': float(score)} for idx, score in zip(all_indices, all_scores)}
        return {'agg_value': float(np.mean(all_scores)), 'value_by_index': scores_by_index}
