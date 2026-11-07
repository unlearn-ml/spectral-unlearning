from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import evaluate_probability

class LOSSAttack(Attack):

    def compute_batch_values(self, batch):
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        return sample_stats['avg_loss']
