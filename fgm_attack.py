import torch


class FGM:
    """Fast Gradient Method (FGM) for adversarial training on embedding layer"""
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        # Save original embeddings
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # Restore original embeddings
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embeddings' in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}