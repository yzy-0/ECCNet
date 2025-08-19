import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    def __init__(self, temperature=0.5, use_projection=True, sample_ratio=0.1, l2_weight=0.1):
        super(FeatureDistillationLoss, self).__init__()
        self.temperature = temperature
        self.use_projection = use_projection
        self.sample_ratio = sample_ratio
        self.l2_weight = l2_weight
        self.projection = None
        self.attention = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def _create_projection_if_needed(self, in_dim, out_dim=128):
        if self.use_projection and self.projection is None:
            self.projection = nn.Conv2d(in_dim, out_dim, 1, bias=True)
            nn.init.orthogonal_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.constant_(self.projection.bias, 0)
            return self.projection
        return self.projection

    def _align_features(self, student_feat, teacher_feat):
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])

        if student_feat.shape[1] != teacher_feat.shape[1]:
            projection = self._create_projection_if_needed(student_feat.shape[1], teacher_feat.shape[1])
            student_feat = projection(student_feat)

        return student_feat, teacher_feat

    def _compute_boundary_mask(self, teacher_logits, student_logits=None):
        if teacher_logits is None:
            return None
        probs = F.softmax(teacher_logits, dim=1)  # [B, num_classes, H, W]
        max_probs, _ = probs.max(dim=1, keepdim=True)  # [B, 1, H, W]
        boundary = self.attention(max_probs)  # [B, 1, H, W]
        boundary = boundary / (boundary.max() + 1e-6) 

        return boundary

    def _region_weighted_contrastive_loss(self, s_feat, t_feat, boundary_mask=None, teacher_logits=None):
        batch_size, num_channels, height, width = s_feat.shape
        num_pixels = height * width
        num_samples = max(64, int(num_pixels * self.sample_ratio))
        s_feat = F.normalize(s_feat, p=2, dim=1)
        t_feat = F.normalize(t_feat, p=2, dim=1)
        
        if boundary_mask is not None:
            sample_weights = boundary_mask.view(-1) + 1e-6
            sample_weights = sample_weights / sample_weights.sum()
            indices = torch.multinomial(sample_weights, num_samples, replacement=True)
        else:
            indices = torch.randperm(num_pixels, device=s_feat.device)[:num_samples]

        s_pixels = s_feat.permute(0, 2, 3, 1).reshape(-1, num_channels)[indices]
        t_pixels = t_feat.permute(0, 2, 3, 1).reshape(-1, num_channels)[indices]


        batch_indices = torch.arange(batch_size, device=s_feat.device)
        batch_indices = batch_indices.view(-1, 1, 1).expand(-1, height, width)
        batch_indices = batch_indices.reshape(-1)[indices]

        logits = torch.matmul(s_pixels, t_pixels.t()) / self.temperature
        positives = torch.zeros_like(logits, dtype=torch.bool)
        sample_indices = torch.arange(num_samples, device=logits.device)
        positives[sample_indices, sample_indices] = True

        if teacher_logits is not None:
            labels = teacher_logits.argmax(dim=1).view(-1)[indices] 
            class_mask = labels.unsqueeze(1) != labels.unsqueeze(0)  
            batch_class_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)) & class_mask
            neg_mask = batch_class_mask | (batch_indices.unsqueeze(1) != batch_indices.unsqueeze(0))
        else:
            neg_mask = batch_indices.unsqueeze(1) != batch_indices.unsqueeze(0)

        if boundary_mask is not None:
            boundary_weights = boundary_mask.view(-1)[indices].float()
            weights = 1.0 + boundary_weights
        else:
            weights = torch.ones(num_samples, device=s_feat.device)

        pos_logits = logits[positives].view(-1, 1)
        neg_logits = logits[neg_mask].view(num_samples, -1)

        exp_neg_logits = torch.exp(neg_logits)
        log_prob = pos_logits - torch.log(torch.exp(pos_logits) + exp_neg_logits.sum(dim=1, keepdim=True))

        weighted_log_prob = log_prob.view(-1) * weights
        contrastive_loss = -weighted_log_prob.mean()
        l2_loss = F.mse_loss(s_feat, t_feat, reduction='mean')

        return contrastive_loss + self.l2_weight * l2_loss

    def forward(self, student_feat, teacher_feat, teacher_logits=None, student_logits=None):
        student_feat, teacher_feat = self._align_features(student_feat, teacher_feat)
        boundary_mask = self._compute_boundary_mask(teacher_logits, student_logits)
        loss = self._region_weighted_contrastive_loss(student_feat, teacher_feat, boundary_mask, teacher_logits)
        return loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    input_h, input_w = 480, 640
    cs1_h, cs1_w = 120, 160
    student_cs1 = torch.randn(batch_size, 64, cs1_h, cs1_w).to(device)
    teacher_cs1 = torch.randn(batch_size, 64, cs1_h, cs1_w).to(device)
    criterion = FeatureDistillationLoss().to(device)
    loss = criterion(student_cs1, teacher_cs1)
    loss = loss
    print(f"Total Loss: {loss.item():.6f}")
    # print(f"Current Weight: {weight:.6f}")


if __name__ == "__main__":
    main()
