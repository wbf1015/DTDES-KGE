import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossv1(nn.Module):
    def __init__(self, args):
        """
        Contrastive Loss implementation combining two teacher embeddings.

        Args:
            args: An object containing necessary arguments like contrastive_tau,
                  input_dim (for teacher), target_dim (for student),
                  entity_mul, relation_mul.
        """
        super(ContrastiveLossv1, self).__init__()
        self.args = args
        self.tau  = self.args.contrastive_tau
        # Assuming teacher input dim is the same for both teachers
        self.teacher_embedding_dim = 256 * 2
        # Student entity embedding input dimension
        self.student_entity_embedding_dim = args.target_dim * args.entity_mul
        # Student relation embedding input dimension (not used in current forward, but good to have)
        self.student_relation_embedding_dim = args.target_dim * args.relation_mul

        # Output dimension of the MLPs for comparison
        self.hidden_dim = 128
        self.layermul = 2 # Factor for the hidden layer size

        # MLP for Teacher 1 embeddings
        self.Teacher1MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding_dim, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # MLP for Teacher 2 embeddings
        self.Teacher2MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding_dim, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # MLP for Student entity embeddings
        # Note: This MLP is applied to both head (eh) and tail (et) student embeddings
        self.StudentMLP = nn.Sequential(
            nn.Linear(self.student_entity_embedding_dim, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        # If student relation embeddings needed processing, add a separate MLP here


    """
    Calculates contrastive loss for inputs of shape [batch, 1, embedding_dim].
    Used for positive pairs (head or tail when batch size is the only dimension).
    Input Shape = [batch, 1, embedding_dim]
    """
    def contrastive_similarity(self, stu_embedding, tea_embedding):
        # Remove the singleton dimension
        stu_embedding = stu_embedding.squeeze(1)  # shape: [batch, embedding_dim]
        tea_embedding = tea_embedding.squeeze(1)  # shape: [batch, embedding_dim]

        # Normalize embeddings
        stu_embedding = torch.nn.functional.normalize(stu_embedding, p=2, dim=1)
        tea_embedding = torch.nn.functional.normalize(tea_embedding, p=2, dim=1)

        # Calculate cosine similarity matrix
        # Result shape: [batch, batch]
        # Sim(i, j) = dot product of student embedding i and teacher embedding j
        cosine_similarity_matrix = torch.matmul(stu_embedding, tea_embedding.T)

        # Scale by temperature
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau

        # Compute log-softmax for numerical stability
        # shape: [batch, batch]
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=1)

        # Create labels: the diagonal elements are the positive pairs
        # For batch i, the positive teacher embedding is the i-th one
        labels = torch.arange(cosine_similarity_matrix.size(0)).to(stu_embedding.device)  # shape: [batch]

        # Calculate NLL loss: -log(P(positive))
        loss = F.nll_loss(softmax_score, labels)

        return loss


    """
    Calculates contrastive loss for inputs of shape [batch, nneg+1, embedding_dim].
    Used for batches containing positive and negative samples (like tails in the original code).
    Input Shape = [batch, nneg+1, embedding_dim]
    """
    def contrastive_similarityv2(self, stu_embedding, tea_embedding):
        # Normalize embeddings along the embedding dimension
        stu_embedding = F.normalize(stu_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]
        tea_embedding = F.normalize(tea_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]

        # Calculate batch-wise matrix multiplication (cosine similarity)
        # Result shape: [batch, nneg+1, nneg+1]
        # Sim(b, i, j) = dot product of student embedding (b, i) and teacher embedding (b, j)
        cosine_similarity_matrix = torch.bmm(stu_embedding, tea_embedding.transpose(1, 2))

        # Scale by temperature
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau

        # Apply log-softmax along the last dimension (teacher samples)
        # This gives log-probabilities of a student sample (b, i) matching a teacher sample (b, j)
        # shape: [batch, nneg+1, nneg+1]
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=2)

        # Create labels: For each batch `b` and each student sample `i` within that batch,
        # the positive teacher sample is the one at index `i` in the same batch.
        # labels shape: [nneg+1] -> [1, nneg+1] -> [batch, nneg+1]
        labels = torch.arange(stu_embedding.size(1)).to(stu_embedding.device)
        labels = labels.unsqueeze(0).repeat(stu_embedding.size(0), 1)

        # Calculate NLL loss. Need to flatten the batch and nneg+1 dimensions for both scores and labels.
        # view(-1, softmax_score.size(-1)) -> [batch*(nneg+1), nneg+1]
        # view(-1) -> [batch*(nneg+1)]
        loss = F.nll_loss(softmax_score.view(-1, softmax_score.size(-1)), labels.view(-1), reduction='mean')
        return loss

    """
    Forward pass for calculating the combined contrastive loss.

    Args:
        eh (Tensor): Student head embeddings. Shape: [batch, 1, student_entity_embedding_dim]
        er (Tensor): Student relation embeddings. Shape: [batch, 1/nneg+1, student_relation_embedding_dim] (Not used in current implementation)
        et (Tensor): Student tail embeddings. Shape: [batch, nneg+1, student_entity_embedding_dim]
        Teacher_embeddings (Tuple): Contains embeddings from two teacher models.
            PT_head1 (Tensor): Teacher 1 head embeddings. Shape: [batch, 1, teacher_embedding_dim]
            PT_relation1 (Tensor): Teacher 1 relation embeddings. Shape: [batch, 1/nneg+1, teacher_embedding_dim] (Not used)
            PT_tail1 (Tensor): Teacher 1 tail embeddings. Shape: [batch, nneg+1, teacher_embedding_dim]
            PT_head2 (Tensor): Teacher 2 head embeddings. Shape: [batch, 1, teacher_embedding_dim]
            PT_relation2 (Tensor): Teacher 2 relation embeddings. Shape: [batch, 1/nneg+1, teacher_embedding_dim] (Not used)
            PT_tail2 (Tensor): Teacher 2 tail embeddings. Shape: [batch, nneg+1, teacher_embedding_dim]

    Returns:
        Tuple: (total_loss, loss_record)
            total_loss (Tensor): The scalar total contrastive loss.
            loss_record (dict): Dictionary containing individual loss components for logging.
    """
    def forward(self, eh, er, et, Teacher_embeddings):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings

        # --- Process Embeddings through MLPs ---
        # Student embeddings
        # eh shape: [batch, 1, student_entity_embedding_dim] -> stu_head shape: [batch, 1, hidden_dim]
        stu_head = self.StudentMLP(eh)
        # et shape: [batch, nneg+1, student_entity_embedding_dim] -> stu_tail shape: [batch, nneg+1, hidden_dim]
        stu_tail = self.StudentMLP(et)
        # Note: Student relation embedding 'er' is not used in the original forward logic, keeping this behavior.

        # Teacher 1 embeddings
        # PT_head1 shape: [batch, 1, teacher_embedding_dim] -> tea_head1 shape: [batch, 1, hidden_dim]
        tea_head1 = self.Teacher1MLP(PT_head1)
        # PT_tail1 shape: [batch, nneg+1, teacher_embedding_dim] -> tea_tail1 shape: [batch, nneg+1, hidden_dim]
        # tea_tail1 = self.Teacher1MLP(PT_tail1)
        # Note: Teacher 1 relation embedding 'PT_relation1' is not used, keeping this behavior.

        # Teacher 2 embeddings
        # PT_head2 shape: [batch, 1, teacher_embedding_dim] -> tea_head2 shape: [batch, 1, hidden_dim]
        tea_head2 = self.Teacher2MLP(PT_head2)
        # PT_tail2 shape: [batch, nneg+1, teacher_embedding_dim] -> tea_tail2 shape: [batch, nneg+1, hidden_dim]
        # tea_tail2 = self.Teacher2MLP(PT_tail2)
        # Note: Teacher 2 relation embedding 'PT_relation2' is not used, keeping this behavior.


        # --- Combine Teacher Embeddings ---
        # Combine head embeddings from Teacher 1 and Teacher 2 by summation
        # Shape: [batch, 1, hidden_dim] + [batch, 1, hidden_dim] = [batch, 1, hidden_dim]
        combined_tea_head = tea_head1 + tea_head2

        # Combine tail embeddings from Teacher 1 and Teacher 2 by summation
        # Shape: [batch, nneg+1, hidden_dim] + [batch, nneg+1, hidden_dim] = [batch, nneg+1, hidden_dim]
        # combined_tea_tail = tea_tail1 + tea_tail2

        # --- Calculate Contrastive Losses ---
        # Head loss: Compare student head to the combined teacher head embedding
        # Uses contrastive_similarity because input shape is [batch, 1, hidden_dim]
        head_loss = self.contrastive_similarity(stu_head, combined_tea_head)

        # Tail loss: Compare student tail to the combined teacher tail embedding
        # Uses contrastive_similarityv2 because input shape is [batch, nneg+1, hidden_dim]
        # tail_loss = self.contrastive_similarityv2(stu_tail, combined_tea_tail)

        # --- Total Loss ---
        # Sum the head and tail losses (can add weights here if needed)
        # loss = head_loss + tail_loss
        loss = head_loss

        # --- Record Losses ---
        loss_record = {}
        # Update record with the new combined losses
        loss_record.update({'combined_head_contrastiveLoss' : head_loss.item()})
        # loss_record.update({'combined_tail_contrastiveLoss' : tail_loss.item()})

        return loss, loss_record






class ContrastiveLossv2(nn.Module):
    def __init__(self, args):
        """
        Contrastive Loss implementation combining two teacher embeddings.

        Args:
            args: An object containing necessary arguments like contrastive_tau,
                  input_dim (for teacher), target_dim (for student),
                  entity_mul, relation_mul.
        """
        super(ContrastiveLossv2, self).__init__()
        self.args = args
        self.tau  = self.args.contrastive_tau
        # Assuming teacher input dim is the same for both teachers
        self.teacher_embedding_dim = 256 * 2
        # Student entity embedding input dimension
        self.student_entity_embedding_dim = args.target_dim * args.entity_mul
        # Student relation embedding input dimension (not used in current forward, but good to have)
        self.student_relation_embedding_dim = args.target_dim * args.relation_mul

        # Output dimension of the MLPs for comparison
        self.hidden_dim = 128
        self.layermul = 2 # Factor for the hidden layer size

        # MLP for Teacher 1 embeddings
        self.Teacher1MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding_dim, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.student_entity_embedding_dim),
            nn.LayerNorm(self.student_entity_embedding_dim)
        )

        # MLP for Teacher 2 embeddings
        self.Teacher2MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding_dim, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.student_entity_embedding_dim),
            nn.LayerNorm(self.student_entity_embedding_dim)
        )


    """
    Calculates contrastive loss for inputs of shape [batch, 1, embedding_dim].
    Used for positive pairs (head or tail when batch size is the only dimension).
    Input Shape = [batch, 1, embedding_dim]
    """
    def contrastive_similarity(self, stu_embedding, tea_embedding):
        # Remove the singleton dimension
        stu_embedding = stu_embedding.squeeze(1)  # shape: [batch, embedding_dim]
        tea_embedding = tea_embedding.squeeze(1)  # shape: [batch, embedding_dim]

        # Normalize embeddings
        stu_embedding = torch.nn.functional.normalize(stu_embedding, p=2, dim=1)
        tea_embedding = torch.nn.functional.normalize(tea_embedding, p=2, dim=1)

        # Calculate cosine similarity matrix
        # Result shape: [batch, batch]
        # Sim(i, j) = dot product of student embedding i and teacher embedding j
        cosine_similarity_matrix = torch.matmul(stu_embedding, tea_embedding.T)

        # Scale by temperature
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau

        # Compute log-softmax for numerical stability
        # shape: [batch, batch]
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=1)

        # Create labels: the diagonal elements are the positive pairs
        # For batch i, the positive teacher embedding is the i-th one
        labels = torch.arange(cosine_similarity_matrix.size(0)).to(stu_embedding.device)  # shape: [batch]

        # Calculate NLL loss: -log(P(positive))
        loss = F.nll_loss(softmax_score, labels)

        return loss


    """
    Calculates contrastive loss for inputs of shape [batch, nneg+1, embedding_dim].
    Used for batches containing positive and negative samples (like tails in the original code).
    Input Shape = [batch, nneg+1, embedding_dim]
    """
    def contrastive_similarityv2(self, stu_embedding, tea_embedding):
        # Normalize embeddings along the embedding dimension
        stu_embedding = F.normalize(stu_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]
        tea_embedding = F.normalize(tea_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]

        # Calculate batch-wise matrix multiplication (cosine similarity)
        # Result shape: [batch, nneg+1, nneg+1]
        # Sim(b, i, j) = dot product of student embedding (b, i) and teacher embedding (b, j)
        cosine_similarity_matrix = torch.bmm(stu_embedding, tea_embedding.transpose(1, 2))

        # Scale by temperature
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau

        # Apply log-softmax along the last dimension (teacher samples)
        # This gives log-probabilities of a student sample (b, i) matching a teacher sample (b, j)
        # shape: [batch, nneg+1, nneg+1]
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=2)

        # Create labels: For each batch `b` and each student sample `i` within that batch,
        # the positive teacher sample is the one at index `i` in the same batch.
        # labels shape: [nneg+1] -> [1, nneg+1] -> [batch, nneg+1]
        labels = torch.arange(stu_embedding.size(1)).to(stu_embedding.device)
        labels = labels.unsqueeze(0).repeat(stu_embedding.size(0), 1)

        # Calculate NLL loss. Need to flatten the batch and nneg+1 dimensions for both scores and labels.
        # view(-1, softmax_score.size(-1)) -> [batch*(nneg+1), nneg+1]
        # view(-1) -> [batch*(nneg+1)]
        loss = F.nll_loss(softmax_score.view(-1, softmax_score.size(-1)), labels.view(-1), reduction='mean')
        return loss

    """
    Forward pass for calculating the combined contrastive loss.

    Args:
        eh (Tensor): Student head embeddings. Shape: [batch, 1, student_entity_embedding_dim]
        er (Tensor): Student relation embeddings. Shape: [batch, 1/nneg+1, student_relation_embedding_dim] (Not used in current implementation)
        et (Tensor): Student tail embeddings. Shape: [batch, nneg+1, student_entity_embedding_dim]
        Teacher_embeddings (Tuple): Contains embeddings from two teacher models.
            PT_head1 (Tensor): Teacher 1 head embeddings. Shape: [batch, 1, teacher_embedding_dim]
            PT_relation1 (Tensor): Teacher 1 relation embeddings. Shape: [batch, 1/nneg+1, teacher_embedding_dim] (Not used)
            PT_tail1 (Tensor): Teacher 1 tail embeddings. Shape: [batch, nneg+1, teacher_embedding_dim]
            PT_head2 (Tensor): Teacher 2 head embeddings. Shape: [batch, 1, teacher_embedding_dim]
            PT_relation2 (Tensor): Teacher 2 relation embeddings. Shape: [batch, 1/nneg+1, teacher_embedding_dim] (Not used)
            PT_tail2 (Tensor): Teacher 2 tail embeddings. Shape: [batch, nneg+1, teacher_embedding_dim]

    Returns:
        Tuple: (total_loss, loss_record)
            total_loss (Tensor): The scalar total contrastive loss.
            loss_record (dict): Dictionary containing individual loss components for logging.
    """
    def forward(self, eh, er, et, Teacher_embeddings, weight=None):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings

        stu_head = eh
        
        tea_head1 = self.Teacher1MLP(PT_head1)
        tea_head2 = self.Teacher2MLP(PT_head2)
        
        if weight is not None:
            weight = weight.squeeze(1)
            w1 = weight[:, 0:1].unsqueeze(-1)        # [B, 1, 1]
            w2 = weight[:, 1:2].unsqueeze(-1)        # [B, 1, 1]
            combined_tea_head = w1 * tea_head1 + w2 * tea_head2
        else:
            combined_tea_head = tea_head1 + tea_head2

        head_loss = self.contrastive_similarity(stu_head, combined_tea_head)
        loss = head_loss
        loss_record = {}
        loss_record.update({'combined_head_contrastiveLoss' : head_loss.item()})

        return loss, loss_record