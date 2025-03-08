"""
Multi-Agent Architecture Search (MaAS) Controller Module

This module implements the core controller components for the MaAS framework, which dynamically
generates customized multi-agent architectures for each query. The controller uses a probabilistic
approach to select appropriate operators (agents) based on the query embedding.

Key components:
- OperatorSelector: Selects operators based on query and previous operator embeddings
- MultiLayerController: Manages multiple layers of operator selection to build a complete
  multi-agent workflow

The controller implements a Mixture-of-Experts (MoE) style mechanism to efficiently select
architectures based on the query, optimizing for both performance and computational cost.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.special import gammaln
from maas.ext.maas.models.utils import SentenceEncoder, sample_operators

class OperatorSelector(torch.nn.Module):
    """
    Operator selection module that computes compatibility scores between a query and available operators.
    
    This module projects query and operator embeddings into a shared space and computes similarity scores
    to determine which operators are most suitable for the given query. For non-first layers, it also
    considers the previously selected operators to maintain contextual coherence in the workflow.
    
    Args:
        input_dim (int): Dimension of input embeddings (default: 384)
        hidden_dim (int): Dimension of the projection space (default: 32)
        device: Computation device (default: None, will use CUDA if available)
        is_first_layer (bool): Whether this selector is the first layer in the controller (default: False)
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, device=None, is_first_layer: bool = False):
        super().__init__()
        self.is_first_layer = is_first_layer
        if self.is_first_layer:
            self.operator_encoder = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.operator_encoder = torch.nn.Linear(input_dim * 2, hidden_dim)
        self.query_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, query_embed: torch.Tensor, operators_embed: torch.Tensor, prev_operators_embed: torch.Tensor = None):
        """
        Forward pass to compute compatibility scores between query and operators.
        
        Args:
            query_embed: Embedding of the query
            operators_embed: Embeddings of all available operators
            prev_operators_embed: Embeddings of previously selected operators (None for first layer)
            
        Returns:
            tuple: (log_probs, probs) containing log probabilities and probabilities for each operator
        """
        # Ensure query is 2D for batch processing
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)

        # Project query into the shared space and normalize
        query_embed = self.query_encoder(query_embed)   
        query_embed = F.normalize(query_embed, p=2, dim=1)  # L2 normalization for cosine similarity

        # For non-first layers, incorporate previous operator context
        if prev_operators_embed is not None and self.is_first_layer is False:
            prev_operator = prev_operators_embed[0].unsqueeze(0)
            # Expand previous operator to match operators dimension for concatenation
            prev_expanded = prev_operator.expand(operators_embed.size(0), -1)
            # Concatenate current operators with previous operator for context-aware encoding
            concat_embed = torch.cat([operators_embed, prev_expanded], dim=1)
            all_operators_embed = self.operator_encoder(concat_embed)
        else:
            # For first layer, just encode the operators without previous context
            all_operators_embed = self.operator_encoder(operators_embed)

        # Normalize operator embeddings for cosine similarity computation
        all_operators_embed = F.normalize(all_operators_embed, p=2, dim=1)       

        # Compute compatibility scores via dot product (equivalent to cosine similarity after normalization)
        scores = torch.matmul(query_embed, all_operators_embed.T)

        # Convert scores to probabilities and log probabilities
        probs = F.softmax(scores, dim=1)
        log_probs = F.log_softmax(scores, dim=1)

        return log_probs, probs

class MultiLayerController(torch.nn.Module):
    """
    Multi-layer controller that manages the sequential selection of operators across multiple layers.
    
    This controller builds a complete multi-agent workflow by sequentially selecting operators
    at each layer based on the query and previously selected operators. It enforces certain
    constraints (e.g., ensuring Generate operators are selected appropriately) and handles
    early stopping conditions.
    
    Args:
        input_dim (int): Dimension of input embeddings (default: 384)
        hidden_dim (int): Dimension of the projection space (default: 32)
        num_layers (int): Number of selection layers (default: 4)
        device: Computation device (default: None, will use CUDA if available)
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, num_layers: int = 4, device=None):
        super().__init__()
        self.text_encoder = SentenceEncoder()
        self.layers = torch.nn.ModuleList([
            OperatorSelector(input_dim, hidden_dim, device, is_first_layer=(i == 0)) 
            for i in range(num_layers)
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def forward(self, query, operators_embedding, selection_operator_names):
        """
        Forward pass to build a multi-agent workflow for the given query.
        
        Args:
            query: The input query text
            operators_embedding: Embeddings of all available operators
            selection_operator_names: Names of the operators corresponding to the embeddings
            
        Returns:
            tuple: (log_probs_layers, selected_names_layers) containing log probabilities and 
                  selected operator names for each layer
        """
        # Encode the query text into an embedding
        query_embedding = self.text_encoder(query).to(self.device)
        operators_embedding = operators_embedding.to(self.device) 
        log_probs_layers = []  # Store log probabilities for each layer
        selected_names_layers = []  # Store selected operator names for each layer
        prev_operators = None  # Track previously selected operators for context

        # Process each layer sequentially
        for layer_idx, layer in enumerate(self.layers):
            # For first layer, don't provide previous operators
            if layer_idx == 0:
                log_probs, probs = layer(query_embedding, operators_embedding)
            else:
                log_probs, probs = layer(query_embedding, operators_embedding, prev_operators)
            
            # Convert batch dimension to 1D for easier processing
            probs_1d = probs.squeeze(0)
            log_probs_1d = log_probs.squeeze(0)
            
            # Sample operators based on probabilities with a threshold
            selected_indices = sample_operators(probs_1d, threshold=0.25)
            selected_indices_list = selected_indices.cpu().tolist()
            selected_names = [selection_operator_names[idx] for idx in selected_indices_list]
            penalty_applied = False

            # Special handling for the first layer to ensure proper operator selection
            if layer_idx == 0:
                # Rule 1: If EarlyStop is selected in the first layer, replace with Generate and apply penalty
                if any(name.lower() == "earlystop" for name in selected_names):
                    penalty_applied = True
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                
                # Rule 2: If no Generate operator is selected, force select Generate
                elif not any("generate" in name.lower() for name in selected_names):
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                
                # Rule 3: If Generate is not the first operator but exists in selection, reorder to put it first
                elif "generate" not in selected_names[0].lower() and any("generate" in name.lower() for name in selected_names):
                    for idx, name in enumerate(selected_names):
                        if "generate" in name.lower():
                            # Reorder to put Generate first
                            selected_names = [selected_names[idx]] + selected_names[:idx] + selected_names[idx+1:]
                            try:
                                new_first_idx = selection_operator_names.index(selected_names[0])
                            except ValueError:
                                new_first_idx = 0
                            new_indices = [new_first_idx] + [selection_operator_names.index(n) for n in selected_names[1:]]
                            selected_indices = torch.tensor(new_indices, device=self.device)
                            break

            # Calculate the log probability for the selected operators
            if selected_indices.numel() > 0:
                layer_log_prob = torch.sum(log_probs_1d[selected_indices])
            else:
                layer_log_prob = torch.tensor(0.0, device=self.device)

            # Apply penalty for inappropriate EarlyStop selection in first layer
            if layer_idx == 0 and penalty_applied:
                # Apply a fixed penalty of -1.5 to discourage this pattern in training
                layer_log_prob = layer_log_prob + torch.tensor(-1.5, device=self.device)

            # Store results for this layer
            log_probs_layers.append(layer_log_prob)
            selected_names_layers.append(selected_names)

            # Update previous operators for next layer's context
            if selected_indices.numel() > 0:
                selected_indices = selected_indices.to(operators_embedding.device)
                prev_operators = operators_embedding[selected_indices]
            else:
                prev_operators = None
            
            # Early stopping conditions:
            # 1. If EarlyStop was inappropriately selected in first layer
            # 2. If EarlyStop is selected in any layer
            if (layer_idx == 0 and penalty_applied) or any(name.lower() == "earlystop" for name in selected_names):
                break

        return log_probs_layers, selected_names_layers

