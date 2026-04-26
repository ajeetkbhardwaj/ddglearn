"""Geometric Deep Learning modules for PyTorch.

Provides native PyTorch neural network layers that operate intrinsically
on 3D manifolds using Discrete Exterior Calculus (DEC).
"""

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # Mock to prevent syntax errors

from spectral.chebyshev import scaled_laplacian


if _HAS_TORCH:
    class ChebConv(nn.Module):
        """Chebyshev Spectral Graph Convolution Layer for meshes.
        
        Filters signals directly on the mesh manifold using the Chebyshev 
        polynomial approximation of the spectral graph convolution.
        """
        def __init__(self, in_channels: int, out_channels: int, order: int = 3):
            super().__init__()
            self.order = order
            self.weight = nn.Parameter(torch.Tensor(order, in_channels, out_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, mesh) -> torch.Tensor:
            """Apply convolution.
            
            Args:
                x: (n_vertices, in_channels) node feature tensor.
                mesh: TorchHalfEdgeMesh instance.
            """
            L_scaled = scaled_laplacian(mesh).to(x.dtype)
            
            Tx_0 = x
            out = torch.matmul(Tx_0, self.weight[0])
            
            if self.order > 1:
                Tx_1 = L_scaled @ x
                out += torch.matmul(Tx_1, self.weight[1])
                
                for k in range(2, self.order):
                    Tk = 2.0 * (L_scaled @ Tx_1) - Tx_0
                    out += torch.matmul(Tk, self.weight[k])
                    Tx_0, Tx_1 = Tx_1, Tk
                    
            return out + self.bias


    class MeshGATConv(nn.Module):
        """Graph Attention Layer for meshes.
        
        Applies attention-based message passing over the mesh edges.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
            self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
            self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
            nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, mesh) -> torch.Tensor:
            import torch.nn.functional as F
            
            h = torch.matmul(x, self.weight)
            
            # Compute attention coefficients
            attn_src = torch.sum(h * self.att_src, dim=-1)
            attn_dst = torch.sum(h * self.att_dst, dim=-1)
            
            device = x.device
            edge_list = torch.tensor(mesh.edge_list, dtype=torch.long, device=device)
            row, col = edge_list[:, 0], edge_list[:, 1]
            
            # Make undirected and add self-loops
            idx = torch.arange(mesh.n_vertices, device=device)
            row_all = torch.cat([row, col, idx])
            col_all = torch.cat([col, row, idx])
            
            e_vals = attn_src[row_all] + attn_dst[col_all]
            e_vals = F.leaky_relu(e_vals, negative_slope=0.2)
            
            adj = torch.sparse_coo_tensor(torch.stack([row_all, col_all]), e_vals, (mesh.n_vertices, mesh.n_vertices))
            
            if hasattr(torch.sparse, "softmax"):
                adj = torch.sparse.softmax(adj, dim=1)
            else:
                # Fallback for older PyTorch versions
                adj = adj.coalesce()
                adj = torch.sparse_coo_tensor(adj.indices(), torch.softmax(adj.values(), dim=0), adj.shape)
                
            out = torch.sparse.mm(adj, h)
            return out + self.bias


    class ShapeAutoencoder(nn.Module):
        """Autoencoder for extracting global shape embeddings from meshes."""
        def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
            super().__init__()
            self.conv1 = ChebConv(in_channels, hidden_channels, order=3)
            self.conv2 = ChebConv(hidden_channels, hidden_channels, order=3)
            self.enc_fc = nn.Linear(hidden_channels, latent_channels)
            self.dec_fc = nn.Linear(latent_channels, in_channels)
            
        def encode(self, x: torch.Tensor, mesh) -> torch.Tensor:
            import torch.nn.functional as F
            x = F.relu(self.conv1(x, mesh))
            x = F.relu(self.conv2(x, mesh))
            x = x.mean(dim=0, keepdim=True)  # Global average pool
            return self.enc_fc(x)
            
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F
            return self.dec_fc(F.relu(z))
            
        def forward(self, x: torch.Tensor, mesh) -> torch.Tensor:
            z = self.encode(x, mesh)
            return self.decode(z).expand(mesh.n_vertices, -1)


    class MeshEdgeConv(nn.Module):
        """MeshCNN-style edge convolution layer.
        
        Operates on edge features. Aggregates features from adjacent edges.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.linear_self = nn.Linear(in_channels, out_channels)
            self.linear_neighbors = nn.Linear(in_channels, out_channels, bias=False)
            
        def forward(self, x: torch.Tensor, mesh) -> torch.Tensor:
            device = x.device
            n_e = mesh.n_edges
            
            if not hasattr(mesh, "_edge_adjacency"):
                adj = [[] for _ in range(n_e)]
                for he in mesh.halfedges:
                    e_idx = mesh.edge_map[(min(he.origin, mesh.halfedges[he.next].origin), 
                                           max(he.origin, mesh.halfedges[he.next].origin))]
                    n1 = he.next
                    n2 = mesh.halfedges[n1].next
                    e1 = mesh.edge_map[(min(mesh.halfedges[n1].origin, mesh.halfedges[n2].origin),
                                        max(mesh.halfedges[n1].origin, mesh.halfedges[n2].origin))]
                    adj[e_idx].append(e1)
                
                adj_tensor = torch.zeros((n_e, 4), dtype=torch.long, device=device)
                for i, neighbors in enumerate(adj):
                    neighbors = list(set(neighbors))[:4]
                    for j, n in enumerate(neighbors):
                        adj_tensor[i, j] = n
                mesh._edge_adjacency = adj_tensor
                
            x_neighbors = x[mesh._edge_adjacency]
            x_pool = x_neighbors.sum(dim=1)
            
            return self.linear_self(x) + self.linear_neighbors(x_pool)

__all__ = ["ChebConv", "MeshGATConv", "ShapeAutoencoder", "MeshEdgeConv"] if _HAS_TORCH else []