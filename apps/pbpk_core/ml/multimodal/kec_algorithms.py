"""
KEC Algorithms - Imported from Biomaterials Plugin
===================================================

Spectral entropy, Forman curvature, Small-world metrics

Author: Agourakis (from Mestrado)
Imported: 2025-10-31T02:15:00Z
Source: darwin-plugin-biomaterials/app/services/kec_calculator.py
Darwin Indexed: Auto
"""

import numpy as np
import networkx as nx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Import scipy if available
try:
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY = True
except ImportError:
    eigsh = None
    HAS_SCIPY = False


class EntropyCalculator:
    """Entropy metrics for molecular graphs"""
    
    @staticmethod
    def _normalized_laplacian(G: nx.Graph):
        """Return normalized Laplacian"""
        try:
            return nx.normalized_laplacian_matrix(G)
        except Exception:
            # Fallback manual
            A = nx.to_numpy_array(G)
            d = np.sum(A, axis=1)
            D_inv_sqrt = np.diag(1.0/np.sqrt(np.maximum(d, 1e-12)))
            L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
            return L
    
    @staticmethod
    def spectral_entropy(G: nx.Graph, k: int = 64, normalized: bool = False, tol: float = 1e-8):
        """
        Spectral entropy (von Neumann) based on normalized Laplacian
        
        H = -Σ p_i log(p_i), where p_i = λ_i / Σλ_i
        
        Args:
            G: NetworkX graph
            k: Number of eigenvalues (for large graphs)
            normalized: Return normalized entropy
            tol: Tolerance for eigenvalue calculation
            
        Returns:
            H_spectral: Spectral entropy (bits)
            eigen_stats: dict with lambda_max, spectral_gap
        """
        n = G.number_of_nodes()
        if n == 0:
            return 0.0, {}
        
        try:
            L = EntropyCalculator._normalized_laplacian(G)
            
            # Calculate eigenvalues
            if HAS_SCIPY and eigsh is not None and hasattr(L, "shape") and min(L.shape) > 128:
                k_use = min(max(2, k), n-1)
                vals = np.abs(eigsh(L, k=k_use, which="LM", return_eigenvectors=False, tol=tol))
                # Approximate with top-k
                s = np.sum(vals)
                p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
            else:
                # Small graph: full eigenvalues
                if hasattr(L, "toarray"):
                    L = L.toarray()
                vals = np.linalg.eigvalsh(L)
                vals = np.abs(vals)
                s = np.sum(vals)
                p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
            
            eigen_stats = {
                'lambda_max': float(np.max(vals)),
                'lambda_min': float(np.min(vals)),
                'spectral_gap': float(np.max(vals) - np.min(vals))
            }
                
        except Exception as e:
            logger.debug(f"Spectral calculation error, using fallback: {e}")
            # Fallback: degree distribution as proxy
            deg = np.array([d for _, d in G.degree()], dtype=float)
            p = deg / np.sum(deg) if deg.sum() > 0 else np.ones(len(deg))/len(deg)
            eigen_stats = {'lambda_max': 2.0, 'lambda_min': 0.0, 'spectral_gap': 2.0}
        
        p = p[p > 1e-15]
        H = float(-np.sum(p * np.log(p)))
        
        if normalized and n > 1:
            H = H / np.log(n)
        
        return H, eigen_stats
    
    @staticmethod
    def random_walk_entropy(G: nx.Graph) -> float:
        """
        Random walk entropy
        
        H_rw = -Σ p_i log(p_i), where p_i = degree_i / Σdegrees
        """
        if G.number_of_nodes() == 0:
            return 0.0
        
        deg = np.array([d for _, d in G.degree()], dtype=float)
        p = deg / np.sum(deg) if deg.sum() > 0 else np.ones(len(deg))/len(deg)
        p = p[p > 1e-15]
        
        return float(-np.sum(p * np.log(p)))


class CurvatureCalculator:
    """Curvature metrics for molecular graphs"""
    
    @staticmethod
    def forman_curvature(G: nx.Graph, return_distribution: bool = True) -> Dict[str, float]:
        """
        Forman curvature per edge
        
        F(u,v) = deg(u) + deg(v) - 2 - t(u,v)
        
        where t(u,v) = number of triangles containing edge (u,v)
        
        Returns:
            dict with mean, std, min, max, negative_pct
        """
        if G.number_of_edges() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, 
                    "p05": 0.0, "p50": 0.0, "p95": 0.0, "negative_pct": 0.0}
        
        deg = dict(G.degree())
        
        def edge_triangles(u, v):
            """Count triangles containing edge (u,v)"""
            Nu = set(G.neighbors(u)) - {v}
            Nv = set(G.neighbors(v)) - {u}
            return len(Nu & Nv)
        
        curvatures = []
        for u, v in G.edges():
            t = edge_triangles(u, v)
            f = (deg[u] + deg[v] - 2) - t
            curvatures.append(f)
        
        a = np.array(curvatures, dtype=float)
        
        stats = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "negative_pct": float(100 * np.sum(a < 0) / len(a))
        }
        
        if return_distribution:
            stats['distribution'] = a
        
        return stats
    
    @staticmethod
    def identify_bottlenecks(G: nx.Graph, curvature_threshold: float = -2.0) -> List:
        """
        Identify bottleneck edges (negative curvature)
        
        Args:
            G: Graph
            curvature_threshold: Edges with F < threshold are bottlenecks
            
        Returns:
            List of (u, v) edges that are bottlenecks
        """
        deg = dict(G.degree())
        
        def edge_triangles(u, v):
            Nu = set(G.neighbors(u)) - {v}
            Nv = set(G.neighbors(v)) - {u}
            return len(Nu & Nv)
        
        bottlenecks = []
        for u, v in G.edges():
            t = edge_triangles(u, v)
            f = (deg[u] + deg[v] - 2) - t
            if f < curvature_threshold:
                bottlenecks.append((u, v, f))
        
        return bottlenecks


class CoherenceCalculator:
    """Coherence (small-world) metrics"""
    
    @staticmethod
    def small_worldness(G: nx.Graph, n_random: int = 20) -> Dict[str, float]:
        """
        Small-worldness metrics (Humphries & Gurney sigma, Muldoon phi)
        
        σ = (C/C_rand) / (L/L_rand)
        
        Args:
            G: NetworkX graph
            n_random: Number of random graphs for comparison
            
        Returns:
            dict with sigma, phi, clustering, path_length
        """
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return {"sigma": 0.0, "phi": 0.0, "clustering": 0.0, "path_length": 0.0}
        
        try:
            C = nx.average_clustering(G)
            
            # Use largest connected component for path length
            largest_cc = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc).copy()
            L = nx.average_shortest_path_length(G_main)
            
            # Random graph equivalents
            Cr, Lr = 0.0, 0.0
            for _ in range(max(1, n_random)):
                Gr = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
                Cr += nx.average_clustering(Gr)
                
                # Use largest component
                if nx.is_connected(Gr):
                    Lr += nx.average_shortest_path_length(Gr)
                else:
                    largest_cc_r = max(nx.connected_components(Gr), key=len)
                    Gr_main = Gr.subgraph(largest_cc_r)
                    if Gr_main.number_of_nodes() > 1:
                        Lr += nx.average_shortest_path_length(Gr_main)
                    else:
                        Lr += L  # Fallback
            
            Cr /= max(1, n_random)
            Lr /= max(1, n_random)
            
            # Sigma (Humphries & Gurney)
            if Cr > 0 and Lr > 0:
                sigma = float((C/Cr) / (L/Lr))
            else:
                sigma = 0.0
            
            # Phi (Small-World Propensity - Muldoon)
            n = G.number_of_nodes()
            k_bar = float(np.mean([d for _, d in G.degree()]))
            
            # Lattice estimates
            C_latt = min(1.0, (3*(k_bar-2)) / (4*(k_bar-1)+1e-9)) if k_bar >= 2 else 0.0
            C_rand = k_bar / (n-1) if n > 1 else 0.0
            
            # Normalize clustering
            C_norm = (C - C_rand) / (C_latt - C_rand + 1e-12)
            C_norm = float(np.clip(C_norm, 0.0, 1.0))
            
            # Normalize path length
            L_latt = n / (2*max(1.0, k_bar))
            L_rand = (np.log(n) / np.log(max(2.0, k_bar))) if k_bar > 1 else L
            L_norm = (L_rand - L) / (L_rand - L_latt + 1e-12)
            L_norm = float(np.clip(L_norm, 0.0, 1.0))
            
            # SWP as average
            phi = float((C_norm + L_norm)/2.0)
            
            return {
                "sigma": sigma,
                "phi": phi,
                "clustering": float(C),
                "path_length": float(L),
                "efficiency": float(nx.global_efficiency(G_main)),
                "modularity": float(nx.algorithms.community.modularity(
                    G, nx.algorithms.community.greedy_modularity_communities(G)
                ))
            }
        
        except Exception as e:
            logger.debug(f"Small-world calculation error: {e}")
            return {"sigma": 0.0, "phi": 0.0, "clustering": 0.0, "path_length": 0.0,
                    "efficiency": 0.0, "modularity": 0.0}


# Aliases for backward compatibility
KECAlgorithms = type('KECAlgorithms', (), {
    'spectral_entropy': EntropyCalculator.spectral_entropy,
    'forman_curvature_stats': CurvatureCalculator.forman_curvature,
    'small_world_sigma': lambda G, n_random=20: CoherenceCalculator.small_worldness(G, n_random)['sigma'],
    'small_world_propensity': lambda G, n_random=20: CoherenceCalculator.small_worldness(G, n_random)['phi']
})

