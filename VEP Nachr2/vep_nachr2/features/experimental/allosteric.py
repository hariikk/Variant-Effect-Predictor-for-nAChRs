"""
Allosteric-network position features (H2).

H2 = GOF/LOF direction is a function of a residue's position in the allosteric
*communication network* — not its static geometry (Runs XIII–XVI) and not a coarse
functional-region label (H1, Run XVII). The idea: nAChR gating is a global, cooperative
conformational change, and a mutation's sign of effect depends on where the residue sits
along the ligand-binding → pore transduction pathway. A residue that is a network hub
(high betweenness) or on the pocket→pore shortest path is the kind of residue whose
perturbation propagates to the gate; peripheral residues are not.

Features (4), built from a residue-residue contact graph of the mapped PDB structure
(Cβ–Cβ < 8 Å; Cα for Gly), over ALL chains (so cross-subunit allostery is captured):

  1. betweenness      : normalized betweenness centrality — "allosteric hub" score.
  2. closeness        : normalized closeness centrality — global network centrality.
  3. gate_proximity   : 1 / (1 + d), where d = shortest-path edges to the nearest M2
                        gate-ring residue (middle 40% of TM2, mechanism.in_gate_ring).
                        1.0 = AT the gate, →0 = far. 0.0 = unmapped (is_unmappable in
                        structural_core flags the same thing).
  4. on_allosteric_path : 1.0 if the residue lies on any shortest pocket→gate path
                        (α subunits with annotated binding sites only, else 0.0).

Graceful degradation mirrors structural_nachr: unmappable/unavailable → 0.0. Alignment
maps are shared with StructuralExtractor (`_build_all_alignment_maps`), so numbering is
the same human-precursor frame already validated in Run XIV.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from Bio.PDB.Polypeptide import is_aa

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.structural import _build_all_alignment_maps
from vep_nachr2.features.experimental import mechanism as mech
from vep_nachr2.config import PDB_MAPPING


_CONTACT_CUTOFF = 8.0  # Å, Cβ–Cβ (Cα for Gly)


def _cb_or_ca(residue):
    """Representative coordinate: Cβ (Cα for Gly), else None."""
    try:
        if residue.resname == "GLY":
            return residue["CA"].coord if "CA" in residue else None
        if "CB" in residue:
            return residue["CB"].coord
        return residue["CA"].coord if "CA" in residue else None
    except (KeyError, AttributeError):
        return None


def _build_contact_graph(structure):
    """Residue-residue contact graph over all chains (nodes = (chain_id, resseq))."""
    nodes = []
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res):
                    continue
                c = _cb_or_ca(res)
                if c is None:
                    continue
                nodes.append((chain.id, res.id[1]))
                coords.append(c)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if len(coords) >= 2:
        arr = np.asarray(coords, dtype=np.float64)
        tree = cKDTree(arr)
        for i, j in tree.query_pairs(r=_CONTACT_CUTOFF):
            G.add_edge(nodes[i], nodes[j])
    return G


def _gate_nodes(pdb_id, alignment_maps, G):
    """M2 gate-ring residues (middle 40% of TM2) across every gene mapped to this PDB."""
    gate = set()
    for gene, info in PDB_MAPPING.items():
        if info["pdb_id"] != pdb_id:
            continue
        tm2 = mech.get_tm(gene, 2)
        if tm2 is None:
            continue
        s, e = tm2
        lo = s + int((e - s) * 0.30)
        hi = s + int((e - s) * 0.70)
        amap = alignment_maps.get(gene, {})
        chain_id = info["chain"]
        for pos in range(lo, hi + 1):
            r = amap.get(pos)
            if r is not None:
                node = (chain_id, r[1])
                if node in G:
                    gate.add(node)
    return gate


def _binding_nodes(gene, alignment_maps, G):
    """Agonist-binding-site residues of one α gene, mapped to the PDB graph."""
    residues = mech._BINDING_SITE_RESIDUES.get(gene, [])
    if not residues or gene not in PDB_MAPPING:
        return set()
    amap = alignment_maps.get(gene, {})
    chain_id = PDB_MAPPING[gene]["chain"]
    nodes = set()
    for pos in residues:
        r = amap.get(pos)
        if r is not None:
            node = (chain_id, r[1])
            if node in G:
                nodes.add(node)
    return nodes


def _multi_source_distance(G, sources):
    """BFS distances from a set of source nodes → {node: int}. Missing = unreachable."""
    from collections import deque

    dist = {}
    q = deque()
    for s in sources:
        if s in G:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in G[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _on_shortest_path_nodes(G, sources, targets):
    """Nodes lying on ≥1 shortest source→target path (subset betweenness > 0)."""
    sources = [s for s in sources if s in G]
    targets = [t for t in targets if t in G]
    if not sources or not targets:
        return set()
    bc = nx.betweenness_centrality_subset(
        G, sources=sources, targets=targets, normalized=False
    )
    return {node for node, val in bc.items() if val > 0}


class AllostericNetworkExtractor(FeatureExtractor):
    """Allosteric-network position features (H2)."""

    name = "allosteric"
    n_features = 4
    feature_names = [
        "betweenness", "closeness", "gate_proximity", "on_allosteric_path",
    ]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        if pdb_resources is None or ref_seqs is None:
            return features

        alignment_maps = _build_all_alignment_maps(ref_seqs, pdb_resources)

        # Precompute per-PDB graph + metrics (once, not per variant).
        pdb_data = {}
        for pdb_id, resources in pdb_resources.items():
            if resources is None:
                continue
            G = _build_contact_graph(resources["structure"])
            gate = _gate_nodes(pdb_id, alignment_maps, G)
            d = {
                "G": G,
                "betweenness": nx.betweenness_centrality(G, normalized=True) if G.number_of_nodes() else {},
                "closeness": nx.closeness_centrality(G) if G.number_of_nodes() else {},
                "dist_to_gate": _multi_source_distance(G, gate),
                "on_path": {},  # {gene: set(nodes on pocket→gate shortest path)}
            }
            for gene, info in PDB_MAPPING.items():
                if info["pdb_id"] != pdb_id:
                    continue
                binding = _binding_nodes(gene, alignment_maps, G)
                if binding:
                    d["on_path"][gene] = _on_shortest_path_nodes(G, binding, gate)
            pdb_data[pdb_id] = d

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))

            if gene not in PDB_MAPPING:
                continue
            info = PDB_MAPPING[gene]
            pdb_id = info["pdb_id"]
            d = pdb_data.get(pdb_id)
            if d is None:
                continue

            amap = alignment_maps.get(gene, {})
            r = amap.get(position)
            if r is None:
                continue
            node = (info["chain"], r[1])
            if node not in d["G"]:
                continue

            features[i, 0] = d["betweenness"].get(node, 0.0)
            features[i, 1] = d["closeness"].get(node, 0.0)
            dist = d["dist_to_gate"].get(node)
            if dist is not None:
                features[i, 2] = 1.0 / (1.0 + dist)
            features[i, 3] = 1.0 if node in d["on_path"].get(gene, set()) else 0.0

        return features
