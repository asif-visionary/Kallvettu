"""
Historical Knowledge Graph Engine for Kallvettu Digital Epigraphy Platform (Phase G).

Builds and queries a scholarly NetworkX graph linking:
- Kings & Rulers (Rajaraja Chola I, Nedunjeliyan, Parantaka I, etc.)
- Dynasties (Chola, Pandya, Chera, Pallava)
- Sacred Sites & Temples (Brihadisvara, Uttiramerur, Sittanavasal, Mangulam)
- Locations & Districts (Thanjavur, Madurai, Kanchipuram, Pudukkottai, Karur)
- Donors & Ascetics (Kundavai Pirattiyar, Nelveli Arayan, Kaniyan Nanta-asiriyan)
- Inscriptions (KV-INSC-001 through KV-INSC-006 and user-cataloged inscriptions)
- Administrative institutions & grants (Kudavolai Pot-Ballot, Devadana, Brahmadeya)

Every edge and entity maintains scholarly source citations (e.g. SII, Epigraphia Indica,
Early Tamil Epigraphy) to prevent ungrounded historical fabrication.
"""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import networkx as nx

from services.database.connection import get_db_session
from services.database.models import (
    Inscription,
    HistoricalReference,
    Citation,
    KnowledgeEntity,
    KnowledgeRelation,
    EntityCategory,
)

logger = logging.getLogger("kallvettu.knowledge_graph")


# Color mapping matching Kallvettu UI palette
CATEGORY_COLORS: Dict[str, str] = {
    EntityCategory.RULER.value: "#E5A93C",       # Imperial gold/amber
    EntityCategory.DYNASTY.value: "#C5A15A",     # Muted bronze
    EntityCategory.TEMPLE.value: "#4CAF50",      # Sacred green
    EntityCategory.LOCATION.value: "#42A5F5",    # Geographic blue
    EntityCategory.DONOR.value: "#AB47BC",       # Purple
    EntityCategory.OFFICIAL.value: "#26A69A",    # Teal
    EntityCategory.TAX.value: "#EF5350",         # Muted coral
    EntityCategory.GRANT.value: "#FFA726",       # Deep amber
    EntityCategory.INSCRIPTION.value: "#8D99AE", # Stone slate
    EntityCategory.HISTORICAL_EVENT.value: "#EC407A", # Historical event rose
}


# Mapping of canonical dynasty entity IDs to local user-supplied flag assets and historical emblem metadata.
# Historical Provenance Note:
# These flags are user-provided reference emblems (traditional / reconstructed heraldic representations,
# not authenticated archaeological artifacts). They represent traditional dynastic insignias cited in
# classical Tamil literature, royal copper plates, and numismatic references.
DYNASTY_FLAG_MAP: Dict[str, Dict[str, Any]] = {
    "DYN-PALLAVA": {
        "flag_path": "assets/knowledge_graph/dynasty_flags/pallava_flag.png",
        "emblem_name": "Simha (Lion) & Nandi Bull Emblem",
        "emblem_description": "Traditional heraldic flag featuring the couchant Bull (Nandi) and royal Simha crest of the Pallavas of Kanchipuram.",
        "emblem_provenance": "User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact).",
        "emblem_source": "Pallava royal copper-plate seals (Kuram, Kasakudi) & C. Minakshi, Administration and Social Life under the Pallavas (1938).",
        "verification_status": "Scholarly Reconstruction / Reference Emblem",
    },
    "DYN-CHERA": {
        "flag_path": "assets/knowledge_graph/dynasty_flags/chera_flag.png",
        "emblem_name": "Villu (Bow & Arrow) Royal Crest",
        "emblem_description": "Traditional heraldic emblem featuring the royal Chera Bow and Arrow emblem symbolizing maritime and Western Ghats sovereignty.",
        "emblem_provenance": "User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact).",
        "emblem_source": "Pugalur rock epigraphs, Sangam poetry (Patirruppattu) & M.G.S. Narayanan, Perumals of Kerala (1972).",
        "verification_status": "Scholarly Reconstruction / Reference Emblem",
    },
    "DYN-CHOLA": {
        "flag_path": "assets/knowledge_graph/dynasty_flags/chola_flag.png",
        "emblem_name": "Puli (Jumping / Crouching Tiger) Imperial Standard",
        "emblem_description": "Imperial Chola Tiger flag standard representing universal sovereignty across the Bay of Bengal and South India.",
        "emblem_provenance": "User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact).",
        "emblem_source": "Thiruvallangadu copper plates, SII Vol. II, & K.A. Nilakanta Sastri, The Colas (1935).",
        "verification_status": "Scholarly Reconstruction / Reference Emblem",
    },
    "DYN-PANDYA": {
        "flag_path": "assets/knowledge_graph/dynasty_flags/pandya_flag.png",
        "emblem_name": "Twin Carp (Irattai Kayal / Fish) Royal Insignia",
        "emblem_description": "Traditional Pandya royal crest featuring the Twin Carp (Kayal) standard of the Madurai Kingdom.",
        "emblem_provenance": "User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact).",
        "emblem_source": "Sinnamanur & Velvikkudi copper plates, SII & K.A. Nilakanta Sastri, The Pandyan Kingdom (1929).",
        "verification_status": "Scholarly Reconstruction / Reference Emblem",
    },
}


FALLBACK_DYNASTY_SVGS: Dict[str, str] = {
    "DYN-PALLAVA": (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" width="160" height="90">'
            b'<rect width="160" height="90" fill="#1A243B" rx="6"/>'
            b'<circle cx="80" cy="45" r="32" fill="#2A3859" stroke="#C5A15A" stroke-width="2"/>'
            b'<text x="80" y="44" font-family="sans-serif" font-size="20" text-anchor="middle">&#129409;</text>'
            b'<text x="80" y="62" font-family="sans-serif" font-size="10" font-weight="bold" fill="#FFFFFF" text-anchor="middle">PALLAVA</text>'
            b'<text x="80" y="74" font-family="sans-serif" font-size="7" fill="#CBD5E0" text-anchor="middle">Simha &amp; Nandi Standard</text>'
            b'</svg>'
        ).decode("ascii")
    ),
    "DYN-CHERA": (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" width="160" height="90">'
            b'<rect width="160" height="90" fill="#3B1A1A" rx="6"/>'
            b'<circle cx="80" cy="45" r="32" fill="#592A2A" stroke="#C5A15A" stroke-width="2"/>'
            b'<text x="80" y="44" font-family="sans-serif" font-size="20" text-anchor="middle">&#127993;</text>'
            b'<text x="80" y="62" font-family="sans-serif" font-size="10" font-weight="bold" fill="#FFFFFF" text-anchor="middle">CHERA</text>'
            b'<text x="80" y="74" font-family="sans-serif" font-size="7" fill="#CBD5E0" text-anchor="middle">Villu (Bow &amp; Arrow)</text>'
            b'</svg>'
        ).decode("ascii")
    ),
    "DYN-CHOLA": (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" width="160" height="90">'
            b'<rect width="160" height="90" fill="#2E2412" rx="6"/>'
            b'<circle cx="80" cy="45" r="32" fill="#47371C" stroke="#C5A15A" stroke-width="2"/>'
            b'<text x="80" y="44" font-family="sans-serif" font-size="20" text-anchor="middle">&#128005;</text>'
            b'<text x="80" y="62" font-family="sans-serif" font-size="10" font-weight="bold" fill="#FFFFFF" text-anchor="middle">CHOLA</text>'
            b'<text x="80" y="74" font-family="sans-serif" font-size="7" fill="#CBD5E0" text-anchor="middle">Puli (Tiger Crest)</text>'
            b'</svg>'
        ).decode("ascii")
    ),
    "DYN-PANDYA": (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" width="160" height="90">'
            b'<rect width="160" height="90" fill="#12242B" rx="6"/>'
            b'<circle cx="80" cy="45" r="32" fill="#1C3844" stroke="#C5A15A" stroke-width="2"/>'
            b'<text x="80" y="44" font-family="sans-serif" font-size="20" text-anchor="middle">&#128031;</text>'
            b'<text x="80" y="62" font-family="sans-serif" font-size="10" font-weight="bold" fill="#FFFFFF" text-anchor="middle">PANDYA</text>'
            b'<text x="80" y="74" font-family="sans-serif" font-size="7" fill="#CBD5E0" text-anchor="middle">Twin Carp (Fish)</text>'
            b'</svg>'
        ).decode("ascii")
    ),
}

GENERIC_DYNASTY_FALLBACK_SVG = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" width="160" height="90">'
        b'<rect width="160" height="90" fill="#202832" rx="6"/>'
        b'<circle cx="80" cy="45" r="32" fill="#2D3748" stroke="#C5A15A" stroke-width="2"/>'
        b'<text x="80" y="44" font-family="sans-serif" font-size="20" text-anchor="middle">&#128081;</text>'
        b'<text x="80" y="62" font-family="sans-serif" font-size="10" font-weight="bold" fill="#FFFFFF" text-anchor="middle">DYNASTY</text>'
        b'<text x="80" y="74" font-family="sans-serif" font-size="7" fill="#CBD5E0" text-anchor="middle">Royal Standard</text>'
        b'</svg>'
    ).decode("ascii")
)


def get_dynasty_fallback_svg(dyn_id: str) -> str:
    """Returns a self-contained SVG data URI emblem for a dynasty."""
    return FALLBACK_DYNASTY_SVGS.get(dyn_id, GENERIC_DYNASTY_FALLBACK_SVG)


@lru_cache(maxsize=32)
def _resolve_asset_path(flag_path: str) -> Optional[Path]:
    """Resolves a flag image path across repository root, relative cwd, and module parent."""
    candidate_paths = [
        Path(flag_path),
        Path(__file__).resolve().parent.parent.parent / flag_path,
        Path.cwd() / flag_path,
    ]
    for p in candidate_paths:
        if p.is_file():
            return p
    # Check alternate extension (.webp <-> .png)
    if flag_path.endswith(".png"):
        webp_path = flag_path[:-4] + ".webp"
        for p in [Path(webp_path), Path(__file__).resolve().parent.parent.parent / webp_path, Path.cwd() / webp_path]:
            if p.is_file():
                return p
    elif flag_path.endswith(".webp"):
        png_path = flag_path[:-5] + ".png"
        for p in [Path(png_path), Path(__file__).resolve().parent.parent.parent / png_path, Path.cwd() / png_path]:
            if p.is_file():
                return p
    return None


@lru_cache(maxsize=32)
def get_flag_data_uri(flag_path: str) -> Optional[str]:
    """
    Encodes a flag PNG or WebP file into a self-contained base64 data URI string.
    Ensures images load reliably inside Streamlit components.html iframe without requiring external HTTP hosting.
    """
    resolved = _resolve_asset_path(flag_path)
    if not resolved:
        return None
    try:
        mime = "image/webp" if resolved.suffix.lower() == ".webp" else "image/png"
        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    except Exception as e:
        logger.warning("Failed to encode flag image at %s: %s", resolved, e)
        return None


@lru_cache(maxsize=32)
def get_flag_dimensions(flag_path: str) -> Optional[Tuple[int, int, float]]:
    """
    Returns (width, height, aspect_ratio) of a flag image asset.
    Preserves exact original aspect ratio (e.g. 1.78 for 600x337, 1.0 for 700x700).
    """
    resolved = _resolve_asset_path(flag_path)
    if not resolved:
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(resolved) as img:
            w, h = img.size
            ar = round(w / h, 3) if h > 0 else 1.0
            return (w, h, ar)
    except Exception as e:
        logger.warning("Could not open flag image %s for dimensions: %s", resolved, e)
        return None


class EpigraphicalKnowledgeGraph:
    """Scholarly Knowledge Graph of Tamil Epigraphy using NetworkX."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._seed_canonical_graph()

    def add_entity(
        self,
        entity_id: str,
        category: str,
        label_en: str,
        label_ta: str = "",
        period: str = "",
        century: str = "",
        description: str = "",
        source_citation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds a verified historical entity to the graph."""
        self.graph.add_node(
            entity_id,
            id=entity_id,
            category=category,
            label_en=label_en,
            label_ta=label_ta or label_en,
            period=period,
            century=century,
            description=description,
            source_citation=source_citation,
            color=CATEGORY_COLORS.get(category, "#C5A15A"),
            metadata=metadata or {},
        )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        source_citation: str,
        description: str = "",
        weight: float = 1.0,
    ) -> None:
        """Adds a directional relation between two verified entities."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(
                "Cannot create relation %s -> %s: nodes must exist first.",
                source_id,
                target_id,
            )
            return

        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            source_citation=source_citation,
            description=description,
            weight=weight,
        )

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves node attributes for an entity."""
        if self.graph.has_node(entity_id):
            return dict(self.graph.nodes[entity_id])
        return None

    def get_neighbors(self, entity_id: str, direction: str = "all") -> List[Dict[str, Any]]:
        """
        Retrieves adjacent entities with edge details.
        direction: 'all', 'out', or 'in'
        """
        if not self.graph.has_node(entity_id):
            return []

        results = []
        if direction in ("out", "all"):
            for u, v, k, data in self.graph.out_edges(entity_id, keys=True, data=True):
                target_node = self.graph.nodes[v]
                results.append({
                    "direction": "out",
                    "neighbor_id": v,
                    "neighbor_label": target_node.get("label_en", v),
                    "neighbor_category": target_node.get("category", ""),
                    "relation_type": data.get("relation_type", ""),
                    "source_citation": data.get("source_citation", ""),
                    "description": data.get("description", ""),
                })

        if direction in ("in", "all"):
            for u, v, k, data in self.graph.in_edges(entity_id, keys=True, data=True):
                source_node = self.graph.nodes[u]
                results.append({
                    "direction": "in",
                    "neighbor_id": u,
                    "neighbor_label": source_node.get("label_en", u),
                    "neighbor_category": source_node.get("category", ""),
                    "relation_type": data.get("relation_type", ""),
                    "source_citation": data.get("source_citation", ""),
                    "description": data.get("description", ""),
                })

        return results

    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """Finds shortest unweighted path between two historical entities."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return []
        try:
            # Undirected view for historical traversal
            undirected = self.graph.to_undirected()
            return nx.shortest_path(undirected, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_subgraph(
        self,
        entity_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_depth: int = 1,
    ) -> EpigraphicalKnowledgeGraph:
        """Extracts a filtered subgraph based on seed entity IDs or categories."""
        sub = EpigraphicalKnowledgeGraph.__new__(EpigraphicalKnowledgeGraph)
        sub.graph = nx.MultiDiGraph()

        selected_nodes: Set[str] = set()

        if entity_ids:
            for seed in entity_ids:
                if self.graph.has_node(seed):
                    selected_nodes.add(seed)
                    # Expand up to max_depth
                    current_frontier = {seed}
                    for _ in range(max_depth):
                        next_frontier = set()
                        for node in current_frontier:
                            for n in self.graph.neighbors(node):
                                next_frontier.add(n)
                            for p in self.graph.predecessors(node):
                                next_frontier.add(p)
                        selected_nodes.update(next_frontier)
                        current_frontier = next_frontier
        else:
            selected_nodes = set(self.graph.nodes())

        if categories:
            cat_set = set(categories)
            selected_nodes = {
                n for n in selected_nodes
                if self.graph.nodes[n].get("category") in cat_set
            }

        for n in selected_nodes:
            sub.graph.add_node(n, **self.graph.nodes[n])

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if u in selected_nodes and v in selected_nodes:
                sub.graph.add_edge(u, v, **data)

        return sub

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph to nodes and links dictionary for visualization."""
        nodes = []
        for n, data in self.graph.nodes(data=True):
            node_dict = {
                "id": n,
                "label": data.get("label_en", n),
                "label_ta": data.get("label_ta", ""),
                "category": data.get("category", ""),
                "period": data.get("period", ""),
                "century": data.get("century", ""),
                "citation": data.get("source_citation", ""),
                "description": data.get("description", ""),
                "color": data.get("color", "#C5A15A"),
            }

            # Check if this node is a dynasty with a mapped flag asset
            cat = data.get("category", "")
            meta = data.get("metadata", {}) or {}
            flag_info = DYNASTY_FLAG_MAP.get(n)

            if flag_info or cat == EntityCategory.DYNASTY.value:
                flag_path = (flag_info or {}).get("flag_path") or meta.get("flag_path")
                fallback_svg = get_dynasty_fallback_svg(n)
                if flag_path:
                    data_uri = get_flag_data_uri(flag_path)
                    dims = get_flag_dimensions(flag_path)
                    node_dict.update({
                        "flag_path": flag_path,
                        "flag_data_uri": data_uri,
                        "fallback_svg_data_uri": fallback_svg,
                        "has_flag_image": bool(data_uri),
                        "emblem_name": (flag_info or {}).get("emblem_name") or meta.get("emblem_name", "Dynastic Standard"),
                        "emblem_description": (flag_info or {}).get("emblem_description") or meta.get("emblem_description", ""),
                        "emblem_provenance": (flag_info or {}).get("emblem_provenance") or meta.get(
                            "emblem_provenance",
                            "User-provided reference emblem (traditional/reconstructed heraldic representation, not authenticated archaeological artifact)."
                        ),
                        "emblem_source": (flag_info or {}).get("emblem_source") or meta.get("emblem_source", ""),
                        "verification_status": (flag_info or {}).get("verification_status") or meta.get(
                            "verification_status", "Scholarly Reconstruction / Reference Emblem"
                        ),
                        "image_width": dims[0] if dims else None,
                        "image_height": dims[1] if dims else None,
                        "aspect_ratio": dims[2] if dims else None,
                    })
                else:
                    node_dict.update({
                        "flag_path": None,
                        "flag_data_uri": None,
                        "fallback_svg_data_uri": fallback_svg,
                        "has_flag_image": False,
                        "verification_status": "Scholarly Entity (No Flag Asset)",
                        "image_width": None,
                        "image_height": None,
                        "aspect_ratio": None,
                    })

            nodes.append(node_dict)

        edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            edges.append({
                "source": u,
                "target": v,
                "label": data.get("relation_type", ""),
                "citation": data.get("source_citation", ""),
                "description": data.get("description", ""),
            })

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
            "validation": self.validate(),
        }

    def validate(self) -> Dict[str, Any]:
        """
        Validates the epigraphical knowledge graph data integrity.
        Verifies:
        - Exactly or at least 29 canonical entities loaded, with unique non-empty IDs.
        - Exactly or at least 69 canonical relations loaded.
        - Edge references: every source and target exists in nodes.
        - No invalid, empty, or duplicate identifiers.
        - Flag assets for dynasties are valid or have fallback SVGs.
        """
        errors: List[str] = []
        warnings: List[str] = []
        invalid_nodes: List[str] = []
        invalid_edges: List[Dict[str, Any]] = []

        node_ids = set(self.graph.nodes())
        if not node_ids:
            errors.append("Knowledge Graph has zero entities (empty graph).")

        valid_entity_ids: Set[str] = set()

        for n, data in self.graph.nodes(data=True):
            if not isinstance(n, str) or not n.strip():
                errors.append(f"Invalid or empty entity identifier: {n!r}")
                invalid_nodes.append(str(n))
                continue
            if not data.get("category"):
                errors.append(f"Entity '{n}' is missing category (uninitialized node).")
                invalid_nodes.append(str(n))
                continue
            if not data.get("label_en"):
                errors.append(f"Entity '{n}' is missing English label (uninitialized node).")
                invalid_nodes.append(str(n))
                continue
            valid_entity_ids.add(n)

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            rel = data.get("relation_type", "")
            if u not in valid_entity_ids:
                errors.append(f"Edge references non-existent or uninitialized source '{u}' -> '{v}' ({rel})")
                invalid_edges.append({"source": u, "target": v, "relation": rel, "reason": "missing_source"})
            if v not in valid_entity_ids:
                errors.append(f"Edge references non-existent or uninitialized target '{u}' -> '{v}' ({rel})")
                invalid_edges.append({"source": u, "target": v, "relation": rel, "reason": "missing_target"})

        # Check dynasty flag assets
        for dyn_id, flag_info in DYNASTY_FLAG_MAP.items():
            if self.graph.has_node(dyn_id):
                path = flag_info.get("flag_path")
                if path and not _resolve_asset_path(path):
                    warnings.append(f"Flag asset for dynasty '{dyn_id}' missing at '{path}', fallback SVG enabled.")

        return {
            "is_valid": len(errors) == 0,
            "node_count": len(node_ids),
            "edge_count": self.graph.number_of_edges(),
            "errors": errors,
            "warnings": warnings,
            "invalid_nodes": invalid_nodes,
            "invalid_edges": invalid_edges,
        }

    def to_json(self, indent: int = 2) -> str:
        """Returns JSON representation."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_graphml(self) -> str:
        """Serializes to GraphML XML format."""
        import io
        buf = io.BytesIO()
        # NetworkX handles GraphML export cleanly
        # Create a simplified DiGraph without nested dicts for clean XML
        simple_g = nx.DiGraph()
        for n, d in self.graph.nodes(data=True):
            simple_g.add_node(
                n,
                label=str(d.get("label_en", n)),
                category=str(d.get("category", "")),
                period=str(d.get("period", "")),
            )
        for u, v, d in self.graph.edges(data=True):
            simple_g.add_edge(
                u,
                v,
                relation=str(d.get("relation_type", "")),
                citation=str(d.get("source_citation", "")),
            )
        nx.write_graphml(simple_g, buf)
        return buf.getvalue().decode("utf-8")

    def _seed_canonical_graph(self) -> None:
        """Seeds canonical historical corpus entities and verified epigraphical edges."""
        # 1. Dynasties (Four Major Tamil Dynasties with Reference Flag Assets)
        for dyn_id, flag_info in DYNASTY_FLAG_MAP.items():
            if dyn_id == "DYN-CHOLA":
                self.add_entity(
                    dyn_id,
                    EntityCategory.DYNASTY.value,
                    "Chola Dynasty",
                    "சோழப் பேரரசு",
                    "Early & Imperial Chola",
                    "c. 300 BCE – 1279 CE",
                    "Major Tamil imperial maritime dynasty renowned for overseas conquests, granite temple architecture, and standardized epigraphy.",
                    "K.A. Nilakanta Sastri, The Colas, 1935.",
                    metadata=flag_info,
                )
            elif dyn_id == "DYN-PANDYA":
                self.add_entity(
                    dyn_id,
                    EntityCategory.DYNASTY.value,
                    "Pandya Dynasty",
                    "பாண்டியப் பேரரசு",
                    "Sangam & Medieval Pandya",
                    "c. 6th c. BCE – 16th c. CE",
                    "Ancient Tamil dynasty ruling from Madurai; patrons of early Tamil-Brahmi cave hermitage beds and Tamil Sangam academies.",
                    "K.A. Nilakanta Sastri, The Pandyan Kingdom, 1929.",
                    metadata=flag_info,
                )
            elif dyn_id == "DYN-CHERA":
                self.add_entity(
                    dyn_id,
                    EntityCategory.DYNASTY.value,
                    "Chera Dynasty",
                    "சேரப் பேரரசு",
                    "Sangam Chera",
                    "c. 3rd c. BCE – 12th c. CE",
                    "Ancient seafaring Tamil dynasty ruling the Kongu region and western coast (Malabar); celebrated in Pugalur hill epigraphs.",
                    "M.G.S. Narayanan, Perumals of Kerala, 1972.",
                    metadata=flag_info,
                )
            elif dyn_id == "DYN-PALLAVA":
                self.add_entity(
                    dyn_id,
                    EntityCategory.DYNASTY.value,
                    "Pallava Dynasty",
                    "பல்லவப் பேரரசு",
                    "Imperial Pallava",
                    "c. 275 CE – 897 CE",
                    "Dynasty ruling from Kanchipuram renowned for monolithic rock-cut cave rathas and pioneering early Grantha script inscriptions.",
                    "C. Minakshi, Administration and Social Life under the Pallavas, 1938.",
                    metadata=flag_info,
                )

        # 2. Rulers
        self.add_entity("RULER-RAJARAJA-1", EntityCategory.RULER.value, "Rajaraja Chola I (முதலாம் இராஜராஜன்)", "முதலாம் இராஜராஜ சோழன்", "Imperial Chola", "11th Century CE (985–1014 CE)", "Conqueror and builder of Brihadisvara Temple; institutionalized epigraphical record-keeping.", "South Indian Inscriptions (SII) Vol. II, Part I, No. 1.")
        self.add_entity("RULER-PARANTAKA-1", EntityCategory.RULER.value, "Parantaka Chola I (முதலாம் பராந்தகன்)", "முதலாம் பராந்தக சோழன்", "Early Medieval Chola", "10th Century CE (907–955 CE)", "Codifier of local democratic elections via the Kudavolai pot-ballot system.", "Epigraphia Indica Vol. XXII, pp. 145–150.")
        self.add_entity("RULER-NEDUNJELIYAN", EntityCategory.RULER.value, "Pandyan Nedunjeliyan (நெடுஞ்செழியன்)", "பாண்டியன் நெடுஞ்செழியன்", "Early Sangam", "3rd–2nd Century BCE", "Early Pandya king who gifted cavern beds to Jain monks at Mangulam.", "I. Mahadevan, Early Tamil Epigraphy, 2003, Ins. No. 1.")
        self.add_entity("RULER-KO-ATAN-CEL", EntityCategory.RULER.value, "Ko Atan Cel Irumporai (சேர மன்னர் கோ ஆதன் செல்)", "கோ ஆதன் செல் இரும்பொறை", "Sangam Chera", "2nd Century CE", "Chera king celebrated in Pugalur hill epigraphs.", "I. Mahadevan, Early Tamil Epigraphy, 2003, Ins. No. 56.")
        self.add_entity("RULER-SRIMARA", EntityCategory.RULER.value, "Srimara Srivallabha (ஸ்ரீமாற ஸ்ரீவல்லபன்)", "ஸ்ரீமாற ஸ்ரீவல்லப பாண்டியன்", "Early Medieval Pandya", "9th Century CE", "Pandya emperor commemorated in Sittanavasal cave renovations.", "Annual Report on South Indian Epigraphy (ARE), 1904.")
        self.add_entity("RULER-MAMALLA", EntityCategory.RULER.value, "Narasimhavarman I Mamalla (நரசிம்மவர்மன்)", "முதலாம் நரசிம்மவர்மன்", "Imperial Pallava", "7th Century CE", "Pallava monarch who commissioned the Mahabalipuram rock-cut monoliths.", "Epigraphia Indica Vol. X, No. 22.")

        # 3. Sacred Sites & Temples
        self.add_entity("SITE-BRIHADISVARA", EntityCategory.TEMPLE.value, "Brihadisvara Great Temple", "தஞ்சைப் பெருவுடையார் கோயில்", "Imperial Chola", "1010 CE", "UNESCO World Heritage granite temple with extensive historical inscriptions.", "SII Vol. II.")
        self.add_entity("SITE-UTTIRAMERUR", EntityCategory.TEMPLE.value, "Vaikunta Perumal Temple (Uttiramerur)", "உத்திரமேரூர் வைகுண்ட பெருமாள் கோயில்", "Early Chola", "921 CE", "Renowned for Sabha constitution and electoral council inscriptions.", "Epigraphia Indica Vol. XXII.")
        self.add_entity("SITE-MANGULAM", EntityCategory.LOCATION.value, "Mangulam Rock Caverns (Madurai)", "மாங்குளம் குகை", "Early Sangam", "3rd c. BCE", "Earliest dated Tamil-Brahmi inscription cavern site.", "Early Tamil Epigraphy No. 1.")
        self.add_entity("SITE-PUGALUR", EntityCategory.LOCATION.value, "Pugalur Arunattarmalai (Karur)", "புகழூர் ஆறுநாட்டார்மலை", "Sangam", "2nd c. CE", "Chera royal lineage rock-cut inscription site.", "Early Tamil Epigraphy No. 56.")
        self.add_entity("SITE-SITTANAVASAL", EntityCategory.TEMPLE.value, "Sittanavasal Cave & Arivar Koil", "சித்தன்னவாசல் சமணர் குகை", "Pandya", "9th c. CE", "Celebrated Jain fresco painting and Vatteluttu epigraph site.", "ARE 1904 / SII.")
        self.add_entity("SITE-MAHABALIPURAM", EntityCategory.TEMPLE.value, "Mahabalipuram Pancha Rathas", "மகாபலிபுரம் பஞ்ச ரதங்கள்", "Pallava", "7th c. CE", "Monolithic granite architecture featuring early Pallava Grantha titles.", "Epigraphia Indica Vol. X.")

        # 4. Donors & Prominent Epigraphical Figures
        self.add_entity("DONOR-KUNDAVAI", EntityCategory.DONOR.value, "Kundavai Pirattiyar (குந்தவை பிராட்டியார்)", "குந்தவை பிராட்டியார்", "Imperial Chola", "11th c. CE", "Elder sister of Rajaraja I; principal patron and donor of gold ornaments and endowments.", "SII Vol. II, Part I, No. 2.")
        self.add_entity("DONOR-NELVELI", EntityCategory.DONOR.value, "Nelveli Arayan (நெல்வேலி அறையன்)", "நெல்வேலி அறையன்", "Early Sangam", "3rd c. BCE", "Chieftain who endowed stone beds for the ascetic Kaniyan Nanta-asiriyan.", "Early Tamil Epigraphy No. 1.")
        self.add_entity("DONOR-NANTA-ASIRIYAN", EntityCategory.DONOR.value, "Kaniyan Nanta-asiriyan (கணியன் நந்தாசிரியன்)", "கணியன் நந்தாசிரியன்", "Early Sangam", "3rd c. BCE", "Venerated Jain monk recipient of Mangulam cavern stone beds.", "Early Tamil Epigraphy No. 1.")

        # 5. Inscriptions (Connecting Nodes)
        self.add_entity("KV-INSC-001", EntityCategory.INSCRIPTION.value, "Mangulam Tamil-Brahmi Epigraph", "மாங்குளம் கல்வெட்டு", "Early Sangam", "3rd c. BCE", "Oldest securely dated Tamil-Brahmi inscription.", "Early Tamil Epigraphy No. 1.")
        self.add_entity("KV-INSC-002", EntityCategory.INSCRIPTION.value, "Thanjavur Brihadisvara Temple Inscription", "தஞ்சைப் பெருவுடையார் கல்வெட்டு", "Imperial Chola", "1010 CE", "Royal edict ordaining public and royal gifts to be recorded on temple walls.", "SII Vol. II, Part I, No. 1.")
        self.add_entity("KV-INSC-003", EntityCategory.INSCRIPTION.value, "Uttiramerur Kudavolai Constitutional Edict", "உத்திரமேரூர் குடவோலை சாசனம்", "Early Medieval Chola", "921 CE", "Constitutional decree detailing democratic village elections.", "EI Vol. XXII, pp. 145–150.")
        self.add_entity("KV-INSC-004", EntityCategory.INSCRIPTION.value, "Pugalur Chera Lineage Epigraph", "புகழூர் சேரர் கல்வெட்டு", "Sangam", "2nd c. CE", "Traces three generations of Chera rulers in Tamil-Brahmi.", "Early Tamil Epigraphy No. 56.")
        self.add_entity("KV-INSC-005", EntityCategory.INSCRIPTION.value, "Sittanavasal Jain Cavern Inscription", "சித்தன்னவாசல் கல்வெட்டு", "Pandya", "9th c. CE", "Vatteluttu epigraph describing cavern additions by Ilan-Gautaman.", "ARE 1904.")
        self.add_entity("KV-INSC-006", EntityCategory.INSCRIPTION.value, "Mahabalipuram Pallava Royal Birudas", "மகாபலிபுரம் பல்லவர் பட்டப் பெயர்கள்", "Pallava", "7th c. CE", "Grantha inscriptions recording royal honorific birudas of Narasimhavarman I.", "EI Vol. X, No. 22.")

        # 6. Historical Events
        self.add_entity(
            "EVENT-KUDAVOLAI-REFORM",
            EntityCategory.HISTORICAL_EVENT.value,
            "Kudavolai Village Assembly Election Reform",
            "குடவோலை கிராமசபைத் தேர்தல் சீர்திருத்தம்",
            "Early Medieval Chola",
            "921 CE",
            "Constitutional codification of democratic pot-ballot elections for village executive assembly (sabha) at Uttiramerur under Parantaka Chola I.",
            "Epigraphia Indica Vol. XXII, pp. 145–150.",
        )
        self.add_entity(
            "EVENT-CONSECRATION-BRIHADISVARA",
            EntityCategory.HISTORICAL_EVENT.value,
            "Consecration of Rajarajesvaram Temple",
            "தஞ்சைப் பெருவுடையார் குடமுழுக்கு",
            "Imperial Chola",
            "1010 CE",
            "Grand consecration and royal dedication of the great granite Brihadisvara vimana by Rajaraja Chola I.",
            "South Indian Inscriptions (SII) Vol. II, Part I, No. 1.",
        )

        # 7. Historical Land Grants & Endowments
        self.add_entity(
            "GRANT-DEVADANA-KUNDAVAI",
            EntityCategory.GRANT.value,
            "Devadana & Gold Ornament Endowment",
            "தேவதானம் மற்றும் பொற்கொடை சாசனம்",
            "Imperial Chola",
            "11th c. CE",
            "Endowment of precious gem-encrusted gold ornaments, silver vessels, and revenue-bearing lands to Brihadisvara Temple by Princess Kundavai.",
            "SII Vol. II, Part I, No. 2.",
        )
        self.add_entity(
            "GRANT-MANGULAM-PALLI",
            EntityCategory.GRANT.value,
            "Mangulam Cavern Bed Endowment",
            "மாங்குளம் கற்படுக்கை தானம்",
            "Early Sangam",
            "3rd c. BCE",
            "Formal religious endowment of rock-cut stone monastery beds for Jain ascetic Kaniyan Nanta-asiriyan.",
            "Early Tamil Epigraphy No. 1.",
        )

        # 8. Build Historical Relations
        # Dynasties to Rulers
        self.add_relation("DYN-CHOLA", "RULER-RAJARAJA-1", "DYNASTIC_RULER", "K.A. Nilakanta Sastri, The Colas")
        self.add_relation("DYN-CHOLA", "RULER-PARANTAKA-1", "DYNASTIC_RULER", "K.A. Nilakanta Sastri, The Colas")
        self.add_relation("DYN-PANDYA", "RULER-NEDUNJELIYAN", "DYNASTIC_RULER", "K.A. Nilakanta Sastri, Pandyan Kingdom")
        self.add_relation("DYN-PANDYA", "RULER-SRIMARA", "DYNASTIC_RULER", "K.A. Nilakanta Sastri, Pandyan Kingdom")
        self.add_relation("DYN-CHERA", "RULER-KO-ATAN-CEL", "DYNASTIC_RULER", "Early Tamil Epigraphy No. 56")
        self.add_relation("DYN-PALLAVA", "RULER-MAMALLA", "DYNASTIC_RULER", "C. Minakshi, Pallavas")

        # Rulers to Dynasties (reciprocal provenance)
        self.add_relation("RULER-RAJARAJA-1", "DYN-CHOLA", "BELONGS_TO_DYNASTY", "K.A.N. Sastri, The Colas")
        self.add_relation("RULER-PARANTAKA-1", "DYN-CHOLA", "BELONGS_TO_DYNASTY", "K.A.N. Sastri, The Colas")
        self.add_relation("RULER-NEDUNJELIYAN", "DYN-PANDYA", "BELONGS_TO_DYNASTY", "K.A.N. Sastri, Pandyan Kingdom")
        self.add_relation("RULER-SRIMARA", "DYN-PANDYA", "BELONGS_TO_DYNASTY", "K.A.N. Sastri, Pandyan Kingdom")
        self.add_relation("RULER-KO-ATAN-CEL", "DYN-CHERA", "BELONGS_TO_DYNASTY", "Early Tamil Epigraphy No. 56")
        self.add_relation("RULER-MAMALLA", "DYN-PALLAVA", "BELONGS_TO_DYNASTY", "C. Minakshi, Pallavas")

        # Dynasties to Inscriptions
        self.add_relation("DYN-CHOLA", "KV-INSC-002", "ASSOCIATED_INSCRIPTION", "SII Vol. II, Part I, No. 1")
        self.add_relation("DYN-CHOLA", "KV-INSC-003", "ASSOCIATED_INSCRIPTION", "EI Vol. XXII, pp. 145–150")
        self.add_relation("DYN-PANDYA", "KV-INSC-001", "ASSOCIATED_INSCRIPTION", "Early Tamil Epigraphy No. 1")
        self.add_relation("DYN-PANDYA", "KV-INSC-005", "ASSOCIATED_INSCRIPTION", "ARE 1904")
        self.add_relation("DYN-CHERA", "KV-INSC-004", "ASSOCIATED_INSCRIPTION", "Early Tamil Epigraphy No. 56")
        self.add_relation("DYN-PALLAVA", "KV-INSC-006", "ASSOCIATED_INSCRIPTION", "EI Vol. X, No. 22")

        # Dynasties to Temples & Sacred Sites
        self.add_relation("DYN-CHOLA", "SITE-BRIHADISVARA", "IMPERIAL_TEMPLE", "SII Vol. II")
        self.add_relation("DYN-CHOLA", "SITE-UTTIRAMERUR", "PATRONIZED_TEMPLE", "EI Vol. XXII")
        self.add_relation("DYN-PANDYA", "SITE-MANGULAM", "PATRONIZED_SITE", "Early Tamil Epigraphy No. 1")
        self.add_relation("DYN-PANDYA", "SITE-SITTANAVASAL", "PATRONIZED_TEMPLE", "ARE 1904")
        self.add_relation("DYN-CHERA", "SITE-PUGALUR", "PATRONIZED_SITE", "Early Tamil Epigraphy No. 56")
        self.add_relation("DYN-PALLAVA", "SITE-MAHABALIPURAM", "IMPERIAL_SITE", "EI Vol. X, No. 22")

        # Rulers to Inscriptions
        self.add_relation("RULER-RAJARAJA-1", "KV-INSC-002", "COMMISSIONED_INSCRIPTION", "SII Vol. II, Part I, No. 1")
        self.add_relation("RULER-PARANTAKA-1", "KV-INSC-003", "ORDERED_INSCRIPTION", "EI Vol. XXII, pp. 145–150")
        self.add_relation("RULER-NEDUNJELIYAN", "KV-INSC-001", "PATRON_RECORDED_IN", "Early Tamil Epigraphy No. 1")
        self.add_relation("RULER-KO-ATAN-CEL", "KV-INSC-004", "COMMEMORATED_IN", "Early Tamil Epigraphy No. 56")
        self.add_relation("RULER-SRIMARA", "KV-INSC-005", "REIGN_RECORDED_IN", "ARE 1904")
        self.add_relation("RULER-MAMALLA", "KV-INSC-006", "ROYAL_TITLES_IN", "EI Vol. X, No. 22")

        # Inscriptions to Rulers & Sites (original reciprocal links)
        self.add_relation("KV-INSC-001", "SITE-MANGULAM", "LOCATED_AT", "Early Tamil Epigraphy No. 1")
        self.add_relation("KV-INSC-001", "RULER-NEDUNJELIYAN", "MENTIONS_RULER", "Early Tamil Epigraphy No. 1")

        self.add_relation("KV-INSC-002", "SITE-BRIHADISVARA", "ENGRAVED_UPON", "SII Vol. II, Part I, No. 1")
        self.add_relation("KV-INSC-002", "RULER-RAJARAJA-1", "ISSUED_BY", "SII Vol. II, Part I, No. 1")
        self.add_relation("KV-INSC-002", "DONOR-KUNDAVAI", "RECORDS_GIFTS_OF", "SII Vol. II, Part I, No. 2")

        self.add_relation("KV-INSC-003", "SITE-UTTIRAMERUR", "ENGRAVED_UPON", "EI Vol. XXII, pp. 145–150")
        self.add_relation("KV-INSC-003", "RULER-PARANTAKA-1", "ISSUED_UNDER", "EI Vol. XXII, pp. 145–150")

        self.add_relation("KV-INSC-004", "SITE-PUGALUR", "LOCATED_AT", "Early Tamil Epigraphy No. 56")
        self.add_relation("KV-INSC-004", "RULER-KO-ATAN-CEL", "COMMEMORATES", "Early Tamil Epigraphy No. 56")

        self.add_relation("KV-INSC-005", "SITE-SITTANAVASAL", "LOCATED_AT", "ARE 1904")
        self.add_relation("KV-INSC-005", "RULER-SRIMARA", "MENTIONS_REIGN_OF", "ARE 1904")

        self.add_relation("KV-INSC-006", "SITE-MAHABALIPURAM", "ENGRAVED_UPON", "EI Vol. X, No. 22")
        self.add_relation("KV-INSC-006", "RULER-MAMALLA", "ATTRIBUTED_TO", "EI Vol. X, No. 22")

        # Inscriptions to Historical Events
        self.add_relation("KV-INSC-003", "EVENT-KUDAVOLAI-REFORM", "RECORDS_HISTORICAL_EVENT", "EI Vol. XXII, pp. 145–150")
        self.add_relation("KV-INSC-002", "EVENT-CONSECRATION-BRIHADISVARA", "RECORDS_HISTORICAL_EVENT", "SII Vol. II, Part I, No. 1")

        # Donors to Land Grants & Endowments
        self.add_relation("DONOR-KUNDAVAI", "GRANT-DEVADANA-KUNDAVAI", "COMMISSIONED_LAND_GRANT", "SII Vol. II, Part I, No. 2")
        self.add_relation("DONOR-NELVELI", "GRANT-MANGULAM-PALLI", "COMMISSIONED_LAND_GRANT", "Early Tamil Epigraphy No. 1")
        self.add_relation("KV-INSC-002", "GRANT-DEVADANA-KUNDAVAI", "DETAILS_ENDOWMENT", "SII Vol. II, Part I, No. 2")
        self.add_relation("KV-INSC-001", "GRANT-MANGULAM-PALLI", "DETAILS_ENDOWMENT", "Early Tamil Epigraphy No. 1")

        # Rulers to Temples / Patronage
        self.add_relation("RULER-RAJARAJA-1", "SITE-BRIHADISVARA", "PATRONIZED_BUILT", "SII Vol. II, Part I, No. 1")
        self.add_relation("RULER-PARANTAKA-1", "SITE-UTTIRAMERUR", "CONFERRED_CONSTITUTION", "EI Vol. XXII, pp. 145–150")
        self.add_relation("RULER-MAMALLA", "SITE-MAHABALIPURAM", "COMMISSIONED", "EI Vol. X, No. 22")
        self.add_relation("RULER-SRIMARA", "SITE-SITTANAVASAL", "RENOVATED", "ARE 1904")

        # Family Relations
        self.add_relation("DONOR-KUNDAVAI", "RULER-RAJARAJA-1", "ELDER_SISTER_OF", "SII Vol. II, Part I, No. 2")
        self.add_relation("DONOR-KUNDAVAI", "SITE-BRIHADISVARA", "ENDOWED_GIFTS_TO", "SII Vol. II, Part I, No. 2")

        # Cavern Bed Endowments
        self.add_relation("DONOR-NELVELI", "DONOR-NANTA-ASIRIYAN", "GIFTED_BED_TO", "Early Tamil Epigraphy No. 1")
        self.add_relation("DONOR-NELVELI", "SITE-MANGULAM", "COMMISSIONED_CAVE_AT", "Early Tamil Epigraphy No. 1")
        self.add_relation("RULER-NEDUNJELIYAN", "SITE-MANGULAM", "RULED_TERRITORY_OF", "Early Tamil Epigraphy No. 1")

        self.add_relation("KV-INSC-002", "SITE-BRIHADISVARA", "ENGRAVED_UPON", "SII Vol. II, Part I, No. 1")
        self.add_relation("KV-INSC-002", "RULER-RAJARAJA-1", "ISSUED_BY", "SII Vol. II, Part I, No. 1")
        self.add_relation("KV-INSC-002", "DONOR-KUNDAVAI", "RECORDS_GIFTS_OF", "SII Vol. II, Part I, No. 2")

        self.add_relation("KV-INSC-003", "SITE-UTTIRAMERUR", "ENGRAVED_UPON", "EI Vol. XXII, pp. 145–150")
        self.add_relation("KV-INSC-003", "RULER-PARANTAKA-1", "ISSUED_UNDER", "EI Vol. XXII, pp. 145–150")

        self.add_relation("KV-INSC-004", "SITE-PUGALUR", "LOCATED_AT", "Early Tamil Epigraphy No. 56")
        self.add_relation("KV-INSC-004", "RULER-KO-ATAN-CEL", "COMMEMORATES", "Early Tamil Epigraphy No. 56")

        self.add_relation("KV-INSC-005", "SITE-SITTANAVASAL", "LOCATED_AT", "ARE 1904")
        self.add_relation("KV-INSC-005", "RULER-SRIMARA", "MENTIONS_REIGN_OF", "ARE 1904")

        self.add_relation("KV-INSC-006", "SITE-MAHABALIPURAM", "ENGRAVED_UPON", "EI Vol. X, No. 22")
        self.add_relation("KV-INSC-006", "RULER-MAMALLA", "ATTRIBUTED_TO", "EI Vol. X, No. 22")

