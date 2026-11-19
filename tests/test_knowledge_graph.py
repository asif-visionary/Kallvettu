"""
Unit Tests for Historical Knowledge Graph Engine & Renderer (Phase G).
"""

import json
import pytest
import networkx as nx
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph
from services.knowledge_graph.graph_renderer import render_knowledge_graph_html
from services.database.models import EntityCategory


def test_knowledge_graph_initialization():
    """Verify default graph is created and populated with canonical entities."""
    kg = EpigraphicalKnowledgeGraph()
    data = kg.to_dict()

    assert data["node_count"] >= 15
    assert data["edge_count"] >= 12

    # Check key canonical nodes exist
    assert kg.graph.has_node("DYN-CHOLA")
    assert kg.graph.has_node("RULER-RAJARAJA-1")
    assert kg.graph.has_node("SITE-BRIHADISVARA")
    assert kg.graph.has_node("KV-INSC-002")


def test_node_attributes_and_citations():
    """Verify nodes preserve scholarly citations and bilingual titles."""
    kg = EpigraphicalKnowledgeGraph()
    rajaraja = kg.get_entity("RULER-RAJARAJA-1")

    assert rajaraja is not None
    assert rajaraja["category"] == EntityCategory.RULER.value
    assert "முதலாம் இராஜராஜ" in rajaraja["label_ta"]
    assert "South Indian Inscriptions" in rajaraja["source_citation"]

    mangulam = kg.get_entity("KV-INSC-001")
    assert mangulam is not None
    assert "Early Tamil Epigraphy" in mangulam["source_citation"]


def test_pathfinding_and_traversal():
    """Verify historical pathfinding between connected entities."""
    kg = EpigraphicalKnowledgeGraph()

    # Path from Kundavai to Brihadisvara Temple
    path = kg.find_path("DONOR-KUNDAVAI", "SITE-BRIHADISVARA")
    assert len(path) >= 2
    assert path[0] == "DONOR-KUNDAVAI"
    assert path[-1] == "SITE-BRIHADISVARA"

    # Neighbors of Rajaraja Chola I
    neighbors = kg.get_neighbors("RULER-RAJARAJA-1", direction="all")
    assert len(neighbors) >= 2
    neighbor_ids = [n["neighbor_id"] for n in neighbors]
    assert "DYN-CHOLA" in neighbor_ids
    assert "SITE-BRIHADISVARA" in neighbor_ids


def test_subgraph_extraction():
    """Verify filtering subgraph by seed nodes and categories."""
    kg = EpigraphicalKnowledgeGraph()

    # Extract subgraph around Nedunjeliyan with depth 1
    sub = kg.get_subgraph(entity_ids=["RULER-NEDUNJELIYAN"], max_depth=1)
    sub_data = sub.to_dict()
    assert sub_data["node_count"] >= 2
    assert any(n["id"] == "DYN-PANDYA" for n in sub_data["nodes"])

    # Extract only Dynasties
    dyn_sub = kg.get_subgraph(categories=[EntityCategory.DYNASTY.value])
    dyn_data = dyn_sub.to_dict()
    assert all(n["category"] == EntityCategory.DYNASTY.value for n in dyn_data["nodes"])
    assert dyn_data["node_count"] == 4


def test_graph_export_formats():
    """Verify JSON and GraphML export formats."""
    kg = EpigraphicalKnowledgeGraph()

    # JSON export
    json_str = kg.to_json()
    parsed = json.loads(json_str)
    assert "nodes" in parsed
    assert "edges" in parsed
    assert parsed["node_count"] == len(parsed["nodes"])

    # GraphML export
    graphml_xml = kg.to_graphml()
    assert '<?xml' in graphml_xml or '<graphml' in graphml_xml
    assert 'RULER-RAJARAJA-1' in graphml_xml


def test_graph_html_renderer():
    """Verify HTML template generation for Streamlit embed."""
    kg = EpigraphicalKnowledgeGraph()
    html = render_knowledge_graph_html(kg, height_px=600)

    assert "<!DOCTYPE html>" in html
    assert "vis-network" in html
    assert "RULER-RAJARAJA-1" in html
    assert "Brihadisvara Great Temple" in html
    assert "filterGraph" in html


def test_dynasty_flag_mapping_and_assets():
    """Verify exact local flag assets exist and encode to valid base64 data URIs."""
    from services.knowledge_graph.graph_engine import (
        DYNASTY_FLAG_MAP,
        get_flag_data_uri,
        get_flag_dimensions,
    )
    import os

    expected_dynasties = {"DYN-PALLAVA", "DYN-CHERA", "DYN-CHOLA", "DYN-PANDYA"}
    assert set(DYNASTY_FLAG_MAP.keys()) == expected_dynasties

    for dyn_id, flag_info in DYNASTY_FLAG_MAP.items():
        flag_path = flag_info["flag_path"]
        assert os.path.exists(flag_path), f"Asset missing for {dyn_id}: {flag_path}"

        data_uri = get_flag_data_uri(flag_path)
        assert data_uri is not None, f"Failed to encode flag {flag_path}"
        assert data_uri.startswith("data:image/png;base64,")
        assert len(data_uri) > 1000  # Non-trivial image payload

        # Check metadata attributes
        assert flag_info["emblem_name"]
        assert flag_info["emblem_description"]
        assert "reference emblem" in flag_info["emblem_provenance"].lower()
        assert flag_info["verification_status"] == "Scholarly Reconstruction / Reference Emblem"


def test_flag_aspect_ratios_and_dimensions():
    """Verify original flag dimensions and aspect ratios are accurately computed and preserved."""
    from services.knowledge_graph.graph_engine import get_flag_dimensions

    # Pallava: 600 x 337 (1.78 aspect ratio)
    pal_dims = get_flag_dimensions("assets/knowledge_graph/dynasty_flags/pallava_flag.png")
    assert pal_dims == (600, 337, 1.78)

    # Chera: 700 x 700 (1.0 square aspect ratio)
    cher_dims = get_flag_dimensions("assets/knowledge_graph/dynasty_flags/chera_flag.png")
    assert cher_dims == (700, 700, 1.0)

    # Chola: 600 x 337 (1.78 aspect ratio)
    chol_dims = get_flag_dimensions("assets/knowledge_graph/dynasty_flags/chola_flag.png")
    assert chol_dims == (600, 337, 1.78)

    # Pandya: 600 x 337 (1.78 aspect ratio)
    pan_dims = get_flag_dimensions("assets/knowledge_graph/dynasty_flags/pandya_flag.png")
    assert pan_dims == (600, 337, 1.78)


def test_dynasty_node_attributes_and_provenance():
    """Verify serialized graph nodes contain flag metadata, provenance disclaimer, and data URIs."""
    kg = EpigraphicalKnowledgeGraph()
    data = kg.to_dict()

    dynasties = [n for n in data["nodes"] if n["category"] == EntityCategory.DYNASTY.value]
    assert len(dynasties) == 4

    for dyn in dynasties:
        assert dyn["has_flag_image"] is True
        assert dyn["flag_data_uri"].startswith("data:image/png;base64,")
        assert dyn["emblem_name"]
        assert dyn["emblem_description"]
        assert "reference emblem" in dyn["emblem_provenance"].lower()
        assert dyn["verification_status"] == "Scholarly Reconstruction / Reference Emblem"
        assert dyn["aspect_ratio"] in (1.78, 1.0)


def test_missing_flag_asset_fallback():
    """Verify graceful fallback to diamond shape when flag asset is unavailable."""
    from services.knowledge_graph.graph_engine import get_flag_data_uri, get_flag_dimensions

    assert get_flag_data_uri("assets/non_existent_flag.png") is None
    assert get_flag_dimensions("assets/non_existent_flag.png") is None

    # Custom dynasty with missing asset
    kg = EpigraphicalKnowledgeGraph()
    kg.add_entity(
        "DYN-KALABHRA",
        EntityCategory.DYNASTY.value,
        "Kalabhra Dynasty",
        "களப்பிரர்",
        "Late Classical",
        "c. 3rd – 6th c. CE",
        "Enigmatic dynasty with unknown heraldic standard.",
        "Epigraphia Indica",
        metadata={"flag_path": "assets/non_existent_flag.png"},
    )
    node = [n for n in kg.to_dict()["nodes"] if n["id"] == "DYN-KALABHRA"][0]
    assert node["has_flag_image"] is False
    assert node["flag_data_uri"] is None


def test_graph_relationships_preservation():
    """Verify all 6 core epigraphical relationship patterns are preserved and accessible."""
    kg = EpigraphicalKnowledgeGraph()
    edges = kg.to_dict()["edges"]

    # 1. Dynasty -> Ruler
    dyn_ruler = [e for e in edges if e["source"] == "DYN-CHOLA" and e["target"] == "RULER-RAJARAJA-1"]
    assert len(dyn_ruler) >= 1
    assert dyn_ruler[0]["label"] == "DYNASTIC_RULER"

    # 2. Dynasty -> Inscription
    dyn_insc = [e for e in edges if e["source"] == "DYN-CHOLA" and e["target"] == "KV-INSC-002"]
    assert len(dyn_insc) >= 1
    assert dyn_insc[0]["label"] == "ASSOCIATED_INSCRIPTION"

    # 3. Dynasty -> Temple
    dyn_temple = [e for e in edges if e["source"] == "DYN-CHOLA" and e["target"] == "SITE-BRIHADISVARA"]
    assert len(dyn_temple) >= 1
    assert dyn_temple[0]["label"] == "IMPERIAL_TEMPLE"

    # 4. Ruler -> Inscription
    ruler_insc = [e for e in edges if e["source"] == "RULER-RAJARAJA-1" and e["target"] == "KV-INSC-002"]
    assert len(ruler_insc) >= 1
    assert ruler_insc[0]["label"] == "COMMISSIONED_INSCRIPTION"

    # 5. Inscription -> Historical Event
    insc_event = [e for e in edges if e["source"] == "KV-INSC-003" and e["target"] == "EVENT-KUDAVOLAI-REFORM"]
    assert len(insc_event) >= 1
    assert insc_event[0]["label"] == "RECORDS_HISTORICAL_EVENT"

    # 6. Donor -> Land Grant
    donor_grant = [e for e in edges if e["source"] == "DONOR-KUNDAVAI" and e["target"] == "GRANT-DEVADANA-KUNDAVAI"]
    assert len(donor_grant) >= 1
    assert donor_grant[0]["label"] == "COMMISSIONED_LAND_GRANT"

    # Verify reciprocal relationships
    ruler_dyn = [e for e in edges if e["source"] == "RULER-RAJARAJA-1" and e["target"] == "DYN-CHOLA"]
    assert len(ruler_dyn) >= 1
    assert ruler_dyn[0]["label"] == "BELONGS_TO_DYNASTY"


def test_non_dynasty_nodes_preserved():
    """Verify non-dynasty nodes retain standard shapes (box for inscriptions, dot for rulers)."""
    kg = EpigraphicalKnowledgeGraph()
    data = kg.to_dict()

    insc_nodes = [n for n in data["nodes"] if n["category"] == EntityCategory.INSCRIPTION.value]
    assert len(insc_nodes) >= 6
    for insc in insc_nodes:
        assert "has_flag_image" not in insc

    ruler_nodes = [n for n in data["nodes"] if n["category"] == EntityCategory.RULER.value]
    assert len(ruler_nodes) >= 6

    event_nodes = [n for n in data["nodes"] if n["category"] == EntityCategory.HISTORICAL_EVENT.value]
    assert len(event_nodes) >= 2

    grant_nodes = [n for n in data["nodes"] if n["category"] == EntityCategory.GRANT.value]
    assert len(grant_nodes) >= 2


def test_html_renderer_visjs_flag_node_configuration():
    """Verify HTML renderer generates compliant Vis.js image nodes and contain styling."""
    kg = EpigraphicalKnowledgeGraph()
    html = render_knowledge_graph_html(kg, height_px=650)

    # Vis.js image node mechanisms
    assert "shapeProperties" in html
    assert "useBorderWithImage: true" in html
    assert "useImageSize: false" in html
    assert "interpolation: true" in html
    assert "isDynastyWithFlag" in html
    assert "shape = 'image'" in html

    # Dark background compatible border
    assert "#C5A15A" in html

    # Hover tooltip and click inspection
    assert "flag-preview-card" in html
    assert "Provenance Note" in html
    assert "Associated Rulers" in html
    assert "Associated Inscription Records" in html
    assert "Associated Temples" in html
    assert "inspectNodeById" in html

    # Zoom and navigation controls
    assert "navigationButtons: true" in html
    assert "keyboard: true" in html
    assert "network.fit({animation: true})" in html


def test_empty_graph_handling():
    """Verify empty graph validation and error reporting without crashing renderer."""
    empty_kg = EpigraphicalKnowledgeGraph.__new__(EpigraphicalKnowledgeGraph)
    empty_kg.graph = nx.MultiDiGraph()
    val = empty_kg.validate()
    assert val["is_valid"] is False
    assert val["node_count"] == 0
    assert "empty graph" in val["errors"][0].lower()

    # HTML renderer handles empty graph gracefully
    html = render_knowledge_graph_html(empty_kg, height_px=500)
    assert "<!DOCTYPE html>" in html
    assert "0 Entities | 0 Verified Relations" in html


def test_single_node_graph():
    """Verify single-node graph initialization and rendering."""
    single_kg = EpigraphicalKnowledgeGraph.__new__(EpigraphicalKnowledgeGraph)
    single_kg.graph = nx.MultiDiGraph()
    single_kg.add_entity(
        "RULER-SOLITARY",
        EntityCategory.RULER.value,
        "Solitary King",
        "தனி மன்னர்",
        "Sangam",
        "1st c. CE",
        "Isolated ruler entity.",
        "Test Citation",
    )
    val = single_kg.validate()
    assert val["is_valid"] is True
    assert val["node_count"] == 1
    assert val["edge_count"] == 0

    html = render_knowledge_graph_html(single_kg)
    assert "RULER-SOLITARY" in html
    assert "1 Entities | 0 Verified Relations" in html


def test_disconnected_nodes_handling():
    """Verify graph with disconnected island components."""
    kg = EpigraphicalKnowledgeGraph()
    # Add an isolated island node
    kg.add_entity(
        "ISLAND-SITE",
        EntityCategory.LOCATION.value,
        "Isolated Cavern",
        "தனித்த குகை",
        "Prehistoric",
        "c. 1000 BCE",
        "Unlinked prehistoric rock cavern.",
        "Archaeological Survey of India",
    )
    val = kg.validate()
    assert val["is_valid"] is True
    assert val["node_count"] == 30

    # Path to connected node should be empty
    path = kg.find_path("ISLAND-SITE", "SITE-BRIHADISVARA")
    assert path == []


def test_invalid_edge_references_validation():
    """Verify validation detects and isolates edges pointing to missing nodes."""
    kg = EpigraphicalKnowledgeGraph()
    # Force inject invalid edges directly into underlying graph
    kg.graph.add_edge("NON-EXISTENT-SRC", "RULER-RAJARAJA-1", relation_type="PHANTOM_EDGE", source_citation="None")
    kg.graph.add_edge("RULER-RAJARAJA-1", "NON-EXISTENT-TGT", relation_type="PHANTOM_EDGE", source_citation="None")

    val = kg.validate()
    assert val["is_valid"] is False
    assert len(val["invalid_edges"]) == 2
    assert any(e["reason"] == "missing_source" for e in val["invalid_edges"])
    assert any(e["reason"] == "missing_target" for e in val["invalid_edges"])

    # Ensure HTML renderer still produces valid output and filters invalid edges safely
    html = render_knowledge_graph_html(kg)
    assert "<!DOCTYPE html>" in html
    assert "validEdges" in html


def test_all_four_dynasty_flags_and_fallbacks():
    """Verify all 4 dynasty nodes contain both primary flag data URIs and fallback SVG data URIs."""
    kg = EpigraphicalKnowledgeGraph()
    data = kg.to_dict()
    dynasties = [n for n in data["nodes"] if n["category"] == EntityCategory.DYNASTY.value]
    assert len(dynasties) == 4

    expected_ids = {"DYN-PALLAVA", "DYN-CHERA", "DYN-CHOLA", "DYN-PANDYA"}
    found_ids = {d["id"] for d in dynasties}
    assert found_ids == expected_ids

    for d in dynasties:
        assert d["has_flag_image"] is True
        assert d["flag_data_uri"].startswith("data:image/")
        assert d["fallback_svg_data_uri"].startswith("data:image/svg+xml;base64,")
        assert d["aspect_ratio"] in (1.78, 1.0)
        assert d["image_width"] in (600, 700)


def test_canvas_dimensions_and_resilience_in_html():
    """Verify explicit canvas layout dimensions, ResizeObserver, and fallback engine in HTML."""
    kg = EpigraphicalKnowledgeGraph()
    html = render_knowledge_graph_html(kg, height_px=650)

    # Container sizing guarantees
    assert "min-height: 520px" in html
    assert "height: calc(100% - 52px)" in html
    assert "ResizeObserver" in html
    assert "FallbackCanvasGraphEngine" in html
    assert "fallback-canvas" in html
    assert "assignInitialCoordinates" in html
    assert "network.setSize('100%', '100%')" in html


def test_filtering_and_inspection_capabilities_in_html():
    """Verify category filtering, search input, inspection, fitView and physics controls."""
    kg = EpigraphicalKnowledgeGraph()
    html = render_knowledge_graph_html(kg, height_px=650)

    assert 'id="categoryFilter"' in html
    assert 'id="searchInput"' in html
    assert "fitView()" in html
    assert "togglePhysics()" in html
    assert "showEntityDetails" in html
    assert "selected-badge" in html
    assert "engine-badge" in html



