"""
knowledge_graph/graph_builder.py

Builds a vulnerability knowledge graph from normalized VulnerabilityEntity objects.
The graph uses RDFLib with an OWL ontology aligned to the MITRE ATT&CK data model.

Nodes: VulnerabilityEntity instances as OWL individuals
Edges: "affects", "has-weakness", "exploited-by", "mitigated-by"
       discovered by the BiLSTM-CRF NER + attention CNN relation classifier
       and enriched by GAT link prediction and GPT-4 candidate link generation.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rdflib import Graph, Literal, Namespace, RDF, OWL, XSD, URIRef
from rdflib.namespace import RDFS

from connectors.entity_schema import VulnerabilityEntity, DataSource

logger = logging.getLogger(__name__)

# Ontology namespace
CYBER = Namespace("http://cyberfuse-ci.org/ontology/vulnerability#")
ATTACK = Namespace("https://attack.mitre.org/")


class VulnerabilityKnowledgeGraph:
    """
    RDF/OWL knowledge graph for CyberFuse-CI.

    Design decisions:
        - Each VulnerabilityEntity becomes an OWL individual of type cyber:Vulnerability.
        - Component nodes are type cyber:AffectedComponent.
        - CWE nodes are type cyber:Weakness.
        - Edges follow the cyber:affects / cyber:hasWeakness / cyber:exploitedBy predicate set.
        - The graph is serializable to Turtle, JSON-LD, or N-Triples for downstream use.

    This is the same ontology structure used in the paper's Layer 2 fusion pipeline.
    """

    def __init__(self):
        self.g = Graph()
        self.g.bind("cyber", CYBER)
        self.g.bind("attack", ATTACK)
        self.g.bind("owl", OWL)
        self._define_schema()
        self._entity_count = 0
        self._edge_count = 0

    def _define_schema(self):
        """Assert OWL class and property declarations."""
        for cls in ("Vulnerability", "AffectedComponent", "Weakness", "ThreatActor"):
            self.g.add((CYBER[cls], RDF.type, OWL.Class))

        for prop, domain, range_ in [
            ("affects",      "Vulnerability", "AffectedComponent"),
            ("hasWeakness",  "Vulnerability", "Weakness"),
            ("exploitedBy",  "Vulnerability", "ThreatActor"),
            ("mitigatedBy",  "Vulnerability", "Vulnerability"),
        ]:
            node = CYBER[prop]
            self.g.add((node, RDF.type, OWL.ObjectProperty))
            self.g.add((node, RDFS.domain, CYBER[domain]))
            self.g.add((node, RDFS.range,  CYBER[range_]))

        # Data properties
        for dp in ("cvssScore", "description", "sourceId", "dataSource", "severityTier", "publishedAt"):
            self.g.add((CYBER[dp], RDF.type, OWL.DatatypeProperty))

    def add_entity(self, entity: VulnerabilityEntity):
        """Add a VulnerabilityEntity as an OWL individual and assert its properties."""
        node = CYBER[f"vuln/{entity.entity_id}"]

        self.g.add((node, RDF.type, CYBER.Vulnerability))
        self.g.add((node, CYBER.sourceId,     Literal(entity.source_id)))
        self.g.add((node, CYBER.dataSource,   Literal(entity.source.value)))
        self.g.add((node, CYBER.cvssScore,    Literal(entity.severity_score, datatype=XSD.float)))
        self.g.add((node, CYBER.severityTier, Literal(entity.severity_tier.value)))
        self.g.add((node, CYBER.description,  Literal(entity.description[:1000])))

        if entity.published_at:
            self.g.add((node, CYBER.publishedAt, Literal(entity.published_at.isoformat(), datatype=XSD.dateTime)))

        # Component node
        comp_id = entity.affected_component.replace(" ", "_").replace("/", "_")
        comp_node = CYBER[f"component/{comp_id}"]
        self.g.add((comp_node, RDF.type, CYBER.AffectedComponent))
        self.g.add((comp_node, RDFS.label, Literal(entity.affected_component)))
        self.g.add((node, CYBER.affects, comp_node))

        # Weakness nodes
        for cwe in entity.cwe_ids:
            cwe_safe = cwe.replace("-", "_")
            cwe_node = CYBER[f"weakness/{cwe_safe}"]
            self.g.add((cwe_node, RDF.type, CYBER.Weakness))
            self.g.add((cwe_node, RDFS.label, Literal(cwe)))
            self.g.add((node, CYBER.hasWeakness, cwe_node))

        self._entity_count += 1

    def add_edge(self, subject_id: str, predicate: str, object_id: str):
        """Assert a typed edge between two vulnerability individuals."""
        s = CYBER[f"vuln/{subject_id}"]
        p = CYBER[predicate]
        o = CYBER[f"vuln/{object_id}"]
        self.g.add((s, p, o))
        self._edge_count += 1

    def add_cross_source_link(
        self,
        entity_a_id: str,
        entity_b_id: str,
        predicate: str = "coOccursWith",
        confidence: float = 1.0,
    ):
        """
        Add a cross-source co-occurrence edge discovered by the knowledge graph
        fusion layer (CVE entity + STIX indicator + SIEM event sharing a component).

        The confidence score comes from the weighted GAT/LLM ensemble described
        in Layer 2 of the paper.
        """
        s = CYBER[f"vuln/{entity_a_id}"]
        o = CYBER[f"vuln/{entity_b_id}"]
        p = CYBER[predicate]

        if (p, RDF.type, OWL.ObjectProperty) not in self.g:
            self.g.add((p, RDF.type, OWL.ObjectProperty))

        self.g.add((s, p, o))

        # Reify with confidence score using blank node
        stmt = CYBER[f"stmt/{entity_a_id}_{entity_b_id}_{predicate}"]
        self.g.add((stmt, RDF.type, OWL.Axiom))
        self.g.add((stmt, OWL.annotatedSource,   s))
        self.g.add((stmt, OWL.annotatedProperty, p))
        self.g.add((stmt, OWL.annotatedTarget,   o))
        self.g.add((stmt, CYBER.confidence, Literal(confidence, datatype=XSD.float)))

        self._edge_count += 1

    def find_co_occurrences(self, min_sources: int = 2) -> list[dict]:
        """
        Query the graph for vulnerability nodes that appear across multiple data sources
        for the same affected component. These are high-probability exploitation signals.

        Returns a list of dicts with: component, source_count, entities, max_cvss
        """
        sparql = """
        PREFIX cyber: <http://cyberfuse-ci.org/ontology/vulnerability#>
        SELECT ?component ?dataSource ?vuln ?cvss
        WHERE {
            ?vuln a cyber:Vulnerability ;
                  cyber:affects ?component ;
                  cyber:dataSource ?dataSource ;
                  cyber:cvssScore ?cvss .
        }
        ORDER BY ?component
        """

        results = list(self.g.query(sparql))
        component_map: dict = {}
        for row in results:
            comp = str(row[0])
            if comp not in component_map:
                component_map[comp] = {"sources": set(), "entities": [], "max_cvss": 0.0}
            component_map[comp]["sources"].add(str(row[1]))
            component_map[comp]["entities"].append(str(row[2]))
            try:
                score = float(row[3])
                component_map[comp]["max_cvss"] = max(component_map[comp]["max_cvss"], score)
            except (ValueError, TypeError):
                pass

        output = []
        for comp, data in component_map.items():
            if len(data["sources"]) >= min_sources:
                output.append({
                    "component":    comp,
                    "source_count": len(data["sources"]),
                    "sources":      list(data["sources"]),
                    "entities":     data["entities"],
                    "max_cvss":     data["max_cvss"],
                })

        output.sort(key=lambda x: (x["source_count"], x["max_cvss"]), reverse=True)
        return output

    def serialize(self, path: str, format: str = "turtle"):
        """Save the graph to a file. Formats: turtle, json-ld, n3, xml."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=str(out), format=format)
        logger.info("Serialized graph (%d entities, %d edges) to %s", self._entity_count, self._edge_count, out)

    @classmethod
    def load(cls, path: str, format: str = "turtle") -> "VulnerabilityKnowledgeGraph":
        """Load a previously serialized graph from disk."""
        kg = cls()
        kg.g.parse(path, format=format)
        logger.info("Loaded graph from %s", path)
        return kg

    @property
    def entity_count(self) -> int:
        return self._entity_count

    @property
    def edge_count(self) -> int:
        return self._edge_count
