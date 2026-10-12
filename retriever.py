"""
Biomedical RAG Retriever – fetches evidence from PubMed via NCBI E-utilities.
Provides BioASQ-style passage retrieval for the medical evidence agent.
"""
import requests
import xml.etree.ElementTree as ET
import logging
import re
from dataclasses import dataclass, field
from config import PUBMED_MAX_RESULTS, PUBMED_EMAIL

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class EvidencePassage:
    """A single evidence passage with metadata."""
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        auth = self.authors[0] if self.authors else "Unknown"
        return f"{auth} et al., {self.journal} ({self.year}) [PMID: {self.pmid}]"

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.abstract}"


class BiomedicalRetriever:
    """Retrieves biomedical evidence from PubMed for RAG."""

    def __init__(self, max_results: int = PUBMED_MAX_RESULTS, email: str = PUBMED_EMAIL):
        self.max_results = max_results
        self.email = email

    def _search_pmids(self, query: str) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "email": self.email,
            "sort": "relevance",
        }
        try:
            resp = requests.get(ESEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info("PubMed search for '%s' returned %d PMIDs", query, len(pmids))
            return pmids
        except Exception as e:
            logger.error("PubMed search error: %s", e)
            return []

    def _fetch_articles(self, pmids: list[str]) -> list[EvidencePassage]:
        """Fetch article details for a list of PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        try:
            resp = requests.get(EFETCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.error("PubMed fetch error: %s", e)
            return []

    def _parse_xml(self, xml_text: str) -> list[EvidencePassage]:
        """Parse PubMed XML response into EvidencePassage objects."""
        passages = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else "Unknown"

                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None else ""

                # Combine all AbstractText sections
                abstract_parts = []
                for abs_el in article.findall(".//AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else ""

                year_el = article.find(".//PubDate/Year")
                if year_el is None:
                    year_el = article.find(".//PubDate/MedlineDate")
                year = year_el.text[:4] if year_el is not None and year_el.text else ""

                authors = []
                for author in article.findall(".//Author"):
                    last = author.find("LastName")
                    if last is not None and last.text:
                        authors.append(last.text)

                if abstract:  # Only include articles with abstracts
                    passages.append(EvidencePassage(
                        pmid=pmid, title=title or "", abstract=abstract,
                        journal=journal, year=year, authors=authors,
                    ))
        except ET.ParseError as e:
            logger.error("XML parse error: %s", e)

        return passages

    def retrieve(self, question: str) -> list[EvidencePassage]:
        """Main entry: retrieve relevant biomedical evidence for a question."""
        # Clean the query for PubMed search
        query = self._build_search_query(question)
        pmids = self._search_pmids(query)
        passages = self._fetch_articles(pmids)
        logger.info("Retrieved %d evidence passages for: %s", len(passages), question)
        return passages

    def _build_search_query(self, question: str) -> str:
        """Build a PubMed-optimized search query from a natural language question."""
        # Remove common question words and add PubMed filters
        stop_patterns = [
            r"\b(what|how|why|when|where|which|who|does|do|is|are|can|could|should|would)\b",
            r"\?", r"\.",
        ]
        query = question
        for pattern in stop_patterns:
            query = re.sub(pattern, " ", query, flags=re.IGNORECASE)
        query = re.sub(r"\s+", " ", query).strip()
        return query
