"""
Full-text paper retrieval module.
Tries multiple sources in order: PMC Open Access API → Unpaywall → mark inaccessible.
"""

import io
import time
import xml.etree.ElementTree as ET

import pdfplumber
import requests

import config


class PaperFetcher:
    """Fetches full-text content of scientific papers from legal open-access sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "nAChR-VEP-DataScraper/1.0 (mailto:{})".format(
                config.NCBI_EMAIL
            )
        })

    def fetch_full_text(self, pmid: str, pmcid: str, doi: str) -> tuple[str | None, str]:
        """
        Attempt to retrieve full-text content for a paper.

        Args:
            pmid: PubMed ID
            pmcid: PubMed Central ID (may be empty)
            doi: Digital Object Identifier (may be empty)

        Returns:
            Tuple of (text_content, source) where source is one of:
            "pmc", "unpaywall_html", "unpaywall_pdf", or None if inaccessible.
            If None, second element is the reason it failed.
        """
        # Try PMC first
        if pmcid:
            text = self._fetch_from_pmc(pmcid)
            if text:
                return text, "pmc"

        # Try Unpaywall
        if doi:
            text, source = self._fetch_from_unpaywall(doi)
            if text:
                return text, source

        # All sources failed
        reason = "No open-access full text available"
        if not pmcid and not doi:
            reason = "No PMCID or DOI available"
        elif not pmcid:
            reason = "No PMCID; Unpaywall had no open-access version"
        return None, reason

    def _fetch_from_pmc(self, pmcid: str) -> str | None:
        """
        Fetch full text from PubMed Central Open Access API.
        Returns plain text extracted from the XML, or None on failure.
        """
        # Normalize PMCID format
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "xml",
            "retmode": "xml",
        }
        if config.NCBI_EMAIL:
            params["email"] = config.NCBI_EMAIL
        if config.NCBI_API_KEY:
            params["api_key"] = config.NCBI_API_KEY

        try:
            resp = self._request_with_retry(url, params=params)
            if resp is None or resp.status_code != 200:
                return None

            # Parse XML and extract text content
            return self._parse_pmc_xml(resp.text)
        except Exception as e:
            print(f"    PMC fetch error for {pmcid}: {e}")
            return None

    def _parse_pmc_xml(self, xml_text: str) -> str | None:
        """Extract readable text from PMC XML format."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        # Find the article body
        body = root.find(".//body")
        if body is None:
            # Some articles have the text in different locations
            body = root.find(".//article")
            if body is None:
                return None

        # Extract all text recursively
        text_parts = []

        # Get title
        title = root.find(".//article-title")
        if title is not None:
            text_parts.append("TITLE: " + self._get_element_text(title))

        # Get abstract
        abstract = root.find(".//abstract")
        if abstract is not None:
            text_parts.append("\nABSTRACT:\n" + self._get_element_text(abstract))

        # Get body text
        body_elem = root.find(".//body")
        if body_elem is not None:
            text_parts.append("\nBODY:\n" + self._get_element_text(body_elem))

        # Get tables (often contain mutation data)
        for table_wrap in root.findall(".//table-wrap"):
            label = table_wrap.find("label")
            caption = table_wrap.find("caption")
            label_text = self._get_element_text(label) if label is not None else ""
            caption_text = self._get_element_text(caption) if caption is not None else ""
            table_text = self._get_element_text(table_wrap)
            text_parts.append(f"\nTABLE {label_text}: {caption_text}\n{table_text}")

        full_text = "\n".join(text_parts)
        return full_text if len(full_text) > 100 else None

    def _get_element_text(self, element) -> str:
        """Recursively extract all text from an XML element."""
        return "".join(element.itertext()).strip()

    def _fetch_from_unpaywall(self, doi: str) -> tuple[str | None, str]:
        """
        Try to fetch full text via Unpaywall API.
        Returns (text, source_type) or (None, "").
        """
        email = config.UNPAYWALL_EMAIL or config.NCBI_EMAIL
        if not email:
            return None, ""

        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": email}

        try:
            resp = self._request_with_retry(url, params=params)
            if resp is None or resp.status_code != 200:
                return None, ""

            data = resp.json()
            best_oa = data.get("best_oa_location")
            if not best_oa:
                return None, ""

            # Try to get the full text URL
            pdf_url = best_oa.get("url_for_pdf")
            html_url = best_oa.get("url_for_landing_page")

            # Prefer PDF as it's more reliable for text extraction
            if pdf_url:
                text = self._download_and_extract_pdf(pdf_url)
                if text:
                    return text, "unpaywall_pdf"

            # Try HTML
            if html_url:
                text = self._download_html_text(html_url)
                if text:
                    return text, "unpaywall_html"

        except Exception as e:
            print(f"    Unpaywall error for {doi}: {e}")

        return None, ""

    def _download_and_extract_pdf(self, url: str) -> str | None:
        """Download a PDF and extract text content."""
        try:
            resp = self._request_with_retry(url, timeout=60)
            if resp is None or resp.status_code != 200:
                return None

            # Check if we actually got a PDF
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                return None

            # Extract text from PDF
            text_parts = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                    # Also extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text_parts.append(
                                    " | ".join(str(cell) if cell else "" for cell in row)
                                )

            full_text = "\n".join(text_parts)
            return full_text if len(full_text) > 100 else None

        except Exception as e:
            print(f"    PDF extraction error: {e}")
            return None

    def _download_html_text(self, url: str) -> str | None:
        """Download HTML page and extract main text content."""
        try:
            resp = self._request_with_retry(url, timeout=30)
            if resp is None or resp.status_code != 200:
                return None

            # Basic HTML text extraction using built-in xml parser
            # Remove script and style tags, then get text
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip = False
                    self.skip_tags = {"script", "style", "nav", "header", "footer"}

                def handle_starttag(self, tag, attrs):
                    if tag in self.skip_tags:
                        self.skip = True

                def handle_endtag(self, tag):
                    if tag in self.skip_tags:
                        self.skip = False

                def handle_data(self, data):
                    if not self.skip:
                        stripped = data.strip()
                        if stripped:
                            self.text_parts.append(stripped)

            parser = TextExtractor()
            parser.feed(resp.text)
            full_text = "\n".join(parser.text_parts)
            return full_text if len(full_text) > 200 else None

        except Exception as e:
            print(f"    HTML extraction error: {e}")
            return None

    def _request_with_retry(
        self, url: str, params: dict = None, timeout: int = 30
    ) -> requests.Response | None:
        """Make an HTTP GET request with retry logic."""
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                if resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = config.RETRY_BACKOFF_FACTOR ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return resp
            except requests.RequestException as e:
                if attempt < config.MAX_RETRIES - 1:
                    wait_time = config.RETRY_BACKOFF_FACTOR ** (attempt + 1)
                    print(f"    Request error ({e}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    Request failed after {config.MAX_RETRIES} attempts: {e}")
                    return None
        return None
