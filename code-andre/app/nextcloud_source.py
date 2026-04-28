import re
import requests
from typing import List, Tuple
from urllib.parse import urlparse, unquote
import xml.etree.ElementTree as ET
from .config import settings

DAV_NS = {
    "d": "DAV:",
    "oc": "http://owncloud.org/ns",
    "nc": "http://nextcloud.org/ns"
}

def _extract_host_and_token(share_url: str) -> Tuple[str, str]:
    """
    From https://drive.upm.es/s/<TOKEN> -> ("https://drive.upm.es", "<TOKEN>")
    """
    u = urlparse(share_url)
    m = re.search(r"/s/([^/]+)", u.path)
    if not m:
        raise ValueError(f"Could not extract token from SOURCE_URL: {share_url}")
    host = f"{u.scheme}://{u.netloc}"
    token = m.group(1)
    return host, token

def _webdav_candidates(host: str, token: str) -> List[str]:
    # Nextcloud older: /public.php/webdav/
    # Nextcloud newer: /public.php/dav/files/{share_token}
    return [
        f"{host}/public.php/dav/files/{token}",
        f"{host}/public.php/webdav/",
    ]

def list_pdfs() -> List[Tuple[str, str]]:
    """
    Returns list of (filename, download_url)
    """
    host, token = _extract_host_and_token(settings.SOURCE_URL)
    password = settings.SOURCE_PASSWORD or ""

    # PROPFIND Depth:1
    body = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:displayname/>
    <d:resourcetype/>
    <d:getcontenttype/>
    <d:getcontentlength/>
  </d:prop>
</d:propfind>
"""

    headers = {"Depth": "1", "Content-Type": "application/xml"}

    last_err = None
    for base in _webdav_candidates(host, token):
        try:
            r = requests.request(
                "PROPFIND",
                base,
                headers=headers,
                data=body,
                auth=(token if "dav/files" in base else token, password),
                timeout=120,
            )
            if r.status_code in (401, 403):
                # Some servers behave weird if password is blank.
                # User can set SOURCE_PASSWORD if needed.
                raise RuntimeError(f"WebDAV auth rejected ({r.status_code}). Set SOURCE_PASSWORD if share is protected.")
            r.raise_for_status()

            # Parse XML
            root = ET.fromstring(r.text)
            pdfs = []
            for resp in root.findall("d:response", DAV_NS):
                href_el = resp.find("d:href", DAV_NS)
                if href_el is None:
                    continue
                href = unquote(href_el.text or "")
                # Skip the folder itself
                if href.endswith("/") and href.rstrip("/").endswith(token):
                    continue

                propstat = resp.find("d:propstat", DAV_NS)
                if propstat is None:
                    continue
                prop = propstat.find("d:prop", DAV_NS)
                if prop is None:
                    continue

                displayname = prop.find("d:displayname", DAV_NS)
                name = (displayname.text if displayname is not None else "").strip()
                if not name.lower().endswith(".pdf"):
                    continue

                # Build download URL:
                # For /public.php/dav/files/{token}/...  -> host + href
                # For /public.php/webdav/... -> host + href
                download_url = f"{host}{href}"
                pdfs.append((name, download_url))

            return pdfs
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed listing PDFs via WebDAV. Last error: {last_err}")

def download_pdf(download_url: str) -> bytes:
    host, token = _extract_host_and_token(settings.SOURCE_URL)
    password = settings.SOURCE_PASSWORD or ""
    r = requests.get(download_url, auth=(token, password), timeout=300)
    r.raise_for_status()
    return r.content