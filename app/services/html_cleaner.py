"""
HTML Cleaner Service
Converts email HTML to clean, readable text.
Removes tracking pixels, normalizes formatting, extracts article content.
"""

import re
from typing import Optional
from bs4 import BeautifulSoup
import html2text


class HTMLCleaner:
    """
    Cleans HTML email content for storage and processing.
    Optimized for newsletter-style content like Stratechery.
    """
    
    def __init__(self):
        # Configure html2text for clean output
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False  # Keep links but format them nicely
        self.h2t.ignore_images = True  # Remove image references
        self.h2t.ignore_emphasis = False  # Keep bold/italic markers
        self.h2t.body_width = 0  # Don't wrap lines
        self.h2t.skip_internal_links = True
        self.h2t.inline_links = False  # Put links at end of text
        self.h2t.protect_links = True
        
        # Tracking pixel patterns (common in newsletters)
        self.tracking_patterns = [
            r'<img[^>]*(?:width|height)\s*=\s*["\']?1["\']?[^>]*>',  # 1x1 pixels
            r'<img[^>]*(?:tracking|pixel|beacon|open|click)[^>]*>',  # Named trackers
            r'<img[^>]*src=["\'][^"\']*(?:track|pixel|beacon|open|click|mailchimp|sendgrid|amazonses)[^"\']*["\'][^>]*>',
        ]
        
        # Elements to remove entirely
        self.remove_selectors = [
            'script', 'style', 'head', 'meta', 'link',
            'noscript', 'iframe', 'object', 'embed',
            '[style*="display:none"]', '[style*="display: none"]',
            '.unsubscribe', '.footer', '.email-footer',
            '[data-tracking]', '.tracking-pixel',
        ]
    
    def clean(self, html_content: str) -> dict:
        """
        Clean HTML and return both cleaned HTML and plain text.
        
        Returns:
            dict with 'html', 'text', and 'metadata' keys
        """
        if not html_content:
            return {
                'html': '',
                'text': '',
                'metadata': {}
            }
        
        # Step 1: Remove tracking pixels using regex (before parsing)
        cleaned_html = self._remove_tracking_pixels(html_content)
        
        # Step 2: Parse with BeautifulSoup
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        
        # Step 3: Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Step 4: Extract any metadata we can find
        metadata = self._extract_metadata(soup)
        
        # Step 5: Get the main content
        main_content = self._extract_main_content(soup)
        
        # Step 6: Convert to clean text
        clean_text = self._html_to_text(str(main_content))
        
        # Step 7: Post-process text
        clean_text = self._post_process_text(clean_text)
        
        return {
            'html': str(main_content),
            'text': clean_text,
            'metadata': metadata
        }
    
    def to_text(self, html_content: str) -> str:
        """Convenience method to just get clean text."""
        result = self.clean(html_content)
        return result['text']
    
    def _remove_tracking_pixels(self, html: str) -> str:
        """Remove tracking pixels using regex patterns."""
        result = html
        for pattern in self.tracking_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        return result
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove script, style, and other unwanted elements."""
        for selector in self.remove_selectors:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                # Some selectors might fail on malformed HTML
                pass
        
        # Remove empty elements
        for element in soup.find_all():
            try:
                if element.name not in ['br', 'hr', 'img'] and not element.get_text(strip=True):
                    # Keep elements with important attributes (like anchors)
                    if not element.get('href') and not element.get('id'):
                        element.decompose()
            except Exception:
                pass
    
    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract any metadata from the HTML."""
        metadata = {}
        
        # Try to find canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']
        
        # Try to find article date
        time_elem = soup.find('time')
        if time_elem:
            metadata['date'] = time_elem.get('datetime') or time_elem.get_text(strip=True)
        
        # Look for Stratechery-specific patterns
        # The canonical URL often contains the article slug
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'stratechery.com' in href and '/article/' in href:
                metadata['canonical_url'] = href
                break
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Try to extract just the main article content.
        Falls back to full body if no article container found.
        """
        # Common article container selectors
        article_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '[role="main"]',
            '.email-body',
            '.message-body',
        ]
        
        for selector in article_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                return content
        
        # Fall back to body or entire soup
        body = soup.find('body')
        return body if body else soup
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to markdown-ish plain text."""
        return self.h2t.handle(html)
    
    def _post_process_text(self, text: str) -> str:
        """Clean up the converted text."""
        # Remove excessive blank lines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove common newsletter footer patterns
        footer_patterns = [
            r'(?i)unsubscribe.*$',
            r'(?i)view in browser.*$',
            r'(?i)©\s*\d{4}.*$',
            r'(?i)all rights reserved.*$',
            r'(?i)update your preferences.*$',
            r'(?i)sent to:.*$',
        ]
        
        for pattern in footer_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Final cleanup
        text = text.strip()
        
        return text


class StratecheryExtractor(HTMLCleaner):
    """
    Specialized extractor for Stratechery newsletter format.
    Inherits from HTMLCleaner with Stratechery-specific optimizations.
    """
    
    def __init__(self):
        super().__init__()
        
        # Stratechery-specific patterns to remove
        self.stratechery_noise = [
            r'This update is for paying subscribers only',
            r'Subscribe to Stratechery',
            r'Already a subscriber\? Sign in',
        ]
    
    def clean(self, html_content: str) -> dict:
        """Clean Stratechery HTML with additional processing."""
        result = super().clean(html_content)
        
        # Remove Stratechery-specific noise from text
        text = result['text']
        for pattern in self.stratechery_noise:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        result['text'] = text.strip()
        
        # Try to extract article title from content
        if not result['metadata'].get('title'):
            first_line = result['text'].split('\n')[0].strip()
            if first_line and len(first_line) < 200:
                result['metadata']['title'] = first_line
        
        return result


# Convenience functions
def clean_html(html_content: str) -> str:
    """Quick helper to get clean text from HTML."""
    cleaner = HTMLCleaner()
    return cleaner.to_text(html_content)


def clean_stratechery(html_content: str) -> dict:
    """Clean Stratechery newsletter HTML."""
    extractor = StratecheryExtractor()
    return extractor.clean(html_content)
