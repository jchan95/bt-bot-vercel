"""
Email Parser Service
Extracts metadata and content from .eml files and raw email data.
"""

import email
from email import policy
from email.parser import BytesParser, Parser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re


@dataclass
class ParsedEmail:
    """Structured representation of a parsed email."""
    subject: str
    sender: str
    recipient: str
    date: datetime
    html_body: Optional[str]
    text_body: Optional[str]
    message_id: str
    content_hash: str
    raw_headers: dict
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "sender": self.sender,
            "recipient": self.recipient,
            "date": self.date.isoformat() if self.date else None,
            "html_body": self.html_body,
            "text_body": self.text_body,
            "message_id": self.message_id,
            "content_hash": self.content_hash,
        }


class EmailParser:
    """
    Parses email files (.eml) and raw email content.
    Handles multipart MIME messages and extracts both HTML and plain text bodies.
    """
    
    def __init__(self):
        self.policy = policy.default
    
    def parse_eml_file(self, file_path: str) -> ParsedEmail:
        """Parse an .eml file from disk."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Email file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=self.policy).parse(f)
        
        return self._extract_email_data(msg)
    
    def parse_raw_bytes(self, raw_bytes: bytes) -> ParsedEmail:
        """Parse raw email bytes (for API uploads)."""
        msg = BytesParser(policy=self.policy).parsebytes(raw_bytes)
        return self._extract_email_data(msg)
    
    def parse_raw_string(self, raw_string: str) -> ParsedEmail:
        """Parse raw email string."""
        msg = Parser(policy=self.policy).parsestr(raw_string)
        return self._extract_email_data(msg)
    
    def _extract_email_data(self, msg: email.message.EmailMessage) -> ParsedEmail:
        """Extract all relevant data from an email message object."""
        
        # Extract headers
        subject = self._clean_subject(msg.get('Subject', ''))
        sender = msg.get('From', '')
        recipient = msg.get('To', '')
        message_id = msg.get('Message-ID', '')
        
        # Parse date
        date = self._parse_date(msg.get('Date', ''))
        
        # Extract bodies
        html_body, text_body = self._extract_bodies(msg)
        
        # Generate content hash for deduplication
        content_for_hash = (html_body or text_body or subject)
        content_hash = self._generate_hash(content_for_hash)
        
        # Capture raw headers for debugging
        raw_headers = {
            'subject': msg.get('Subject'),
            'from': msg.get('From'),
            'to': msg.get('To'),
            'date': msg.get('Date'),
            'message_id': msg.get('Message-ID'),
            'content_type': msg.get_content_type(),
        }
        
        return ParsedEmail(
            subject=subject,
            sender=sender,
            recipient=recipient,
            date=date,
            html_body=html_body,
            text_body=text_body,
            message_id=message_id,
            content_hash=content_hash,
            raw_headers=raw_headers,
        )
    
    def _clean_subject(self, subject: str) -> str:
        """Clean and normalize email subject."""
        if not subject:
            return ""
        
        # Remove common prefixes
        subject = re.sub(r'^(Re:|Fwd:|FW:)\s*', '', subject, flags=re.IGNORECASE)
        
        # Normalize whitespace
        subject = ' '.join(subject.split())
        
        return subject.strip()
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date header into datetime."""
        if not date_str:
            return None
        
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            # Try alternative parsing
            try:
                # Handle some common non-standard formats
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return None
    
    def _extract_bodies(self, msg: email.message.EmailMessage) -> tuple[Optional[str], Optional[str]]:
        """
        Extract HTML and plain text bodies from email.
        Handles multipart messages correctly.
        """
        html_body = None
        text_body = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                if content_type == 'text/html' and html_body is None:
                    html_body = self._decode_payload(part)
                elif content_type == 'text/plain' and text_body is None:
                    text_body = self._decode_payload(part)
        else:
            content_type = msg.get_content_type()
            payload = self._decode_payload(msg)
            
            if content_type == 'text/html':
                html_body = payload
            else:
                text_body = payload
        
        return html_body, text_body
    
    def _decode_payload(self, part: email.message.EmailMessage) -> Optional[str]:
        """Decode email part payload to string."""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return None
            
            # Try to decode with charset from headers, fall back to utf-8
            charset = part.get_content_charset() or 'utf-8'
            try:
                return payload.decode(charset)
            except (UnicodeDecodeError, LookupError):
                # Fall back to utf-8 with error handling
                return payload.decode('utf-8', errors='replace')
        except Exception:
            return None
    
    def _generate_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for deduplication."""
        if not content:
            return ""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Convenience function for quick parsing
def parse_eml(file_path: str) -> ParsedEmail:
    """Quick helper to parse an .eml file."""
    parser = EmailParser()
    return parser.parse_eml_file(file_path)


def parse_mbox(file_path: str, limit: int = None):
    """
    Generator that yields ParsedEmail objects from an .mbox file.
    
    Args:
        file_path: Path to the .mbox file
        limit: Optional limit on number of emails to parse (for testing)
    
    Yields:
        ParsedEmail objects
    """
    import mailbox
    
    parser = EmailParser()
    mbox = mailbox.mbox(file_path)
    
    count = 0
    for message in mbox:
        try:
            # Convert mailbox message to email.message.EmailMessage
            parsed = parser._extract_email_data(message)
            yield parsed
            
            count += 1
            if limit and count >= limit:
                break
        except Exception as e:
            # Log error but continue processing
            print(f"Error parsing message {count}: {e}")
            continue
    
    mbox.close()
