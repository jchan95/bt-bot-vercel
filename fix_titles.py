from email.header import decode_header
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

def decode_mime_header(text):
    if not text or '=?' not in text:
        return text
    try:
        decoded_parts = decode_header(text)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                result.append(part.decode(charset or 'utf-8', errors='replace'))
            else:
                result.append(part)
        return ''.join(result)
    except:
        return text

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Fetch ALL issues
all_issues = []
offset = 0
while True:
    result = supabase.table("stratechery_issues")\
        .select("issue_id, title")\
        .range(offset, offset + 999)\
        .execute()
    if not result.data:
        break
    all_issues.extend(result.data)
    if len(result.data) < 1000:
        break
    offset += 1000

print(f"Total issues: {len(all_issues)}")

fixed = 0
for issue in all_issues:
    title = issue.get("title", "")
    if "=?" in title:
        new_title = decode_mime_header(title)
        if new_title != title:
            print(f"Fixing: {new_title[:60]}...")
            supabase.table("stratechery_issues")\
                .update({"title": new_title})\
                .eq("issue_id", issue["issue_id"])\
                .execute()
            fixed += 1

print(f"\nDone! Fixed {fixed} titles.")
