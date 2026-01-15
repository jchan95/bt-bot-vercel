import re
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

def clean_article_text(text):
    if not text:
        return text
    
    # Remove lines that are just separators like | | | [---] | |
    text = re.sub(r'^[\|\s\[\]\-]+$', '', text, flags=re.MULTILINE)
    
    # Remove the pattern at start: | | | [---] | | # 
    text = re.sub(r'^[\s\|\[\]\-\n]*#', '#', text)
    
    # Remove personalized greetings: #### Hi John M Chan, #### Your subs...
    text = re.sub(r'#{1,4}\s*Hi\s+[^#\n]+#{1,4}[^\n]*\n?', '', text)
    
    # Remove "Your subscription" lines
    text = re.sub(r'^.*Your subs[^\n]*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove #### Hi lines
    text = re.sub(r'^####\s*Hi[^\n]*\n?', '', text, flags=re.MULTILINE)
    
    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Fetch ALL issues
all_issues = []
offset = 0
while True:
    result = supabase.table("stratechery_issues")\
        .select("issue_id, title, cleaned_text")\
        .range(offset, offset + 999)\
        .execute()
    if not result.data:
        break
    all_issues.extend(result.data)
    print(f"Fetched {len(all_issues)}...")
    if len(result.data) < 1000:
        break
    offset += 1000

print(f"\nTotal issues: {len(all_issues)}")

# Find ones needing fixes
needs_fix = []
for issue in all_issues:
    text = issue.get("cleaned_text", "") or ""
    if text[:300].find('| | |') >= 0 or \
       text[:300].find('[---]') >= 0 or \
       text[:300].find('#### Hi') >= 0 or \
       text.strip().startswith('|'):
        needs_fix.append(issue)

print(f"Articles needing text fix: {len(needs_fix)}")

if needs_fix:
    print("\nFixing...")
    fixed = 0
    for issue in needs_fix:
        old_text = issue.get("cleaned_text", "")
        new_text = clean_article_text(old_text)
        
        if new_text != old_text:
            print(f"  Fixing: {issue.get('title', 'Unknown')[:50]}...")
            supabase.table("stratechery_issues")\
                .update({"cleaned_text": new_text})\
                .eq("issue_id", issue["issue_id"])\
                .execute()
            fixed += 1
    
    print(f"\nDone! Fixed {fixed} articles.")
else:
    print("No articles need fixing!")
