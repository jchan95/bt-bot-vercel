from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Fetch ALL issues (paginate if needed)
all_issues = []
offset = 0
batch_size = 1000

while True:
    result = supabase.table("stratechery_issues")\
        .select("issue_id, title")\
        .range(offset, offset + batch_size - 1)\
        .execute()
    
    if not result.data:
        break
    
    all_issues.extend(result.data)
    print(f"Fetched {len(all_issues)} so far...")
    
    if len(result.data) < batch_size:
        break
    
    offset += batch_size

print(f"\nTotal issues: {len(all_issues)}")

needs_fix = [r for r in all_issues if "=?" in r.get("title", "")]
print(f"Titles needing fix: {len(needs_fix)}")

if needs_fix:
    print(f"\nExamples:")
    for r in needs_fix[:3]:
        print(f"  {r['title'][:70]}...")
