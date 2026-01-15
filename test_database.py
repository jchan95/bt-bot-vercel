from app.database import supabase

def test_database_connection():
    """Test that we can connect to Supabase and query tables."""
    print("Testing database connection...")
    
    try:
        # Try to query the issues table (should be empty)
        response = supabase.table("stratechery_issues").select("*").limit(1).execute()
        
        print(f"✓ Connected to Supabase successfully!")
        print(f"✓ Found {len(response.data)} issues in database")
        
        # Check if pgvector extension is enabled
        result = supabase.rpc("match_distillations", {
            "query_embedding": [0.0] * 1536,
            "match_count": 1
        }).execute()
        
        print(f"✓ pgvector extension is working!")
        print(f"✓ Vector search functions are ready!")
        print("\n✅ Database setup complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("\nMake sure you ran the SQL schema in Supabase SQL Editor!")
        return False


if __name__ == "__main__":
    test_database_connection()