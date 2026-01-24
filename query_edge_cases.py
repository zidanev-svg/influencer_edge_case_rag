import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "influencer-pricing-edge-cases"

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def query_similar_cases(query_text, top_k=3):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    embedding = create_embedding(query_text)
    
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results['matches']

def display_results(matches):
    print("\n" + "="*80)
    print("TOP MATCHES:")
    print("="*80)
    
    for i, match in enumerate(matches, 1):
        meta = match['metadata']
        score = match['score']
        
        print(f"\n{i}. {meta.get('influencer_id')} - Similarity: {score:.2%}")
        print(f"   Platform: {meta.get('platform')} | Content: {meta.get('content_type')}")
        print(f"   Category: {meta.get('content_category')} - {meta.get('content_subcategory')}")
        print(f"   Rate: {meta.get('currency')} ${meta.get('rate_amount'):,}")
        print(f"   Followers: {meta.get('followers'):,} | Engagement: {meta.get('engagement_rate_pct')}%")
        print(f"   Edge Case: {meta.get('edge_case_notes')}")
        print(f"   Questions: {meta.get('edge_case_questions')[:150]}...")

if __name__ == "__main__":
    print("Enter your edge case query (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    query = "\n".join(lines)
    
    if not query.strip():
        print("No query entered.")
        exit()
    
    print("\nQuerying for similar edge cases...")
    matches = query_similar_cases(query, top_k=3)
    display_results(matches)
