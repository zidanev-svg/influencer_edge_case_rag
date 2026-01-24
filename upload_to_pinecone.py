import os
import json
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "influencer-pricing-edge-cases"

def load_json_files(directory="mock_data"):
    documents = []
    for json_file in Path(directory).glob("*.json"):
        with open(json_file, 'r') as f:
            documents.append(json.load(f))
    return documents

def create_summary_text(data):
    edge_type = "High price" if data.get("rate_amount", 0) > 10000 else "Low price"
    questions = "; ".join(data.get("edge_case_questions", []))
    
    summary = f"""{edge_type} edge case
Platform: {data.get('platform')} {data.get('content_type')}
Category: {data.get('content_category')} - {data.get('content_subcategory')}
Creator Country: {data.get('creator_country')}
Rate: {data.get('currency')} ${data.get('rate_amount')} for {data.get('followers')} followers
Engagement: {data.get('engagement_rate_pct')}%
Concern: {data.get('edge_case_notes')}
Questions: {questions}"""
    
    return summary

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def prepare_metadata(data):
    metadata = {
        'influencer_id': data.get('influencer_id'),
        'platform': data.get('platform'),
        'content_type': data.get('content_type'),
        'creator_gender': data.get('creator_gender'),
        'primary_language': data.get('primary_language'),
        'creator_country': data.get('creator_country'),
        'creator_age_years': data.get('creator_age_years'),
        'content_category': data.get('content_category'),
        'content_subcategory': data.get('content_subcategory'),
        'audience_female_pct': data.get('audience_female_pct'),
        'audience_male_pct': data.get('audience_male_pct'),
        'dominant_age_group': data.get('dominant_age_group'),
        'reach': data.get('reach'),
        'followers': data.get('followers'),
        'engagement_rate_pct': data.get('engagement_rate_pct'),
        'rate_amount': data.get('rate_amount'),
        'rate_per_1k_followers': data.get('rate_per_1k_followers'),
        'currency': data.get('currency'),
        'edge_case_notes': data.get('edge_case_notes'),
        'confidence_score': data.get('confidence_score'),
    }
    
    questions = data.get('edge_case_questions', [])
    if questions:
        metadata['edge_case_questions'] = "; ".join(questions)
    
    return metadata

def delete_all_vectors():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    print("Deleting all existing vectors...")
    index.delete(delete_all=True)
    print("✅ All vectors deleted")

def upload_to_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    delete_all_vectors()
    
    documents = load_json_files()
    print(f"Loaded {len(documents)} documents")
    
    for doc in documents:
        influencer_id = doc.get('influencer_id')
        print(f"Processing {influencer_id}...")
        
        summary = create_summary_text(doc)
        embedding = create_embedding(summary)
        metadata = prepare_metadata(doc)
        
        index.upsert(vectors=[{
            'id': influencer_id,
            'values': embedding,
            'metadata': metadata
        }])
        
        print(f"✅ Uploaded {influencer_id}")
    
    print(f"\nDone! Uploaded {len(documents)} vectors to Pinecone")

if __name__ == "__main__":
    upload_to_pinecone()
