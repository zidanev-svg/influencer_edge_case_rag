"""
Process Influencer Pricing Edge Cases, create summaries with ChatGPT, and upload to Pinecone
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import openai
from pinecone import Pinecone, ServerlessSpec

# Dummy API Keys (replace with actual keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize APIs
openai.api_key = OPENAI_API_KEY

def initialize_pinecone():
    """Initialize Pinecone client and create index if needed"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = pc.list_indexes()
    if INDEX_NAME not in [idx['name'] for idx in existing_indexes]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENVIRONMENT
            )
        )
    
    return pc.Index(INDEX_NAME)

def load_influencer_documents(data_dir: str = "fake_data") -> List[Dict]:
    """Load all influencer pricing edge case JSON documents"""
    documents = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        with open(json_file, 'r') as f:
            doc = json.load(f)
            doc['source_file'] = json_file.name
            documents.append(doc)
    
    print(f"Loaded {len(documents)} influencer documents")
    return documents

def create_summary_with_chatgpt(influencer_data: Dict) -> Dict[str, str]:
    """
    Use ChatGPT to create multiple types of summaries for edge case pricing
    """
    # Prepare the influencer data for summarization
    influencer_text = json.dumps(influencer_data, indent=2)
    
    summaries = {}
    
    # 1. Edge Case Analysis Summary
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing influencer pricing and identifying edge cases in brand partnerships."
                },
                {
                    "role": "user",
                    "content": f"Analyze this influencer pricing edge case and explain the key concerns in 2-3 sentences:\n\n{influencer_text}"
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        summaries['edge_case_summary'] = response.choices[0].message.content.strip()
    except Exception as e:
        summaries['edge_case_summary'] = f"Error generating summary: {str(e)}"
    
    # 2. Pricing Risk Assessment
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a pricing strategist for influencer marketing campaigns."
                },
                {
                    "role": "user",
                    "content": f"What are the main pricing risks and concerns for this influencer partnership?\n\n{influencer_text}"
                }
            ],
            max_tokens=300,
            temperature=0.4
        )
        summaries['pricing_risk'] = response.choices[0].message.content.strip()
    except Exception as e:
        summaries['pricing_risk'] = f"Error assessing pricing risk: {str(e)}"
    
    # 3. Brand Concerns Analysis
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand manager evaluating influencer partnerships."
                },
                {
                    "role": "user",
                    "content": f"What questions should a brand ask before agreeing to this deal?\n\n{influencer_text}"
                }
            ],
            max_tokens=350,
            temperature=0.5
        )
        summaries['brand_concerns'] = response.choices[0].message.content.strip()
    except Exception as e:
        summaries['brand_concerns'] = f"Error analyzing brand concerns: {str(e)}"
    
    # 4. Recommendation
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an influencer marketing consultant providing strategic advice."
                },
                {
                    "role": "user",
                    "content": f"What recommendations would you give to brands considering this edge case pricing scenario?\n\n{influencer_text}"
                }
            ],
            max_tokens=300,
            temperature=0.4
        )
        summaries['recommendations'] = response.choices[0].message.content.strip()
    except Exception as e:
        summaries['recommendations'] = f"Error generating recommendations: {str(e)}"
    
    return summaries

def create_embedding(text: str) -> List[float]:
    """Create embedding vector using OpenAI"""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        # Return dummy embedding for testing
        return [0.0] * 1536

def prepare_metadata(influencer_data: Dict, summaries: Dict) -> Dict:
    """Prepare metadata for Pinecone storage with all influencer features"""
    # Get age distribution as string for metadata
    age_dist_str = json.dumps(influencer_data.get('age_distribution', {}))
    country_dist_str = json.dumps(influencer_data.get('country_distribution', {}))
    
    metadata = {
        # Core identifiers
        'influencer_id': influencer_data.get('influencer_id', 'unknown'),
        'source_file': influencer_data.get('source_file', 'unknown'),
        
        # Creator info
        'platform': influencer_data.get('platform', 'unknown'),
        'content_type': influencer_data.get('content_type', 'unknown'),
        'creator_gender': influencer_data.get('creator_gender', 'unknown'),
        'primary_language': influencer_data.get('primary_language', 'unknown'),
        'creator_country': influencer_data.get('creator_country', 'unknown'),
        'creator_age_years': influencer_data.get('creator_age_years', 0),
        
        # Content category
        'content_category': influencer_data.get('content_category', 'unknown'),
        'content_subcategory': influencer_data.get('content_subcategory', 'unknown'),
        
        # Audience demographics
        'audience_female_pct': influencer_data.get('audience_female_pct', 0.0),
        'audience_male_pct': influencer_data.get('audience_male_pct', 0.0),
        'audience_non_binary_pct': influencer_data.get('audience_non_binary_pct', 0.0),
        'dominant_age_group': influencer_data.get('dominant_age_group', 'unknown'),
        'female_dominant_flag': influencer_data.get('female_dominant_flag', 0),
        
        # Engagement metrics
        'reach': influencer_data.get('reach', 0),
        'followers': influencer_data.get('followers', 0),
        'engagement_rate_pct': influencer_data.get('engagement_rate_pct', 0.0),
        'engagement_weighted_followers': influencer_data.get('engagement_weighted_followers', 0),
        'reach_to_follower_ratio': influencer_data.get('reach_to_follower_ratio', 0.0),
        
        # Market metrics
        'high_value_market_pct': influencer_data.get('high_value_market_pct', 0.0),
        'audience_concentration_score': influencer_data.get('audience_concentration_score', 0.0),
        'platform_count': influencer_data.get('platform_count', 0),
        'niche_specificity_score': influencer_data.get('niche_specificity_score', 0.0),
        'age_alignment_score': influencer_data.get('age_alignment_score', 0.0),
        
        # Pricing info
        'currency': influencer_data.get('currency', 'USD'),
        'rate_amount': influencer_data.get('rate_amount', 0.0),
        'rate_per_1k_followers': influencer_data.get('rate_per_1k_followers', 0.0),
        'rate_per_engagement_point': influencer_data.get('rate_per_engagement_point', 0.0),
        'confidence_score': influencer_data.get('confidence_score', 0.0),
        
        # Edge case info
        'edge_case_notes': influencer_data.get('edge_case_notes', ''),
        'edge_case_questions': '; '.join(influencer_data.get('edge_case_questions', [])),
        
        # AI-generated summaries
        'edge_case_summary': summaries.get('edge_case_summary', ''),
        'pricing_risk': summaries.get('pricing_risk', ''),
        'brand_concerns': summaries.get('brand_concerns', ''),
        'recommendations': summaries.get('recommendations', ''),
        
        # Distributions (as strings)
        'age_distribution': age_dist_str[:500],  # Truncate if needed
        'country_distribution': country_dist_str[:500]
    }
    
    return metadata

def upload_to_pinecone(index, influencer_data: Dict, summaries: Dict):
    """Upload document with embeddings and metadata to Pinecone"""
    influencer_id = influencer_data.get('influencer_id', 'unknown')
    
    # Create text for embedding (combine key fields)
    embedding_text = f"""
    Influencer: {influencer_data.get('influencer_id')}
    Platform: {influencer_data.get('platform')}
    Content: {influencer_data.get('content_type')} - {influencer_data.get('content_category')} - {influencer_data.get('content_subcategory')}
    Creator: {influencer_data.get('creator_gender')}, Age {influencer_data.get('creator_age_years')}, {influencer_data.get('creator_country')}
    Audience: {influencer_data.get('followers')} followers, {influencer_data.get('engagement_rate_pct')}% engagement
    Pricing: {influencer_data.get('currency')} {influencer_data.get('rate_amount')}
    Edge Case: {influencer_data.get('edge_case_notes', '')}
    Questions: {'; '.join(influencer_data.get('edge_case_questions', []))}
    Analysis: {summaries.get('edge_case_summary', '')}
    """
    
    # Create embedding
    embedding = create_embedding(embedding_text)
    
    # Prepare metadata
    metadata = prepare_metadata(influencer_data, summaries)
    
    # Upload to Pinecone
    try:
        index.upsert(
            vectors=[
                {
                    'id': influencer_id,
                    'values': embedding,
                    'metadata': metadata
                }
            ]
        )
        print(f"✅ Uploaded {influencer_id} to Pinecone")
    except Exception as e:
        print(f"❌ Error uploading {influencer_id}: {str(e)}")

def process_all_documents():
    """Main processing pipeline"""
    print("="*60)
    print("Influencer Pricing Edge Case Processing Pipeline")
    print("="*60)
    print()
    
    # Step 1: Load documents
    print("Step 1: Loading influencer edge case documents...")
    documents = load_influencer_documents()
    print()
    
    # Step 2: Initialize Pinecone
    print("Step 2: Initializing Pinecone...")
    try:
        index = initialize_pinecone()
        print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {str(e)}")
        print("Continuing with processing (summaries will still be generated)")
        index = None
    print()
    
    # Step 3: Process each document
    print("Step 3: Processing documents...")
    print()
    
    for i, doc in enumerate(documents, 1):
        influencer_id = doc.get('influencer_id', 'unknown')
        rate = doc.get('rate_amount', 0)
        currency = doc.get('currency', 'USD')
        
        print(f"[{i}/{len(documents)}] Processing {influencer_id} - {currency} {rate}")
        
        # Generate summaries with ChatGPT
        print(f"  - Generating AI summaries...")
        summaries = create_summary_with_chatgpt(doc)
        
        # Upload to Pinecone
        if index:
            print(f"  - Uploading to Pinecone...")
            upload_to_pinecone(index, doc, summaries)
        
        # Save summaries to file for review
        output_file = f"processed_{influencer_id}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'original_data': doc,
                'ai_summaries': summaries
            }, f, indent=2)
        print(f"  - Saved processed data to {output_file}")
        print()
    
    print("="*60)
    print("✅ Processing complete!")
    print(f"Processed {len(documents)} edge case documents")
    print("="*60)

if __name__ == "__main__":
    # Run the pipeline
    process_all_documents()
