import os
import json
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
INDEX_NAME = os.environ.get("INDEX_NAME", "influencer-pricing-edge-cases")

index = pc.Index(INDEX_NAME)

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        query_text = body.get('query', '')
        top_k = body.get('top_k', 3)
        
        if not query_text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Query text is required'})
            }
        
        embedding = create_embedding(query_text)
        
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results['matches']:
            matches.append({
                'influencer_id': match['metadata'].get('influencer_id'),
                'similarity_score': match['score'],
                'platform': match['metadata'].get('platform'),
                'content_type': match['metadata'].get('content_type'),
                'category': match['metadata'].get('content_category'),
                'subcategory': match['metadata'].get('content_subcategory'),
                'creator_country': match['metadata'].get('creator_country'),
                'rate_amount': match['metadata'].get('rate_amount'),
                'currency': match['metadata'].get('currency'),
                'followers': match['metadata'].get('followers'),
                'engagement_rate_pct': match['metadata'].get('engagement_rate_pct'),
                'edge_case_notes': match['metadata'].get('edge_case_notes'),
                'edge_case_questions': match['metadata'].get('edge_case_questions')
            })
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'query': query_text,
                'matches': matches,
                'count': len(matches)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
